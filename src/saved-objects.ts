import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { SplatEdit, SplatEditSdf, SplatEditSdfType, SplatEditRgbaBlendMode } from '@sparkjsdev/spark';
import type { BackendEnvironment } from './backend';
import type { RoomViewer } from './viewer';
import type { Replacement, Asset } from './object-scene';

type SavedReplacement=Replacement&{asset:Asset};
export type SavedObjects={revision:number;replacements:SavedReplacement[]};
const record=(v:unknown):v is Record<string,any>=>!!v&&typeof v==='object'&&!Array.isArray(v);
const vector=(v:unknown,n:number):v is number[]=>Array.isArray(v)&&v.length===n&&v.every(Number.isFinite);
export function parseSavedObjects(data:unknown,environment:BackendEnvironment):SavedObjects{
 if(!record(data)||data.version!==1||data.kind!=='simready-replacements'||data.units!=='m'||data.upAxis!=='Z'||!Number.isInteger(data.revision)||data.revision<0||!Array.isArray(data.replacements)||data.interactionValidated!==false)throw new Error('Unsupported saved object scene.');
 if(!data.replacements.length)return {revision:data.revision,replacements:[]};
 const source=data.source,expected=environment.visual.sourceSha256;
 if(!expected||!record(source)||source.sha256!==expected)throw new Error('Saved objects belong to a different or unverified source scan.');
 if(!record(source.transform)||source.transform.matrixOrder!=='column-major'||!vector(source.transform.matrix,16)||source.transform.matrix.some((v:number,i:number)=>Math.abs(v-environment.visual.transform.matrix[i])>1e-6))throw new Error('Saved objects use a different room alignment.');
 const ids=new Set<string>();
 for(const item of data.replacements){
  if(!record(item)||typeof item.id!=='string'||ids.has(item.id)||typeof item.assetId!=='string'||!/^[-A-Za-z0-9_]+$/.test(item.assetId)||!record(item.asset)||item.asset.id!==item.assetId||!vector(item.position,3)||!vector(item.quaternion,4)||Math.abs(Math.hypot(...item.quaternion)-1)>1e-5||!Number.isFinite(item.scale)||item.scale<.01||item.scale>100||!record(item.selection)||!vector(item.selection.position,3)||!vector(item.selection.size,3)||item.selection.size.some((v:number)=>v<=0||v>5)||!['manual','decisions'].includes(item.chosenBy))throw new Error('Invalid saved object pose or removal region.');
  ids.add(item.id);
 }
 return {revision:data.revision,replacements:data.replacements as SavedReplacement[]};
}
function free(root:THREE.Object3D){root.traverse(o=>{if(o instanceof THREE.Mesh){o.geometry.dispose();for(const material of Array.isArray(o.material)?o.material:[o.material]){for(const value of Object.values(material))if(value instanceof THREE.Texture)value.dispose();material.dispose()}}})}
export class SavedObjectLayer{
 root=new THREE.Group();private edits:SplatEdit[]=[];private generation=0;
 private items:{id:string;mesh:THREE.Object3D;position:THREE.Vector3;quaternion:THREE.Quaternion}[]=[];
 /** Loaded mesh by replacement id or load order index. */
 getMesh(key:string|number){return (typeof key==='number'?this.items[key]:this.items.find(item=>item.id===key))?.mesh??null}
 /** Saved (scan) pose of a replacement, Z-up world. */
 restPosition(id:string){const item=this.items.find(i=>i.id===id);return item?item.position.clone():null}
 /** Move a replacement mesh in world space (the scan mask stays at the saved location). */
 setPose(id:string,position:THREE.Vector3|number[],quaternion?:THREE.Quaternion){const item=this.items.find(i=>i.id===id);if(!item)return false;if(item.mesh.parent!==this.root)this.detach(id);
  if(Array.isArray(position))item.mesh.position.fromArray(position);else item.mesh.position.copy(position);if(quaternion)item.mesh.quaternion.copy(quaternion);return true}
 /** Parent a replacement to another object, keeping its world transform. */
 attach(id:string,parent:THREE.Object3D){const mesh=this.getMesh(id);if(!mesh)return false;parent.attach(mesh);return true}
 /** Return a replacement to the layer root, keeping its world transform. */
 detach(id:string){const mesh=this.getMesh(id);if(!mesh)return false;this.root.attach(mesh);return true}
 /** Put every replacement back at its saved pose. */
 resetPoses(){for(const item of this.items){this.root.attach(item.mesh);item.mesh.position.copy(item.position);item.mesh.quaternion.copy(item.quaternion)}}
 constructor(private view:RoomViewer){view.scene.add(this.root)}
 async load(scene:SavedObjects){
  const token=++this.generation;
  const results=await Promise.allSettled(scene.replacements.map(async item=>{
   const mesh=(await new GLTFLoader().loadAsync(`/api/objects/assets/${encodeURIComponent(item.assetId)}/model.glb`)).scene;
   mesh.position.fromArray(item.position);mesh.quaternion.set(item.quaternion[1],item.quaternion[2],item.quaternion[3],item.quaternion[0]);mesh.scale.setScalar(item.scale);
   const edit=new SplatEdit({rgbaBlendMode:SplatEditRgbaBlendMode.MULTIPLY,softEdge:0});
   const mask=new SplatEditSdf({type:SplatEditSdfType.BOX,opacity:0});mask.position.fromArray(item.selection.position);mask.scale.fromArray(item.selection.size).multiplyScalar(.5);edit.add(mask);edit.updateMatrixWorld(true);
   return {mesh,edit,id:item.id};
  }));
  const loaded=results.flatMap(result=>result.status==='fulfilled'?[result.value]:[]);
  if(token!==this.generation||results.some(result=>result.status==='rejected')){loaded.forEach(({mesh})=>free(mesh));if(token===this.generation)throw new Error('A saved SimReady mesh could not load.');return;}
  this.clear();
  for(const {mesh,edit,id} of loaded){this.root.add(mesh);this.edits.push(edit);this.items.push({id,mesh,position:mesh.position.clone(),quaternion:mesh.quaternion.clone()})}
  if(this.view.splat){this.view.splat.edits=[...(this.view.splat.edits??[]),...this.edits];this.view.splat.updateVersion()}
 }
 private clear(){for(const item of this.items)if(item.mesh.parent!==this.root){item.mesh.parent?.remove(item.mesh);this.root.add(item.mesh)}this.items=[];if(this.view.splat){this.view.splat.edits=(this.view.splat.edits??[]).filter(edit=>!this.edits.includes(edit));this.view.splat.updateVersion()}this.edits=[];free(this.root);this.root.clear()}
 dispose(){this.generation++;this.clear();this.view.scene.remove(this.root)}
}
