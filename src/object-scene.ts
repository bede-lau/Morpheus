import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { SplatEdit, SplatEditSdf, SplatEditSdfType, SplatEditRgbaBlendMode } from '@sparkjsdev/spark';
import { RoomViewer } from './viewer';

export type Selection = {position:[number,number,number];size:[number,number,number]};
export type Asset = {id:string;name:string;category:string;sourceUrl:string;revision:string;dimensions:[number,number,number];triangles:number;license:string};
export type Replacement = {id:string;assetId:string;position:number[];quaternion:number[];scale:number;selection:Selection;chosenBy:'manual'|'decisions';decision?:unknown};
export const assetUrl=(asset:Asset,file='model.glb')=>`/api/objects/assets/${encodeURIComponent(asset.id)}/${file}`;
export function fitScale(size:number[],dimensions:number[]){
 const scale=Math.max(...size)/Math.max(...dimensions);
 if(!Number.isFinite(scale)||scale<=0)throw new Error('Invalid object dimensions');
 return Math.max(.01,Math.min(100,scale));
}

export class ObjectScene {
 selection:Selection|null=null; picking=false;
 root=new THREE.Group(); helper=new THREE.Box3Helper(new THREE.Box3(),0xf2a848);
 preview:THREE.Group|null=null; previewAsset:Asset|null=null;
 replacements:{data:Replacement;mesh:THREE.Group;edit:SplatEdit}[]=[];
 private down={x:0,y:0};private generation=0;private disposed=false;
 constructor(public view:RoomViewer,private onSelect:(selection:Selection)=>void){
  view.scene.add(this.root,this.helper);this.helper.visible=false;
  const canvas=view.renderer.domElement;
  canvas.addEventListener('pointerdown',this.pointerDown);canvas.addEventListener('pointerup',this.pointerUp);
 }
 private pointerDown=(e:PointerEvent)=>{this.down={x:e.clientX,y:e.clientY}};
 private pointerUp=(e:PointerEvent)=>{
  if(!this.picking||Math.hypot(e.clientX-this.down.x,e.clientY-this.down.y)>5)return;
  const rect=this.view.renderer.domElement.getBoundingClientRect();
  const ray=new THREE.Raycaster();ray.setFromCamera(new THREE.Vector2((e.clientX-rect.left)/rect.width*2-1,-(e.clientY-rect.top)/rect.height*2+1),this.view.camera);
  const hit=this.view.visibleSplatHit(ray);
  if(!hit)return;
  const p=hit.point;
  const selection:Selection={position:[p.x,p.y,p.z],size:[.24,.24,.16]};
  this.setSelection(selection);this.picking=false;this.onSelect(selection);
 };
 setSelection(selection:Selection){
  this.selection=structuredClone(selection);
  this.helper.box.setFromCenterAndSize(new THREE.Vector3(...selection.position),new THREE.Vector3(...selection.size));
  this.helper.visible=true;
 }
 focus(){if(!this.selection)return;const center=new THREE.Vector3(...this.selection.position);this.view.orbit.target.copy(center);this.view.camera.position.copy(center).add(new THREE.Vector3(.5,-.7,.55));this.view.orbit.update()}
 capture(){
  if(!this.selection)throw new Error('Select the object first.');
  const canvas=this.view.renderer.domElement, camera=this.view.camera;
  const center=new THREE.Vector3(...this.selection.position),size=new THREE.Vector3(...this.selection.size);
  const corners=[];
  for(const x of [-1,1])for(const y of [-1,1])for(const z of [-1,1]){
   const p=center.clone().add(new THREE.Vector3(size.x*x/2,size.y*y/2,size.z*z/2));
   if(p.clone().applyMatrix4(camera.matrixWorldInverse).z>=-camera.near)throw new Error('Move the camera back so the complete selection is in front of it.');
   corners.push(p.project(camera));
  }
  const minX=Math.max(0,Math.min(...corners.map(p=>(p.x+1)/2*canvas.width))-16);
  const maxX=Math.min(canvas.width,Math.max(...corners.map(p=>(p.x+1)/2*canvas.width))+16);
  const minY=Math.max(0,Math.min(...corners.map(p=>(1-p.y)/2*canvas.height))-16);
  const maxY=Math.min(canvas.height,Math.max(...corners.map(p=>(1-p.y)/2*canvas.height))+16);
  if(maxX-minX<20||maxY-minY<20)throw new Error('Move closer and keep the selection visible.');
  const visible=this.helper.visible,previewVisible=this.preview?.visible;
  this.helper.visible=false;if(this.preview)this.preview.visible=false;
  try{
   this.view.renderer.render(this.view.scene,camera);
   const crop=document.createElement('canvas');crop.width=512;crop.height=Math.max(64,Math.round(512*(maxY-minY)/(maxX-minX)));if(crop.height>768){crop.width=Math.round(768*crop.width/crop.height);crop.height=768}
   crop.getContext('2d')!.drawImage(canvas,minX,minY,maxX-minX,maxY-minY,0,0,crop.width,crop.height);
   return crop.toDataURL('image/jpeg',.87);
  }finally{this.helper.visible=visible;if(this.preview)this.preview.visible=!!previewVisible}
 }
 async loadPreview(asset:Asset){
  if(!this.selection)throw new Error('Select an object first.');
  const token=++this.generation;
  const group=(await new GLTFLoader().loadAsync(assetUrl(asset))).scene;
  if(this.disposed||token!==this.generation){this.free(group);return null}
  if(this.preview){this.root.remove(this.preview);this.free(this.preview)}
  this.preview=group;this.previewAsset=asset;this.root.add(group);
  group.position.fromArray(this.selection.position);group.scale.setScalar(fitScale(this.selection.size,asset.dimensions));
  return group.scale.x;
 }
 fit(position:number[],rotation:number[],scale:number){if(!this.preview)return;this.preview.position.fromArray(position);this.preview.rotation.set(...rotation.map(v=>v*Math.PI/180) as [number,number,number]);this.preview.scale.setScalar(scale)}
 apply(chosenBy:'manual'|'decisions',decision?:unknown){
  if(!this.preview||!this.previewAsset||!this.selection||!this.view.splat)throw new Error('Preview an asset first.');
  const mesh=this.preview, selection=structuredClone(this.selection);
  const edit=new SplatEdit({rgbaBlendMode:SplatEditRgbaBlendMode.MULTIPLY,softEdge:0});
  const mask=new SplatEditSdf({type:SplatEditSdfType.BOX,opacity:0});
  mask.position.fromArray(selection.position);mask.scale.fromArray(selection.size).multiplyScalar(.5);edit.add(mask);edit.updateMatrixWorld(true);
  this.view.splat.edits=[...(this.view.splat.edits??[]),edit];this.view.splat.updateVersion();
  const q=mesh.quaternion;
  const data:Replacement={id:crypto.randomUUID(),assetId:this.previewAsset.id,position:mesh.position.toArray(),quaternion:[q.w,q.x,q.y,q.z],scale:mesh.scale.x,selection,chosenBy,decision};
  this.replacements.push({data,mesh,edit});this.preview=null;this.previewAsset=null;this.selection=null;this.helper.visible=false;
  return data;
 }
 undo(){const item=this.replacements.pop();if(!item)return;
  if(this.view.splat){this.view.splat.edits=(this.view.splat.edits??[]).filter(edit=>edit!==item.edit);this.view.splat.updateVersion()}
  this.root.remove(item.mesh);this.free(item.mesh);return item.data;
 }
 clearPreview(){this.generation++;if(this.preview){this.root.remove(this.preview);this.free(this.preview)}this.preview=null;this.previewAsset=null}
 private free(group:THREE.Object3D){group.traverse(o=>{if(o instanceof THREE.Mesh){o.geometry.dispose();for(const m of Array.isArray(o.material)?o.material:[o.material]){for(const value of Object.values(m))if(value instanceof THREE.Texture)value.dispose();m.dispose()}}})}
 dispose(){this.disposed=true;this.clearPreview();while(this.replacements.length)this.undo();this.view.scene.remove(this.root,this.helper);this.helper.dispose();this.view.renderer.domElement.removeEventListener('pointerdown',this.pointerDown);this.view.renderer.domElement.removeEventListener('pointerup',this.pointerUp)}
}
