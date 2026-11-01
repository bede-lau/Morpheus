import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { SplatEdit, SplatEditRgbaBlendMode, SplatEditSdf, SplatEditSdfType, type SplatMesh } from '@sparkjsdev/spark';
import type { RoomViewer } from './viewer';

/** Visual effects for the /?view=demo showpiece. Display-only: nothing here changes source splats or exported geometry. */
export const clamp01=(v:number)=>Math.max(0,Math.min(1,v));
export const easeOutCubic=(t:number)=>1-Math.pow(1-clamp01(t),3);
export const easeInOutCubic=(t:number)=>{t=clamp01(t);return t<.5?4*t*t*t:1-Math.pow(-2*t+2,3)/2};
export const backOut=(t:number)=>{t=clamp01(t);const c1=1.9,c3=c1+1;return 1+c3*Math.pow(t-1,3)+c1*Math.pow(t-1,2)};
export const progress=(t:number,start:number,duration:number)=>clamp01((t-start)/duration);
export const GREEN=0x19c37d, AMBER=0xf0a429, INK=0x111111;

function attachEdit(splat:SplatMesh,edit:SplatEdit){splat.edits=[...(splat.edits??[]),edit];splat.updateVersion()}
function detachEdit(splat:SplatMesh,edit:SplatEdit){splat.edits=(splat.edits??[]).filter(e=>e!==edit);splat.updateVersion()}

/** Growing-sphere reveal: opacity 0 OUTSIDE a world-space sphere (detached edit, like cutaway.ts). */
export function createSphereReveal(splat:SplatMesh,center:THREE.Vector3){
  const edit=new SplatEdit({name:'Morpheus demo reveal',rgbaBlendMode:SplatEditRgbaBlendMode.MULTIPLY,softEdge:.35});
  const sdf=new SplatEditSdf({type:SplatEditSdfType.SPHERE,opacity:0,invert:true,radius:.001});
  sdf.position.copy(center);edit.add(sdf);edit.updateMatrixWorld(true);attachEdit(splat,edit);let live=true;
  return {setRadius(r:number){if(!live)return;sdf.radius=Math.max(.001,r);edit.updateMatrixWorld(true);splat.updateVersion()},dispose(){if(!live)return;live=false;detachEdit(splat,edit);edit.remove(sdf)}};
}
/** Hide the original splats inside a replacement's reviewed selection box (same mask as object-scene.ts apply()). */
export function createBoxMask(splat:SplatMesh,selection:{position:number[];size:number[]}){
  const edit=new SplatEdit({rgbaBlendMode:SplatEditRgbaBlendMode.MULTIPLY,softEdge:0});
  const mask=new SplatEditSdf({type:SplatEditSdfType.BOX,opacity:0});mask.position.fromArray(selection.position);mask.scale.fromArray(selection.size).multiplyScalar(.5);edit.add(mask);edit.updateMatrixWorld(true);attachEdit(splat,edit);
  return ()=>detachEdit(splat,edit);
}

const gridVertex='varying vec2 vUv;void main(){vUv=uv;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}';
const gridFragment=`uniform vec2 size;uniform float alpha;uniform vec3 color;varying vec2 vUv;
float line(float v,float w){float d=abs(fract(v-.5)-.5)/fwidth(v);return 1.-min(d/w,1.);}
void main(){vec2 p=vUv*size;float g=max(line(p.x*4.,1.),line(p.y*4.,1.))*.55+max(line(p.x,1.2),line(p.y,1.2))*.45;
vec2 e=min(vUv,1.-vUv)*size;float edge=1.-smoothstep(0.,.06,min(e.x,e.y));
float a=(.10+g*.55+edge*.8)*alpha;gl_FragColor=vec4(color,a);}`;
/** Emissive green scan slab sweeping up through the room, with a bright scanline frame. */
export function createScanPlane(scene:THREE.Scene,min:THREE.Vector3,max:THREE.Vector3){
  const w=max.x-min.x,d=max.y-min.y,group=new THREE.Group();
  const material=new THREE.ShaderMaterial({uniforms:{size:{value:new THREE.Vector2(w,d)},alpha:{value:0},color:{value:new THREE.Color(GREEN)}},vertexShader:gridVertex,fragmentShader:gridFragment,transparent:true,depthWrite:false,side:THREE.DoubleSide});
  const plane=new THREE.Mesh(new THREE.PlaneGeometry(w,d),material);plane.renderOrder=30;group.add(plane);
  const frame=new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-w/2,-d/2,0),new THREE.Vector3(w/2,-d/2,0),new THREE.Vector3(w/2,d/2,0),new THREE.Vector3(-w/2,d/2,0)]),new THREE.LineBasicMaterial({color:GREEN,transparent:true,opacity:0}));frame.renderOrder=31;group.add(frame);
  group.position.set((min.x+max.x)/2,(min.y+max.y)/2,min.z);group.visible=false;scene.add(group);
  return {set(z:number,alpha:number){group.visible=alpha>.001;group.position.z=z;material.uniforms.alpha.value=alpha;(frame.material as THREE.LineBasicMaterial).opacity=alpha},
    dispose(){scene.remove(group);plane.geometry.dispose();material.dispose();frame.geometry.dispose();(frame.material as THREE.Material).dispose()}};
}
/** Pulsing double ring marker on a surface point. */
export function createRing(scene:THREE.Scene,position:number[],color:number){
  const group=new THREE.Group();group.position.fromArray(position);group.position.z+=.01;
  const mats:THREE.MeshBasicMaterial[]=[];
  const make=(inner:number,outer:number)=>{const m=new THREE.MeshBasicMaterial({color,transparent:true,opacity:0,side:THREE.DoubleSide,depthTest:false,depthWrite:false});mats.push(m);const mesh=new THREE.Mesh(new THREE.RingGeometry(inner,outer,48),m);mesh.renderOrder=60;group.add(mesh);return mesh};
  const core=make(.075,.095),pulse=make(.09,.11);const dot=new THREE.Mesh(new THREE.CircleGeometry(.025,20),new THREE.MeshBasicMaterial({color,transparent:true,opacity:0,depthTest:false}));mats.push(dot.material as THREE.MeshBasicMaterial);dot.renderOrder=61;group.add(dot);
  scene.add(group);
  return {group,update(t:number,visible:number){const k=(t*1.1)%1;pulse.scale.setScalar(1+k*2.2);(pulse.material as THREE.MeshBasicMaterial).opacity=visible*(1-k)*.9;(core.material as THREE.MeshBasicMaterial).opacity=visible;(dot.material as THREE.MeshBasicMaterial).opacity=visible;core.scale.setScalar(.6+.4*easeOutCubic(visible))},
    dispose(){scene.remove(group);group.traverse(o=>{if(o instanceof THREE.Mesh)o.geometry.dispose()});mats.forEach(m=>m.dispose())}};
}
/** Expanding white/green flash ring for the SimReady swap. */
export function createFlash(scene:THREE.Scene,position:number[]){
  const m=new THREE.MeshBasicMaterial({color:0xffffff,transparent:true,opacity:0,side:THREE.DoubleSide,depthTest:false,depthWrite:false});
  const g=new THREE.MeshBasicMaterial({color:GREEN,transparent:true,opacity:0,side:THREE.DoubleSide,depthTest:false,depthWrite:false});
  const a=new THREE.Mesh(new THREE.RingGeometry(.8,1,64),m),b=new THREE.Mesh(new THREE.RingGeometry(.94,1,64),g);a.renderOrder=b.renderOrder=62;
  const group=new THREE.Group();group.add(a,b);group.position.fromArray(position);group.position.z+=.005;scene.add(group);
  return {update(p:number){const s=.05+easeOutCubic(p)*.45;a.scale.setScalar(s);b.scale.setScalar(s*1.25);m.opacity=p>0&&p<1?(1-p)*.95:0;g.opacity=p>0&&p<1?(1-p):0},
    dispose(){scene.remove(group);a.geometry.dispose();b.geometry.dispose();m.dispose();g.dispose()}};
}
export async function loadAsset(assetId:string){return (await new GLTFLoader().loadAsync(`/api/objects/assets/${encodeURIComponent(assetId)}/model.glb`)).scene}
export function freeObject(root:THREE.Object3D){root.traverse(o=>{if(o instanceof THREE.Mesh){o.geometry.dispose();for(const m of Array.isArray(o.material)?o.material:[o.material]){for(const v of Object.values(m))if(v instanceof THREE.Texture)v.dispose();m.dispose()}}})}
/** Project a world point to CSS pixels in the viewer canvas; null when behind the camera. */
export function project(viewer:RoomViewer,p:THREE.Vector3){
  const v=p.clone().project(viewer.camera);if(v.z>1||v.z<-1)return null;
  const el=viewer.renderer.domElement;return {x:(v.x+1)/2*el.clientWidth,y:(1-v.y)/2*el.clientHeight};
}
/** Render once and crop a square around a world point (used when no discovery view saw the object). */
export function captureAround(viewer:RoomViewer,p:THREE.Vector3,cssSize=180){
  const s=project(viewer,p);if(!s)return null;const el=viewer.renderer.domElement,r=el.width/el.clientWidth;
  viewer.renderer.render(viewer.scene,viewer.camera);
  const c=document.createElement('canvas');c.width=c.height=256;const size=cssSize*r;
  c.getContext('2d')!.drawImage(el,s.x*r-size/2,s.y*r-size/2,size,size,0,0,256,256);
  try{return c.toDataURL('image/jpeg',.85)}catch{return null}
}
/** Fade every collider box of the viewer (setBoxes materials) to a fraction of its normal opacity. */
export function fadeColliders(viewer:RoomViewer,k:number){
  for(const box of viewer.boxes.values()){const m=box.material as THREE.MeshBasicMaterial;if(m.userData.base===undefined)m.userData.base=m.opacity;m.opacity=m.userData.base*k*.6;
    for(const child of box.children)if(child instanceof THREE.LineSegments){const l=child.material as THREE.LineBasicMaterial;if(l.userData.base===undefined){l.userData.base=l.opacity;l.color.set(GREEN)}l.opacity=l.userData.base*k}}
}
