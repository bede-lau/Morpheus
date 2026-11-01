import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import type { RobotId } from './replay';

/**
 * Real robot meshes exported from MuJoCo Menagerie (scripts/export-robot-models.py): one GLB per robot in
 * metres, Z-up, local +Y forward, lowest point at z=0, with one named node per MJCF body so joints can be posed.
 */
export type JointInfo={name:string;type:'hinge'|'slide';axis:number[];anchor:number[];home:number;range:number[]|null};
export type RobotMeta={robot:string;liftZ:number;bounds:{min:number[];max:number[]};bodies:Record<string,{parent:string;position:number[];joint?:JointInfo}>};
export const MODEL_URLS:Record<RobotId,{glb:string;meta:string}>={
  stretch:{glb:new URL('./robots/stretch.glb',import.meta.url).href,meta:new URL('./robots/stretch.json',import.meta.url).href},
  go2:{glb:new URL('./robots/go2.glb',import.meta.url).href,meta:new URL('./robots/go2.json',import.meta.url).href},
  g1:{glb:new URL('./robots/g1.glb',import.meta.url).href,meta:new URL('./robots/g1.json',import.meta.url).href},
};
const cache=new Map<RobotId,Promise<{scene:THREE.Group;meta:RobotMeta}>>();
export function loadRobotModel(id:RobotId){
  let pending=cache.get(id);
  if(!pending){
    pending=(async()=>{
      const [gltf,meta]=await Promise.all([new GLTFLoader().loadAsync(MODEL_URLS[id].glb),fetch(MODEL_URLS[id].meta).then(r=>{if(!r.ok)throw new Error(`Robot metadata ${r.status}`);return r.json() as Promise<RobotMeta>})]);
      // Materials are shared between meshes: tone each one once for the viewer's strong lights.
      const toned=new Set<THREE.Material>();
      gltf.scene.traverse(o=>{if(o instanceof THREE.Mesh){o.userData.shared=true;const materials=Array.isArray(o.material)?o.material:[o.material];for(const m of materials)if(m instanceof THREE.MeshStandardMaterial&&!toned.has(m)){toned.add(m);m.color.multiplyScalar(.72);m.metalness=Math.min(m.metalness,.3);m.roughness=Math.max(m.roughness,.6)}}});
      return {scene:gltf.scene,meta};
    })();
    cache.set(id,pending);pending.catch(()=>cache.delete(id));
  }
  return pending;
}
/** Poses named MJCF bodies on a cloned model by joint value (radians or metres), relative to the exported home pose. */
export class ArticulatedRobot {
  private joints=new Map<string,{node:THREE.Object3D;joint:JointInfo;homePosition:THREE.Vector3;homeQuaternion:THREE.Quaternion}>();
  constructor(readonly root:THREE.Object3D,readonly meta:RobotMeta){
    for(const [body,info] of Object.entries(meta.bodies)){
      if(!info.joint)continue;const node=root.getObjectByName(body);if(!node)continue;
      this.joints.set(info.joint.name,{node,joint:info.joint,homePosition:node.position.clone(),homeQuaternion:node.quaternion.clone()});
    }
  }
  has(joint:string){return this.joints.has(joint)}
  /** Set an absolute joint value; clamped to the MJCF range when one exists. */
  set(joint:string,value:number){
    const e=this.joints.get(joint);if(!e)return;
    const {node,joint:j,homePosition,homeQuaternion}=e;
    const v=j.range?Math.min(j.range[1],Math.max(j.range[0],value)):value,delta=v-j.home;
    const axis=new THREE.Vector3().fromArray(j.axis).normalize();
    if(j.type==='slide'){node.position.copy(homePosition).add(axis.multiplyScalar(delta).applyQuaternion(homeQuaternion));return}
    const q=new THREE.Quaternion().setFromAxisAngle(axis,delta),anchor=new THREE.Vector3().fromArray(j.anchor);
    node.position.copy(homePosition).add(anchor.clone().sub(anchor.clone().applyQuaternion(q)).applyQuaternion(homeQuaternion));
    node.quaternion.copy(homeQuaternion).multiply(q);
  }
  /** Offset from the exported home value. */
  offset(joint:string,delta:number){const e=this.joints.get(joint);if(e)this.set(joint,e.joint.home+delta)}
}
