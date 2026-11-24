import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import type { PhysicsModel, PhysicsPose, RobotId } from './replay';

/**
 * Real robot meshes exported from MuJoCo Menagerie (scripts/export-robot-models.py): one GLB per robot in
 * metres, Z-up, local +Y forward, lowest point at z=0, with one named node per MJCF body so joints can be posed.
 */
export type JointInfo={name:string;type:'hinge'|'slide';axis:number[];anchor:number[];home:number;range:number[]|null};
export type RobotMeta={robot:string;liftZ:number;bounds:{min:number[];max:number[]};bodies:Record<string,{parent:string;position:number[];free?:boolean;joint?:JointInfo}>};
type VisualRobotId=RobotId|PhysicsModel;
export const MODEL_URLS:Record<VisualRobotId,{glb:string;meta:string}>={
  stretch:{glb:new URL('./robots/stretch.glb',import.meta.url).href,meta:new URL('./robots/stretch.json',import.meta.url).href},
  go2:{glb:new URL('./robots/go2.glb',import.meta.url).href,meta:new URL('./robots/go2.json',import.meta.url).href},
  'go2-arm':{glb:new URL('./robots/go2-arm.glb',import.meta.url).href,meta:new URL('./robots/go2-arm.json',import.meta.url).href},
  'g1-12dof':{glb:new URL('./robots/g1-12dof.glb',import.meta.url).href,meta:new URL('./robots/g1-12dof.json',import.meta.url).href},
  'g1-hands':{glb:new URL('./robots/g1-hands.glb',import.meta.url).href,meta:new URL('./robots/g1-hands.json',import.meta.url).href},
  g1:{glb:new URL('./robots/g1.glb',import.meta.url).href,meta:new URL('./robots/g1.json',import.meta.url).href},
};
const cache=new Map<VisualRobotId,Promise<{scene:THREE.Group;meta:RobotMeta}>>();
export function loadRobotModel(id:VisualRobotId){
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
  /** The parent group supplies measured world position and heading. Apply remaining rotation
   * in the GLB and use the physics state for every articulated joint. */
  applyPhysicsPose(pose:PhysicsPose,heading:number,joints:string[]){
    const exportedRoot=this.root.getObjectByName('robot');
    const baseName=Object.keys(this.meta.bodies).find(name=>this.meta.bodies[name].free);
    const base=baseName&&this.root.getObjectByName(baseName);
    const secondaryHinges:Record<string,string>=this.meta.robot==='stretch'?
      {rubber_left_y:'rubber_tip_left',rubber_right_y:'rubber_tip_right'}:{};
    if(!base||!exportedRoot||joints.some(joint=>!this.has(joint)&&!secondaryHinges[joint])){
      throw new Error('Recorded joints do not match the robot visual model.');
    }
    const [w,x,y,z]=pose.quaternion;
    const world=new THREE.Quaternion(x,y,z,w);
    const inverseHeading=new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0,0,1),-heading);
    exportedRoot.position.set(0,0,0);
    exportedRoot.quaternion.copy(inverseHeading).multiply(world);
    base.position.set(0,0,0);
    base.quaternion.identity();
    joints.forEach((name,index)=>this.set(name,pose.joints[index],false));
    // The Stretch XML stacks pitch and roll at each rubber fingertip. The
    // first hinge is in the GLB sidecar. Compose the second after it.
    for(const [joint,body] of Object.entries(secondaryHinges)){
      const index=joints.indexOf(joint);
      if(index>=0)this.root.getObjectByName(body)?.rotateY(pose.joints[index]);
    }
  }
  has(joint:string){return this.joints.has(joint)}
  /** Set an absolute joint value; clamped to the MJCF range when one exists. */
  set(joint:string,value:number,clamp=true){
    const e=this.joints.get(joint);if(!e)return;
    const {node,joint:j,homePosition,homeQuaternion}=e;
    const v=clamp&&j.range?Math.min(j.range[1],Math.max(j.range[0],value)):value,delta=v-j.home;
    const axis=new THREE.Vector3().fromArray(j.axis).normalize();
    if(j.type==='slide'){node.position.copy(homePosition).add(axis.multiplyScalar(delta).applyQuaternion(homeQuaternion));return}
    const q=new THREE.Quaternion().setFromAxisAngle(axis,delta),anchor=new THREE.Vector3().fromArray(j.anchor);
    node.position.copy(homePosition).add(anchor.clone().sub(anchor.clone().applyQuaternion(q)).applyQuaternion(homeQuaternion));
    node.quaternion.copy(homeQuaternion).multiply(q);
  }
  /** Solve the G1 right palm against a world-space tool handle using the exported joint limits. */
  reachRightHand(target:THREE.Vector3){
    const wrist=this.root.getObjectByName('right_wrist_yaw_link');
    if(!wrist)return;
    const names=['right_wrist_pitch_joint','right_elbow_joint','right_shoulder_yaw_joint','right_shoulder_roll_joint','right_shoulder_pitch_joint'];
    const values=names.map(name=>this.joints.get(name)!.joint.home);
    names.forEach((name,i)=>this.set(name,values[i]));
    const palm=()=>wrist.localToWorld(new THREE.Vector3(.105,0,0));
    for(let pass=0;pass<24;pass++){
      if(palm().distanceToSquared(target)<.000004)break;
      names.forEach((name,i)=>{
        const {node,joint}=this.joints.get(name)!;
        const pivot=node.localToWorld(new THREE.Vector3().fromArray(joint.anchor));
        const axis=new THREE.Vector3().fromArray(joint.axis).applyQuaternion(node.getWorldQuaternion(new THREE.Quaternion())).normalize();
        const from=palm().sub(pivot).projectOnPlane(axis),to=target.clone().sub(pivot).projectOnPlane(axis);
        const angle=Math.atan2(axis.dot(from.clone().cross(to)),from.dot(to));
        values[i]=THREE.MathUtils.clamp(values[i]+angle,joint.range![0],joint.range![1]);this.set(name,values[i]);
      });
    }
  }
  /** Offset from the exported home value. */
  offset(joint:string,delta:number){const e=this.joints.get(joint);if(e)this.set(joint,e.joint.home+delta)}
}
