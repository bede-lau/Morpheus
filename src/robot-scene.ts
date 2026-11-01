import * as THREE from 'three';
import type { RoomViewer } from './viewer';
import { createDemoReplay, sampleReplay, carriedPose, stretchLift, stretchTip, STRETCH_ARM, type Landmark, type Replay, type ReplayFrame, type RobotId } from './replay';
import type { Vec3 as _V } from './types';
import type { SavedObjectLayer } from './saved-objects';
import { EYE_FORWARD, EYE_HEIGHT } from './pov';
import { ArticulatedRobot, loadRobotModel } from './robot-model';

// Footprint disc shown until the real mesh (src/robots/*.glb, exported from MuJoCo Menagerie) has loaded.
const PLACEHOLDER_RADIUS:Record<RobotId,number>={stretch:.25,go2:.3,g1:.22};
// Stretch 3 telescoping arm: four 0.13 m segments (MJCF joint_arm_l0..l3 range).
const STRETCH_SEGMENT=.13;

const LANDMARK_COLORS:Record<string,number>={table:0xd08a2c,counter:0x2b7fb8,sink:0x2b9bb8,tall:0x7a6bb0,door:0x0a8967};
function labelSprite(text:string,color:number){
  const canvas=typeof document!=='undefined'?document.createElement('canvas'):null;const ctx=canvas?.getContext('2d');
  if(!canvas||!ctx)return null;
  const font='600 34px "DM Sans",system-ui,sans-serif';ctx.font=font;const w=Math.ceil(ctx.measureText(text).width)+36,h=56;canvas.width=w;canvas.height=h;
  ctx.font=font;ctx.fillStyle='rgba(252,252,249,.92)';ctx.strokeStyle='#'+color.toString(16).padStart(6,'0');ctx.lineWidth=4;
  ctx.beginPath();ctx.roundRect(2,2,w-4,h-4,14);ctx.fill();ctx.stroke();ctx.fillStyle='#253236';ctx.textBaseline='middle';ctx.fillText(text,18,h/2+1);
  const texture=new THREE.CanvasTexture(canvas);texture.colorSpace=THREE.SRGBColorSpace;
  const sprite=new THREE.Sprite(new THREE.SpriteMaterial({map:texture,depthTest:false,depthWrite:false,transparent:true}));
  const height=.16;sprite.scale.set(height*w/h,height,1);sprite.renderOrder=40;return sprite;
}
function dispose(root:THREE.Object3D){root.traverse(o=>{if(o.userData.shared)return;if(o instanceof THREE.Sprite){o.material.map?.dispose();o.material.dispose();return}if(o instanceof THREE.Mesh||o instanceof THREE.Line){o.geometry.dispose();const materials=Array.isArray(o.material)?o.material:[o.material];materials.forEach(m=>m.dispose())}});root.clear();}
export class RobotScene {
  root=new THREE.Group(); route=new THREE.Group(); reach=new THREE.Group(); robot=new THREE.Group();
  arm=new THREE.Group(); legs:THREE.Object3D[]=[]; replay:Replay=createDemoReplay('stretch'); follow=false;
  private previous:THREE.Vector3|null=null;
  private hasReplay=true; private pose:{position:THREE.Vector3;yaw:number}|null=null;
  eye=new THREE.Group(); private savedVisibility:{route:boolean;reach:boolean;eye:boolean}|null=null;
  hasEnvironmentColliders=false;
  private model:ArticulatedRobot|null=null; private modelToken=0; private lastPose:{frame:ReplayFrame;time:number;moving:boolean}|null=null; private lift=STRETCH_ARM.height;
  private carriedLayer:SavedObjectLayer|null=null;
  /** Saved SimReady meshes that replay.carried entries move (pick, carry at the gripper, place). */
  setCarriedObjects(layer:SavedObjectLayer|null){this.carriedLayer?.resetPoses();this.carriedLayer=layer;if(this.hasReplay&&this.pose)this.update(this.lastTime)}
  private lastTime=0;
  private updateCarried(time:number,frame:{position:number[];yaw:number;armExtension:number}){
    const layer=this.carriedLayer,carried=this.replay.robot==='stretch'?this.replay.carried??[]:[];
    if(this.replay.robot==='stretch')this.arm.position.z=STRETCH_ARM.height;this.lift=STRETCH_ARM.height;
    if(!layer)return;
    if(!carried.length){layer.resetPoses();return}
    const origins:Record<string,_V>={};for(const c of carried){const rest=layer.restPosition(c.objectId);if(rest)origins[c.objectId]=rest.toArray() as _V}
    const lift=stretchLift(carried,time,frame.position[2],origins);this.arm.position.z=lift;this.lift=lift;
    const tip=stretchTip(frame.position as _V,frame.yaw,frame.armExtension,lift);
    for(const c of carried){const rest=origins[c.objectId];if(!rest)continue;layer.setPose(c.objectId,carriedPose(c,time,tip,rest).position)}
  }
  constructor(private viewer:RoomViewer){
    this.root.add(this.route,this.reach,this.robot);viewer.scene.add(this.root);
    viewer.scene.background=new THREE.Color('#efeeea');
    viewer.scene.children.filter(o=>o instanceof THREE.GridHelper).forEach(o=>o.visible=false);
    this.setReplay(this.replay);
    if(import.meta.env?.DEV&&typeof window!=='undefined')(window as unknown as {__robotScene?:RobotScene}).__robotScene=this; // dev-only hook for browser checks
  }
  /** Current robot body pose (Z-up, yaw 0 = +Y) or null when no replay is shown. */
  getPose(){return this.hasReplay&&this.pose?{position:this.pose.position.clone(),yaw:this.pose.yaw,robot:this.replay.robot}:null}
  /** Hide route, landmarks, reach disc and eye marker while rendering the robot POV. */
  setPovPass(active:boolean){
    if(active&&!this.savedVisibility){this.savedVisibility={route:this.route.visible,reach:this.reach.visible,eye:this.eye.visible};this.route.visible=false;this.reach.visible=false;this.eye.visible=false}
    else if(!active&&this.savedVisibility){const v=this.savedVisibility;this.savedVisibility=null;this.route.visible=v.route;this.reach.visible=v.reach;this.eye.visible=v.eye}
  }
  clearReplay(){this.carriedLayer?.resetPoses();this.pose=null;this.hasReplay=false;this.root.visible=false;this.previous=null;}
  setReplay(replay:Replay){this.hasReplay=true;this.root.visible=true;this.replay=replay;this.previous=null;this.buildRobot(replay.robot);dispose(this.route);dispose(this.reach);
    const points=replay.frames.map(f=>new THREE.Vector3(f.position[0],f.position[1],f.position[2]+.025));
    const path=new THREE.Line(new THREE.BufferGeometry().setFromPoints(points),new THREE.LineDashedMaterial({color:0x077c5c,dashSize:.13,gapSize:.075,depthTest:true,transparent:true,opacity:.85}));path.computeLineDistances();path.renderOrder=10;this.route.add(path);
    [points[0],points.at(-1)!].forEach(p=>{const dot=new THREE.Mesh(new THREE.RingGeometry(.08,.13,32),new THREE.MeshBasicMaterial({color:0x0a8967,side:THREE.DoubleSide,depthTest:true}));dot.position.copy(p);dot.renderOrder=11;this.route.add(dot)});
    const zone=new THREE.Mesh(new THREE.CylinderGeometry(.95,.95,.02,64),new THREE.MeshBasicMaterial({color:0x31b18a,transparent:true,opacity:.18,depthWrite:false}));zone.rotation.x=Math.PI/2;zone.position.copy(points.at(-1)!).add(new THREE.Vector3(0,0,.8));this.reach.add(zone);this.reach.visible=false;this.addLandmarks(replay.landmarks??[]);this.update(0);
  }
  private addLandmarks(landmarks:Landmark[]){
    // Tall clusters are mostly the room walls; drawing them boxes the whole kitchen.
    for(const l of landmarks.filter(l=>l.kind!=='tall'&&l.kind!=='object')){const color=LANDMARK_COLORS[l.kind]??0x5e6e6a;const group=new THREE.Group();group.position.set(...l.position);group.userData.landmark=l.id;
      const size=l.size.map(v=>Math.max(.02,Math.abs(v))) as [number,number,number];
      const body=new THREE.Mesh(new THREE.BoxGeometry(...size),new THREE.MeshBasicMaterial({color,transparent:true,opacity:.14,depthWrite:false}));body.renderOrder=5;group.add(body);
      const edges=new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(...size)),new THREE.LineBasicMaterial({color,transparent:true,opacity:.8}));edges.renderOrder=6;group.add(edges);
      const label=labelSprite(l.label,color);if(label){label.position.set(0,0,size[2]/2+.14);group.add(label)}
      this.route.add(group);}
  }
  private buildRobot(id:RobotId){dispose(this.robot);this.arm=new THREE.Group();this.legs=[];this.eye=new THREE.Group();this.model=null;const token=++this.modelToken;
    if(id==='stretch'){this.arm.position.set(STRETCH_ARM.mount[0],STRETCH_ARM.mount[1],STRETCH_ARM.height);this.robot.add(this.arm);}
    const placeholder=new THREE.Mesh(new THREE.CylinderGeometry(PLACEHOLDER_RADIUS[id],PLACEHOLDER_RADIUS[id],.02,32),new THREE.MeshBasicMaterial({color:0x8fa39c,transparent:true,opacity:.35}));placeholder.rotation.x=Math.PI/2;placeholder.position.z=.02;this.robot.add(placeholder);
    const halo=new THREE.Mesh(new THREE.RingGeometry(.32,.35,48),new THREE.MeshBasicMaterial({color:0x12936d,transparent:true,opacity:.65,side:THREE.DoubleSide}));halo.position.z=.015;halo.userData.decoration=true;this.robot.add(halo);
    // Eye marker at the POV camera origin (0.12 m forward along local +Y, eye height above origin).
    this.eye.position.set(0,EYE_FORWARD,EYE_HEIGHT[id]);const lens=new THREE.Mesh(new THREE.SphereGeometry(.028,16,12),new THREE.MeshBasicMaterial({color:0x14d39a}));this.eye.add(lens);
    const ring=new THREE.Mesh(new THREE.TorusGeometry(.04,.008,8,24),new THREE.MeshBasicMaterial({color:0x253236}));ring.rotation.x=Math.PI/2;this.eye.add(ring);this.robot.add(this.eye);
    // Real meshes (metres, Z-up, +Y forward, lowest point at z=0) replace the placeholder once loaded. Node tests have no DOM/fetch.
    if(typeof document==='undefined')return;
    loadRobotModel(id).then(({scene,meta})=>{
      if(token!==this.modelToken)return;
      const clone=scene.clone(true);this.robot.remove(placeholder);placeholder.geometry.dispose();placeholder.material.dispose();this.robot.add(clone);
      this.model=new ArticulatedRobot(clone,meta);if(this.lastPose)this.poseModel(this.lastPose.frame,this.lastPose.time,this.lastPose.moving);
    }).catch(e=>console.warn(`Robot model for ${id} unavailable; showing the footprint only.`,e));
  }
  /** Joint animation on the loaded mesh: gait while moving, arm extension and lift from the replay. */
  private poseModel(frame:ReplayFrame,time:number,moving:boolean){
    const m=this.model;if(!m)return;const id=this.replay.robot,phase=time*7;
    if(id==='go2'){
      for(const [leg,offset] of [['FL',0],['RR',0],['FR',Math.PI],['RL',Math.PI]] as const){const s=moving?Math.sin(phase+offset):0;m.offset(`${leg}_thigh_joint`,-.28*s);m.offset(`${leg}_calf_joint`,.38*Math.max(0,s));}
    }else if(id==='g1'){
      const ext=frame.armExtension;
      for(const [side,offset] of [['left',0],['right',Math.PI]] as const){const s=moving?Math.sin(phase*.75+offset):0;
        m.offset(`${side}_hip_pitch_joint`,-.3*s);m.offset(`${side}_knee_joint`,.5*Math.max(0,-s)+.04);m.offset(`${side}_ankle_pitch_joint`,-.18*Math.max(0,-s));
        m.offset(`${side}_shoulder_pitch_joint`,(moving?.28*Math.sin(phase*.75+offset+Math.PI):0)-1.25*ext);m.offset(`${side}_elbow_joint`,-.7*ext);}
    }else if(id==='stretch'){
      for(const j of ['joint_arm_l0','joint_arm_l1','joint_arm_l2','joint_arm_l3'])m.set(j,STRETCH_SEGMENT*frame.armExtension);
      // joint_lift is the arm height above the base minus the arm's height at joint zero (sidecar position at the exported home value).
      const arm=m.meta.bodies.link_arm_l4;if(arm?.joint)m.set('joint_lift',this.lift-(arm.position[2]-arm.joint.home));
      const roll=moving?time*5:0;m.offset('joint_left_wheel',roll);m.offset('joint_right_wheel',roll);
    }
  }
  update(time:number){if(!this.hasReplay)return;const frame=sampleReplay(this.replay,time);const p=new THREE.Vector3(...frame.position);
    if(this.follow&&this.previous){const d=p.clone().sub(this.previous);this.viewer.camera.position.add(d);this.viewer.orbit.target.add(d)}
    this.previous=p.clone();this.robot.position.copy(p);this.robot.rotation.z=frame.yaw;this.pose={position:p.clone(),yaw:frame.yaw};
    this.lastTime=time;this.updateCarried(time,frame);
    const nearby=sampleReplay(this.replay,Math.min(this.replay.duration,time+.02));
    const moving=this.replay.source==='demo'&&Math.hypot(nearby.position[0]-frame.position[0],nearby.position[1]-frame.position[1])>1e-5;
    this.lastPose={frame,time,moving};this.poseModel(frame,time,moving);
  }
  setLayer(layer:string){this.viewer.setLayers(layer!=='collisions',layer==='collisions'&&!this.hasEnvironmentColliders,layer==='collisions'&&this.hasEnvironmentColliders);this.reach.visible=layer==='reach'&&this.hasReplay&&this.replay.source==='demo';}
  home(){if(this.hasEnvironmentColliders)this.viewer.home();else{this.viewer.orbit.target.set(1.15,0,.65);this.viewer.camera.position.set(6.15,-6.7,6.65);this.viewer.orbit.update()}this.previous=null;}
  top(){this.viewer.top();}
  focus(){if(!this.hasReplay)return;this.viewer.orbit.target.copy(this.robot.position).add(new THREE.Vector3(0,0,.65));this.viewer.camera.position.copy(this.viewer.orbit.target).add(new THREE.Vector3(1.3,-1.7,2.6));this.viewer.orbit.update();}
  dispose(){this.viewer.scene.remove(this.root);dispose(this.root);}
}
