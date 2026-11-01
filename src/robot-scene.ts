import * as THREE from 'three';
import type { RoomViewer } from './viewer';
import { createDemoReplay, sampleReplay, carriedPose, stretchLift, stretchTip, STRETCH_ARM, type Landmark, type Replay, type RobotId } from './replay';
import type { Vec3 as _V } from './types';
import type { SavedObjectLayer } from './saved-objects';
import { EYE_FORWARD, EYE_HEIGHT } from './pov';

const dark=new THREE.MeshStandardMaterial({color:0x272e30,roughness:.65,metalness:.35});
const silver=new THREE.MeshStandardMaterial({color:0xb7bec0,roughness:.32,metalness:.75});
const rubber=new THREE.MeshStandardMaterial({color:0x161c1c,roughness:.95});
const green=new THREE.MeshStandardMaterial({color:0x28bc88,emissive:0x096a45,emissiveIntensity:.7});
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
function box(parent:THREE.Object3D,size:number[],position:number[],material=dark){const m=new THREE.Mesh(new THREE.BoxGeometry(...size as [number,number,number]),material);m.position.set(...position as [number,number,number]);parent.add(m);return m;}
function cylinder(parent:THREE.Object3D,r:number,length:number,position:number[],material=dark){const m=new THREE.Mesh(new THREE.CylinderGeometry(r,r,length,20),material);m.rotation.x=Math.PI/2;m.position.set(...position as [number,number,number]);parent.add(m);return m;}
function dispose(root:THREE.Object3D){root.traverse(o=>{if(o instanceof THREE.Sprite){o.material.map?.dispose();o.material.dispose();return}if(o instanceof THREE.Mesh||o instanceof THREE.Line){o.geometry.dispose();const materials=Array.isArray(o.material)?o.material:[o.material];materials.forEach(m=>{if(![dark,silver,rubber,green].includes(m))m.dispose()})}});root.clear();}
export class RobotScene {
  root=new THREE.Group(); route=new THREE.Group(); reach=new THREE.Group(); robot=new THREE.Group();
  arm=new THREE.Group(); legs:THREE.Object3D[]=[]; replay:Replay=createDemoReplay('stretch'); follow=false;
  private previous:THREE.Vector3|null=null;
  private hasReplay=true; private pose:{position:THREE.Vector3;yaw:number}|null=null;
  eye=new THREE.Group(); private savedVisibility:{route:boolean;reach:boolean;eye:boolean}|null=null;
  hasEnvironmentColliders=false;
  private carriedLayer:SavedObjectLayer|null=null;
  /** Saved SimReady meshes that replay.carried entries move (pick, carry at the gripper, place). */
  setCarriedObjects(layer:SavedObjectLayer|null){this.carriedLayer?.resetPoses();this.carriedLayer=layer;if(this.hasReplay&&this.pose)this.update(this.lastTime)}
  private lastTime=0;
  private updateCarried(time:number,frame:{position:number[];yaw:number;armExtension:number}){
    const layer=this.carriedLayer,carried=this.replay.robot==='stretch'?this.replay.carried??[]:[];
    if(this.replay.robot==='stretch')this.arm.position.z=STRETCH_ARM.height;
    if(!layer)return;
    if(!carried.length){layer.resetPoses();return}
    const origins:Record<string,_V>={};for(const c of carried){const rest=layer.restPosition(c.objectId);if(rest)origins[c.objectId]=rest.toArray() as _V}
    const lift=stretchLift(carried,time,frame.position[2],origins);this.arm.position.z=lift;
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
  private buildRobot(id:RobotId){dispose(this.robot);this.arm=new THREE.Group();this.legs=[];this.eye=new THREE.Group();
    if(id==='stretch'){
      box(this.robot,[.48,.53,.19],[0,0,.19]);box(this.robot,[.42,.45,.035],[0,0,.3],silver);
      for(const x of [-.255,.255]){const wheel=cylinder(this.robot,.13,.08,[x,0,.14],rubber);wheel.rotation.z=Math.PI/2;}
      box(this.robot,[.085,.095,1.05],[-.13,.05,.84]);box(this.robot,[.025,.025,.98],[-.19,.05,.85],silver);
      box(this.robot,[.19,.12,.075],[-.13,.05,1.39]);box(this.robot,[.05,.025,.025],[-.13,-.018,1.39],green);
      this.arm.position.set(-.13,.05,.98);this.robot.add(this.arm);box(this.arm,[.16,.14,.15],[0,0,0]);
      box(this.arm,[.1,.5,.07],[0,.24,0],silver);box(this.arm,[.15,.09,.08],[0,.5,0]);
      box(this.arm,[.025,.13,.025],[-.065,.58,0],silver);box(this.arm,[.025,.13,.025],[.065,.58,0],silver);
      box(this.robot,[.1,.015,.02],[0,-.273,.21],green);
    }else if(id==='go2'){
      box(this.robot,[.29,.57,.19],[0,0,.48],silver);box(this.robot,[.24,.14,.16],[0,.32,.47]);box(this.robot,[.12,.02,.03],[0,.397,.5],green);
      for(const x of [-.19,.19])for(const y of [-.2,.2]){const leg=new THREE.Group();leg.position.set(x,y,.45);box(leg,[.065,.08,.24],[0,.035,-.1]);box(leg,[.045,.06,.24],[0,.06,-.3],silver);cylinder(leg,.055,.06,[0,.06,-.4],rubber);this.robot.add(leg);this.legs.push(leg);}
    }else{
      box(this.robot,[.32,.32,.07],[0,0,.06]);cylinder(this.robot,.1,.15,[0,0,.16],silver);this.robot.add(this.arm);this.arm.position.z=.24;
      const lower=box(this.arm,[.08,.09,.34],[0,.06,.16],silver);lower.rotation.x=-.35;
      const upper=box(this.arm,[.07,.32,.07],[0,.2,.33]);upper.rotation.x=.15;
      box(this.arm,[.12,.06,.08],[0,.38,.3],silver);box(this.arm,[.025,.12,.035],[-.05,.43,.3]);box(this.arm,[.025,.12,.035],[.05,.43,.3]);
    }
    const halo=new THREE.Mesh(new THREE.RingGeometry(.32,.35,48),new THREE.MeshBasicMaterial({color:0x12936d,transparent:true,opacity:.65,side:THREE.DoubleSide}));halo.position.z=.015;halo.userData.decoration=true;this.robot.add(halo);
    // Eye marker at the POV camera origin (0.12 m forward along local +Y, eye height above origin).
    this.eye.position.set(0,EYE_FORWARD,EYE_HEIGHT[id]);const lens=new THREE.Mesh(new THREE.SphereGeometry(.028,16,12),new THREE.MeshBasicMaterial({color:0x14d39a}));this.eye.add(lens);
    const ring=new THREE.Mesh(new THREE.TorusGeometry(.04,.008,8,24),new THREE.MeshBasicMaterial({color:0x253236}));ring.rotation.x=Math.PI/2;this.eye.add(ring);this.robot.add(this.eye);
  }
  update(time:number){if(!this.hasReplay)return;const frame=sampleReplay(this.replay,time);const p=new THREE.Vector3(...frame.position);
    if(this.follow&&this.previous){const d=p.clone().sub(this.previous);this.viewer.camera.position.add(d);this.viewer.orbit.target.add(d)}
    this.previous=p.clone();this.robot.position.copy(p);this.robot.rotation.z=frame.yaw;this.pose={position:p.clone(),yaw:frame.yaw};
    if(this.replay.robot==='stretch')this.arm.scale.y=.65+frame.armExtension;
    this.lastTime=time;this.updateCarried(time,frame);
    if(this.replay.robot==='soarm')this.arm.rotation.z=this.replay.source==='demo'?Math.sin(time*.3)*.55:0;
    const nearby=sampleReplay(this.replay,Math.min(this.replay.duration,time+.02));
    const moving=this.replay.source==='demo'&&Math.hypot(nearby.position[0]-frame.position[0],nearby.position[1]-frame.position[1])>1e-5;
    this.legs.forEach((leg,i)=>leg.rotation.x=moving?Math.sin(time*6+(i===0||i===3?0:Math.PI))*.22:0);
  }
  setLayer(layer:string){this.viewer.setLayers(layer!=='collisions',layer==='collisions'&&!this.hasEnvironmentColliders,layer==='collisions'&&this.hasEnvironmentColliders);this.reach.visible=layer==='reach'&&this.hasReplay&&this.replay.source==='demo';}
  home(){if(this.hasEnvironmentColliders)this.viewer.home();else{this.viewer.orbit.target.set(1.15,0,.65);this.viewer.camera.position.set(6.15,-6.7,6.65);this.viewer.orbit.update()}this.previous=null;}
  top(){this.viewer.top();}
  focus(){if(!this.hasReplay)return;this.viewer.orbit.target.copy(this.robot.position).add(new THREE.Vector3(0,0,.65));this.viewer.camera.position.copy(this.viewer.orbit.target).add(new THREE.Vector3(1.3,-1.7,2.6));this.viewer.orbit.update();}
  dispose(){this.viewer.scene.remove(this.root);dispose(this.root);}
}
