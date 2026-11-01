import { NativeReplayView } from './native-replay';
import type { PolicyReplayBundle } from '../web/policy-bridge';
import * as THREE from 'three';
import type { RoomViewer } from './viewer';
import { createDemoReplay, sampleReplay, type Replay, type RobotId, type PolicyRun } from './replay';

const dark=new THREE.MeshStandardMaterial({color:0x272e30,roughness:.65,metalness:.35});
const silver=new THREE.MeshStandardMaterial({color:0xb7bec0,roughness:.32,metalness:.75});
const rubber=new THREE.MeshStandardMaterial({color:0x161c1c,roughness:.95});
const green=new THREE.MeshStandardMaterial({color:0x28bc88,emissive:0x096a45,emissiveIntensity:.7});
function box(parent:THREE.Object3D,size:number[],position:number[],material=dark){const m=new THREE.Mesh(new THREE.BoxGeometry(...size as [number,number,number]),material);m.position.set(...position as [number,number,number]);parent.add(m);return m;}
function cylinder(parent:THREE.Object3D,r:number,length:number,position:number[],material=dark){const m=new THREE.Mesh(new THREE.CylinderGeometry(r,r,length,20),material);m.rotation.x=Math.PI/2;m.position.set(...position as [number,number,number]);parent.add(m);return m;}
function dispose(root:THREE.Object3D){root.traverse(o=>{if(o instanceof THREE.Mesh||o instanceof THREE.Line){o.geometry.dispose();const materials=Array.isArray(o.material)?o.material:[o.material];materials.forEach(m=>{if(![dark,silver,rubber,green].includes(m))m.dispose()})}});root.clear();}
export class RobotScene {
  root=new THREE.Group(); route=new THREE.Group(); alternatives=new THREE.Group(); cleaning=new THREE.Group(); reach=new THREE.Group(); robot=new THREE.Group();
  arm=new THREE.Group(); legs:THREE.Object3D[]=[]; replay:Replay=createDemoReplay('stretch'); follow=false;
  private native:NativeReplayView|null=null; private previous:THREE.Vector3|null=null; private intentMarker:THREE.Mesh|null=null; private intentPoints:THREE.Vector3[]=[];
  constructor(private viewer:RoomViewer){
    this.root.add(this.route,this.alternatives,this.cleaning,this.reach,this.robot);viewer.scene.add(this.root);
    viewer.scene.background=new THREE.Color('#efeeea');
    viewer.scene.children.filter(o=>o instanceof THREE.GridHelper).forEach(o=>o.visible=false);
    this.setReplay(this.replay);
  }
  setReplay(replay:Replay){this.replay=replay;this.previous=null;this.buildRobot(replay.robot);dispose(this.route);dispose(this.reach);
    const points=replay.frames.map(f=>new THREE.Vector3(f.position[0],f.position[1],f.position[2]+.025));
    const path=new THREE.Line(new THREE.BufferGeometry().setFromPoints(points),new THREE.LineDashedMaterial({color:0x077c5c,dashSize:.13,gapSize:.075,depthTest:true,transparent:true,opacity:.85}));path.computeLineDistances();path.renderOrder=10;this.route.add(path);
    [points[0],points.at(-1)!].forEach(p=>{const dot=new THREE.Mesh(new THREE.RingGeometry(.08,.13,32),new THREE.MeshBasicMaterial({color:0x0a8967,side:THREE.DoubleSide,depthTest:true}));dot.position.copy(p);dot.renderOrder=11;this.route.add(dot)});
    const zone=new THREE.Mesh(new THREE.CylinderGeometry(.95,.95,.02,64),new THREE.MeshBasicMaterial({color:0x31b18a,transparent:true,opacity:.18,depthWrite:false}));zone.rotation.x=Math.PI/2;zone.position.copy(points.at(-1)!).add(new THREE.Vector3(0,0,.8));this.reach.add(zone);this.reach.visible=false;this.update(0);
  }
  private buildRobot(id:RobotId){dispose(this.robot);this.arm=new THREE.Group();this.legs=[];
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
    }else if(id==='g1'){
      box(this.robot,[.26,.19,.32],[0,0,1.02],silver);box(this.robot,[.28,.19,.16],[0,0,.78]);
      box(this.robot,[.18,.17,.18],[0,0,1.31],silver);box(this.robot,[.13,.02,.055],[0,.092,1.32],rubber);box(this.robot,[.09,.01,.015],[0,.106,1.32],green);
      for(const x of [-.19,.19]){box(this.robot,[.07,.09,.28],[x,0,1.04],silver);box(this.robot,[.065,.07,.23],[x,.02,.8]);}
      for(const x of [-.09,.09]){const leg=new THREE.Group();leg.position.set(x,0,.72);box(leg,[.09,.12,.32],[0,0,-.15],silver);box(leg,[.075,.1,.31],[0,.015,-.45]);box(leg,[.1,.23,.055],[0,.055,-.69],rubber);this.robot.add(leg);this.legs.push(leg);}
    }else{
      box(this.robot,[.32,.32,.07],[0,0,.06]);cylinder(this.robot,.1,.15,[0,0,.16],silver);this.robot.add(this.arm);this.arm.position.z=.24;
      const lower=box(this.arm,[.08,.09,.34],[0,.06,.16],silver);lower.rotation.x=-.35;
      const upper=box(this.arm,[.07,.32,.07],[0,.2,.33]);upper.rotation.x=.15;
      box(this.arm,[.12,.06,.08],[0,.38,.3],silver);box(this.arm,[.025,.12,.035],[-.05,.43,.3]);box(this.arm,[.025,.12,.035],[.05,.43,.3]);
    }
    const halo=new THREE.Mesh(new THREE.RingGeometry(.32,.35,48),new THREE.MeshBasicMaterial({color:0x12936d,transparent:true,opacity:.65,side:THREE.DoubleSide}));halo.position.z=.015;halo.userData.decoration=true;this.robot.add(halo);
  }
  setPolicyRun(run:PolicyRun|null,selectedId:string,cleaningId:string){
    dispose(this.alternatives);dispose(this.cleaning);this.intentPoints=[];this.intentMarker=null;
    if(!run)return;
    const palette=[0x1577a0,0xb56a22,0x8263a8,0x008e67];
    run.routes.forEach((route,i)=>{
      const selected=route.id===selectedId,points=route.points.map(p=>new THREE.Vector3(p[0],p[1],p[2]+.045));
      const material=new THREE.LineDashedMaterial({color:route.kind==='learned'?0x008e67:palette[i%palette.length],transparent:true,opacity:selected?1:.42,dashSize:selected?.17:.08,gapSize:selected?.03:.1,depthTest:false});
      const line=new THREE.Line(new THREE.BufferGeometry().setFromPoints(points),material);line.computeLineDistances();line.renderOrder=20;this.alternatives.add(line);
      if(selected&&points.length>1){const curve=new THREE.CurvePath<THREE.Vector3>();for(let j=1;j<points.length;j++)curve.add(new THREE.LineCurve3(points[j-1],points[j]));const tube=new THREE.Mesh(new THREE.TubeGeometry(curve,Math.min(500,points.length*4),.014,6,false),new THREE.MeshBasicMaterial({color:material.color,transparent:true,opacity:.75,depthTest:false}));tube.renderOrder=21;this.alternatives.add(tube);}
    });
    const intent=run.cleaning.find(c=>c.id===cleaningId);
    if(intent){
      this.intentPoints=intent.points.map(p=>new THREE.Vector3(...p));
      const trace=new THREE.Line(new THREE.BufferGeometry().setFromPoints(this.intentPoints),new THREE.LineDashedMaterial({color:0xd57b24,dashSize:.045,gapSize:.035,depthTest:false,transparent:true,opacity:.85}));trace.computeLineDistances();trace.renderOrder=25;this.cleaning.add(trace);
      this.intentMarker=new THREE.Mesh(new THREE.SphereGeometry(.045,16,12),new THREE.MeshBasicMaterial({color:0xf0a448,depthTest:false}));this.intentMarker.renderOrder=26;this.cleaning.add(this.intentMarker);
      for(let i=0;i<this.intentPoints.length;i+=Math.max(1,Math.floor(this.intentPoints.length/10))){const patch=new THREE.Mesh(new THREE.RingGeometry(.055,.075,24),new THREE.MeshBasicMaterial({color:0xcb7b2a,transparent:true,opacity:.55,side:THREE.DoubleSide,depthTest:false}));patch.position.copy(this.intentPoints[i]);patch.renderOrder=24;this.cleaning.add(patch);}
    }
  }
  setNativeReplay(bundle:PolicyReplayBundle|null){if(this.native){this.root.remove(this.native.root);this.native.dispose();this.native=null;}this.robot.visible=!bundle;this.alternatives.visible=!bundle;this.cleaning.visible=!bundle;this.route.visible=!bundle;this.reach.visible=false;this.previous=null;if(bundle){this.native=new NativeReplayView(bundle);this.root.add(this.native.root);this.update(0);}}
  update(time:number){if(this.native){const p=this.native.update(time);if(this.follow&&this.previous){const d=p.clone().sub(this.previous);this.viewer.camera.position.add(d);this.viewer.orbit.target.add(d);}this.previous=p;return;}const frame=sampleReplay(this.replay,time);const p=new THREE.Vector3(...frame.position);
    if(this.follow&&this.previous){const d=p.clone().sub(this.previous);this.viewer.camera.position.add(d);this.viewer.orbit.target.add(d)}
    this.previous=p.clone();this.robot.position.copy(p);this.robot.rotation.z=frame.yaw;
    if(this.replay.robot==='stretch')this.arm.scale.y=.65+frame.armExtension;
    if(this.replay.robot==='soarm')this.arm.rotation.z=Math.sin(time*.3)*.55;
    const nearby=sampleReplay(this.replay,Math.min(this.replay.duration,time+.02));
    const moving=Math.hypot(nearby.position[0]-frame.position[0],nearby.position[1]-frame.position[1])>1e-5;
    this.legs.forEach((leg,i)=>leg.rotation.x=moving?Math.sin(time*6+(i===0||i===3?0:Math.PI))*.22:0);
    if(this.intentMarker&&this.intentPoints.length){const t=(time/this.replay.duration)*(this.intentPoints.length-1),i=Math.floor(t);this.intentMarker.position.copy(this.intentPoints[i]).lerp(this.intentPoints[Math.min(i+1,this.intentPoints.length-1)],t-i);}

  }
  setLayer(layer:string){this.viewer.setLayers(layer!=='collisions',false,layer==='collisions');this.reach.visible=layer==='reach';}
  home(){this.viewer.orbit.target.set(1.15,0,.65);this.viewer.camera.position.set(6.15,-6.7,6.65);this.viewer.orbit.update();this.previous=null;}
  top(){this.viewer.top();}
  focus(){
    if(this.native){
      const bounds=new THREE.Box3().setFromPoints(this.native.bundle.samples.map(s=>new THREE.Vector3(...s.body_poses!['robot/pelvis'].slice(0,3) as [number,number,number])));
      const center=bounds.getCenter(new THREE.Vector3()),size=bounds.getSize(new THREE.Vector3());center.z=.65;
      const span=Math.max(2.8,size.x+1.5,size.y+1.5);this.viewer.orbit.target.copy(center);this.viewer.camera.position.copy(center).add(new THREE.Vector3(span*.8,-span,span*.9));
    }else{this.viewer.orbit.target.copy(this.robot.position).add(new THREE.Vector3(0,0,.65));this.viewer.camera.position.copy(this.viewer.orbit.target).add(new THREE.Vector3(1.3,-1.7,2.6));}
    this.viewer.orbit.update();
  }
  dispose(){this.setNativeReplay(null);this.viewer.scene.remove(this.root);dispose(this.root);}
}
