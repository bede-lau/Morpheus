import * as T from 'three';
import {sampleReplay, type Replay, type Wipe} from './replay';
import type {ArticulatedRobot} from './robot-model';

const v=(x=0,y=0,z=0)=>new T.Vector3(x,y,z);
const ease=(x:number)=>{x=Math.max(0,Math.min(1,x));return x*x*(3-2*x)};
const up=v(0,0,1);
function solid(geo:T.BufferGeometry,color:number){
  return new T.Mesh(geo,new T.MeshStandardMaterial({color,roughness:.85}));
}
function rod(mesh:T.Mesh,a:T.Vector3,b:T.Vector3){
  const delta=b.clone().sub(a);
  mesh.position.copy(a).lerp(b,.5);mesh.scale.y=delta.length();
  mesh.quaternion.setFromUnitVectors(v(0,1,0),delta.normalize());
}

// The scan identifies a sink counter but has no basin collider. Render the shallow
// training basin explicitly, with its inside floor directly under the pad.
export function trainingBasin(point:number[],towards:T.Vector3,overGlass=false){
  const basin=new T.Group();
  basin.name='Simulated sink basin';
  basin.position.fromArray(point);
  basin.rotation.z=Math.atan2(towards.y,towards.x);
  const panel=(width:number,length:number,height:number,x:number,y:number,z:number)=>{
    const mesh=solid(new T.BoxGeometry(width,length,height),0x91a9af);
    mesh.position.set(x,y,z);basin.add(mesh);
  };
  panel(.26,.49,.012,0,0,-.017);
  for(const sign of [-1,1]){
    panel(.012,.49,.052,sign*.124,0,.009);
    panel(.26,.012,.052,0,sign*.239,.009);
  }
  const drain=solid(new T.CircleGeometry(.018,24),0x34444b);
  drain.position.set(.073,.15,-.0105);basin.add(drain);
  const y=overGlass?0:.18;
  const spout=new T.CatmullRomCurve3([v(.12,y,.032),v(.12,y,.16),v(.10,y,.23),v(.035,y,.23),v(overGlass?0:.035,y,.19)]);
  basin.add(solid(new T.TubeGeometry(spout,24,.009,10,false),0x718a91));
  return basin;
}

/** Scripted contact demonstration. The folded pad and its grip move as one tool. */
export class Cleaning {
  root=new T.Group();
  cloth=new T.Group();
  private grip=solid(new T.CylinderGeometry(.019,.019,1,12),0x303d46);
  private wristLoop=solid(new T.TorusGeometry(.032,.006,8,24),0x353537);
  private boom=solid(new T.CylinderGeometry(.026,.036,1,12),0xe7bb59);
  private piston=solid(new T.CylinderGeometry(.017,.017,1,12),0x687780);
  private patches:{step:Wipe;side:T.Vector3;spots:T.Mesh[]}[]=[];
  constructor(private replay:Replay){
    this.root.add(this.cloth,this.boom,this.piston);
    const pad=solid(new T.BoxGeometry(.155,.12,.022),0xff8141);
    const seam=solid(new T.BoxGeometry(.148,.014,.004),0xffe7aa);
    seam.position.set(0,.035,.012);this.cloth.add(pad,seam,this.grip);
    if(replay.robot==='g1')this.cloth.add(this.wristLoop);
    for(const x of [-.068,-.060,.060,.068]){
      const hem=solid(new T.BoxGeometry(.002,.114,.002),0xf9ba86);
      hem.position.set(x,0,.0115);this.cloth.add(hem);
    }
    this.cloth.visible=!!replay.wipes?.length;
    const table=replay.landmarks?.find(l=>l.fixture==='practice-table');
    if(table){
      const [x,y,z]=table.position,[width,depth,height]=table.size;
      const top=z+height/2;
      const board=solid(new T.BoxGeometry(width,depth,.026),0xbb8b60);
      board.position.set(x,y,top-.013);this.root.add(board);
      for(const dx of [-1,1])for(const dy of [-1,1]){
        const leg=solid(new T.BoxGeometry(.024,.024,height-.026),0x414a4b);
        leg.position.set(x+dx*(width/2-.035),y+dy*(depth/2-.035),z-.013);this.root.add(leg);
      }
    }
    for(const step of replay.wipes??[]){
      const base=sampleReplay(replay,step.start).position;
      const towards=v(step.point[0]-base[0],step.point[1]-base[1],0).normalize();
      const side=v(-towards.y,towards.x,0);
      if(step.target==='sink')this.root.add(trainingBasin(step.point,towards));
      const spots=[-.09,-.045,0,.045,.09].map(offset=>{
        const spot=solid(new T.CircleGeometry(.034,15),0x694625);
        spot.material.transparent=true;
        spot.scale.x=1.4;spot.position.fromArray(step.point).addScaledVector(side,offset);
        spot.position.z-=.008;this.root.add(spot);return spot;
      });
      this.patches.push({step,side,spots});
    }
    this.boom.visible=this.piston.visible=replay.robot==='go2'&&replay.cleaningKit===true;
  }

  get robotId(){return this.replay.robot;}

  update(time:number,robot:T.Group,model:ArticulatedRobot|null){
    const patches=this.patches;
    if(!patches.length){
      this.reach(robot,null,robot.localToWorld(v(.15,.25,.65)),false);
      return;
    }
    const index=Math.max(0,patches.filter(p=>p.step.start<=time).length-1);
    const {step,side}=patches[index];
    const elapsed=time-step.start;
    const last=index===patches.length-1;
    const start=v(...step.point).addScaledVector(side,-.09);
    const nav=sampleReplay(this.replay,time);
    const carry=robot.localToWorld(this.replay.robot==='stretch'?v(.47,-.02,.84):v(.17,.32,.84));
    const stroke=Math.max(0,Math.min(1,(elapsed-2)/6));
    const work=v(...step.point).addScaledVector(side,-.09*Math.cos(stroke*6*Math.PI));
    const lifted=start.clone().addScaledVector(up,.12);
    const pickup=index===0;
    let destination=start.clone();
    if(elapsed<0&&pickup)destination.copy(start);
    else if(elapsed<1){
      if(!pickup)destination.copy(carry).lerp(lifted,ease(elapsed));
    }
    else if(elapsed<1.5)destination.copy(pickup?start:lifted).lerp(lifted,ease((elapsed-1)*2));
    else if(elapsed<2)destination.copy(lifted).lerp(start,ease((elapsed-1.5)*2));
    else if(elapsed<=8)destination.copy(work);
    else if(last)destination.copy(start);
    else destination.copy(start).lerp(carry,ease((elapsed-8)/1.5));
    this.cloth.position.copy(destination);
    // The pad has a rigid angled handle, allowing the wrist to stay clear of the surface.
    const facing=v(...step.point).sub(v(...sampleReplay(this.replay,step.start).position));facing.z=0;facing.normalize();
    const carryFacing=v(-Math.sin(nav.yaw),Math.cos(nav.yaw),0);
    if(elapsed>8&&!last)facing.lerp(carryFacing,ease((elapsed-8)/1.5));
    if(!pickup&&elapsed<1)facing.lerp(carryFacing,1-ease(elapsed));
    const handle=v(0,0,.14);
    if(this.replay.robot==='g1'){handle.z=.20;handle.addScaledVector(facing,-.22);}
    if(this.replay.robot==='stretch')handle.z=.045;
    rod(this.grip,v(),handle);
    this.wristLoop.position.copy(handle).addScaledVector(up,.016);
    this.wristLoop.quaternion.setFromUnitVectors(v(0,0,1),facing);
    const target=destination.clone().add(handle);
    const active=elapsed>=0&&(!last||time<=step.end);
    const homeWorld=robot.localToWorld(this.replay.robot==='stretch'?v(.47,-.02,.88):v(.17,.27,.92));
    let blend=1;
    if(pickup&&elapsed<1)blend=ease(elapsed);
    if(last&&elapsed>8)blend=1-ease((elapsed-8)/1.5);
    const hand=homeWorld.lerp(target,blend);
    this.reach(robot,model,hand,active);
    for(const patch of patches){
      const progress=Math.min(1,Math.max(0,(time-patch.step.start-2)/6));
      // A mark fades each time the pad passes it. Derive coverage from time so scrubbing is reversible.
      const sweeps=progress*6;
      const completed=Math.floor(sweeps);
      const current=-.09*Math.cos(sweeps*Math.PI);
      patch.spots.forEach((spot,i)=>{
        const offset=-.09+.045*i;
        const passed=completed%2===0?current>=offset:current<=offset;
        const passes=completed+Number(progress>0&&passed);
        (spot.material as T.MeshStandardMaterial).opacity=Math.max(0,1-passes/6);
        spot.visible=passes<6;
      });
    }
  }

  /** Aim the robot-specific effector at a world-space tool or dish grip. */
  reach(robot:T.Group,model:ArticulatedRobot|null,hand:T.Vector3,active:boolean){
    if(model&&active&&this.replay.robot==='g1')model.reachRightHand(hand);
    if(model&&active&&this.replay.robot==='stretch'){
      const dx=hand.x-robot.position.x,dy=hand.y-robot.position.y;
      const distance=Math.hypot(dx,dy);
      robot.rotation.z=Math.atan2(dy,dx)-Math.asin(-.0214/distance);
      const extension=T.MathUtils.clamp((Math.sqrt(distance*distance-.0214**2)-.3644)/.52,0,1);
      for(let j=0;j<4;j++)model.set('joint_arm_l'+j,.13*extension);
      model.set('joint_lift',hand.z-robot.position.z-.11465);
    }
    if(this.boom.visible){
      const mount=robot.localToWorld(v(0,.11,.36));
      const end=hand;
      const elbow=mount.clone().lerp(end,.48).addScaledVector(up,.14);
      rod(this.boom,mount,elbow);rod(this.piston,elbow,end);
    }
  }
}
