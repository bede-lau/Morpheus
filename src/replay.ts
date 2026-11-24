import type { Vec3 } from './types';

export type RobotId = 'stretch' | 'go2' | 'g1';
export type PhysicsPose = { quaternion:[number,number,number,number]; joints:number[] };
export type PhysicsModel = 'g1-12dof' | 'g1-hands' | 'go2' | 'go2-arm';
export type PhysicsEvidence = {
  engine:string; modelRevision:string; visualModel?:PhysicsModel; jointNames:string[];
  goalError:number; minimumClearance:number; heldOutPassed:number; heldOutTotal:number;
  contactDwellSeconds?:number; peakContactForceNewtons?:number;
};
export type ReplayFrame = { time: number; position: Vec3; yaw: number; armExtension: number; physicsPose?:PhysicsPose; contactForce?:number; drawerDisplacement?:number };
export type ReplayEvent = { time: number; label: string };
export type StepStatus = 'done' | 'failed' | 'skipped';
export type Verdict = { status: 'completed' | 'partial' | 'failed'; summary: string; steps: { name: string; status: StepStatus; note?: string }[]; planner?: string };
export type Landmark = { fixture?: string; id: string; label: string; position: Vec3; size: Vec3; kind: string };
/** A saved SimReady object the robot carries: picked at pickTime, set down at placeAt at placeTime. */
export type CarriedObject = { objectId: string; assetId?: string; pickTime: number; placeTime: number; placeAt: Vec3; carryOffset: Vec3; pickAt?: Vec3 };
export type Transfer = {
  kind:'glass'|'bowl'; objectId?:string; pick:Vec3; place:Vec3;
  reach:number; grasp:number; lift:number; lower:number; release:number; end:number;
  fillStart?:number; fillEnd?:number;
};
export type ContactTrial = {
  kind:'drawer'|'counter'; point:Vec3; travel:Vec3; padSize?:Vec3;
  drawerParts?:{position:Vec3;size:Vec3}[];
  start:number; contact:number; release:number; end:number;
};
export type Wipe = { target: string; point: Vec3; start: number; end: number };
export type RecordedCloth = {
  faces:[number,number,number][];
  marks:Vec3[];
  frames:{time:number;points:number[];tabPosition:Vec3;tabQuaternion:PhysicsPose['quaternion'];wiped:number[]}[];
};
export type Replay = { version: 1; duration: number; source: 'demo' | 'backend'; robot: RobotId; frames: ReplayFrame[]; task?: string; events?: ReplayEvent[]; verdict?: Verdict; landmarks?: Landmark[]; carried?: CarriedObject[]; wipes?: Wipe[]; cleaningKit?: boolean; transfers?: Transfer[]; contact?: ContactTrial; physics?: PhysicsEvidence; cloth?:RecordedCloth };
export const ROBOTS = [
  { id: 'go2' as const, name: 'Go2', type: 'Quadruped', description: 'A mobile quadruped for exploring a room. This demo model has no manipulation arm.', reach: 'No arm', clearance: 'Review', drawer: 'Not tested' },
  { id: 'g1' as const, name: 'Unitree G1', type: 'Humanoid', description: 'A 1.3 m humanoid that walks between work areas and reaches with two arms at counter height.', reach: 'Reachable', clearance: 'Review', drawer: 'Not tested' },
  { id: 'stretch' as const, name: 'Stretch', type: 'Mobile manipulator', description: 'A mobile base moves between work areas. A telescoping arm approaches the counter target.', reach: 'Reachable', clearance: 'Tight', drawer: 'Not tested' },
];
// Z-up coordinates in the aligned sample scene. Illustrative only; not a planned or collision-tested route.
export function createDemoReplay(robot: RobotId): Replay {
  const route: Vec3[] = [[1.95,-1.2,0],[1.95,-.6,0],[1.95,0,0],[1.95,.6,0],[1.95,1.2,0],[1.95,1.2,0]];
  return { version: 1, duration: 24, source: 'demo', robot, frames: route.map((position,i)=>({ time: i*4.8, position, yaw: 0, armExtension: i<4?0:.25 })) };
}
export function sampleReplay(replay: Replay, time: number): ReplayFrame {
  const t=Math.max(0,Math.min(replay.duration,time));
  const end=replay.frames.findIndex(f=>f.time>t);
  if(end<0)return {...replay.frames[replay.frames.length-1],time:t};
  if(end===0)return {...replay.frames[0],time:t};
  const a=replay.frames[end-1],b=replay.frames[end],mix=(t-a.time)/(b.time-a.time);
  const angle=Math.atan2(Math.sin(b.yaw-a.yaw),Math.cos(b.yaw-a.yaw));
  const drawerDisplacement=a.drawerDisplacement!==undefined&&b.drawerDisplacement!==undefined?
    a.drawerDisplacement+(b.drawerDisplacement-a.drawerDisplacement)*mix:undefined;
  return {time:t,contactForce:a.contactForce,drawerDisplacement,physicsPose:blendPhysics(a.physicsPose,b.physicsPose,mix),position:a.position.map((v,i)=>v+(b.position[i]-v)*mix) as Vec3,yaw:a.yaw+angle*mix,armExtension:a.armExtension+(b.armExtension-a.armExtension)*mix};
}
function blendPhysics(a:PhysicsPose|undefined,b:PhysicsPose|undefined,fraction:number):PhysicsPose|undefined{
  if(!a||!b)return undefined;
  const sign=a.quaternion.reduce((sum,v,i)=>sum+v*b.quaternion[i],0)<0?-1:1;
  const rotation=a.quaternion.map((v,i)=>v+(sign*b.quaternion[i]-v)*fraction);
  const norm=Math.hypot(...rotation);
  return {quaternion:rotation.map(v=>v/norm) as PhysicsPose['quaternion'],
    joints:a.joints.map((value,i)=>value+(b.joints[i]-value)*fraction)};
}
export function parseReplay(data: unknown): Replay {
  const r=data as Replay;
  if(!r||r.version!==1||!['demo','backend'].includes(r.source)||!ROBOTS.some(b=>b.id===r.robot)||!Number.isFinite(r.duration)||r.duration<=0||!Array.isArray(r.frames)||r.frames.length<2||r.frames.length>100000)throw new Error('Expected a version 1 replay with a supported robot and at least two frames.');
  let previous=-1;
  for(const frame of r.frames){
    if(!Number.isFinite(frame.time)||frame.time<=previous||frame.time<0||frame.time>r.duration||!Array.isArray(frame.position)||frame.position.length!==3||!frame.position.every(Number.isFinite)||!Number.isFinite(frame.yaw)||!Number.isFinite(frame.armExtension)||frame.armExtension<0||frame.armExtension>1)throw new Error('Replay frames need ordered times, finite Z-up positions, yaw, and armExtension between 0 and 1.');
    previous=frame.time;
  }
  if(r.frames[0].time!==0||r.frames.at(-1)!.time!==r.duration)throw new Error('Replay frames must cover time 0 through duration.');
  validateTaskFields(r);
  return r;
}
const vec3=(v:unknown)=>Array.isArray(v)&&v.length===3&&v.every(Number.isFinite);
function validateTaskFields(r:Replay){
  if(r.frames.some(f=>f.contactForce!==undefined&&(!Number.isFinite(f.contactForce)||f.contactForce<0))){
    throw new Error('Invalid recorded contact force.');
  }
  if(r.physics!==undefined){
    const p=r.physics;
    if(!p||typeof p.engine!=='string'||typeof p.modelRevision!=='string'
      ||!(p.visualModel===undefined||r.robot==='g1'&&(p.visualModel==='g1-12dof'||p.visualModel==='g1-hands')
        ||r.robot==='go2'&&(p.visualModel==='go2'||p.visualModel==='go2-arm'))
      ||!Array.isArray(p.jointNames)||p.jointNames.length<1||p.jointNames.length>64
      ||p.jointNames.some(name=>typeof name!=='string'||!name)||new Set(p.jointNames).size!==p.jointNames.length
      ||![p.goalError,p.minimumClearance].every(value=>Number.isFinite(value)&&value>=0)
      ||(p.contactDwellSeconds!==undefined||p.peakContactForceNewtons!==undefined)
        &&(!Number.isFinite(p.contactDwellSeconds)||p.contactDwellSeconds!<0||!Number.isFinite(p.peakContactForceNewtons)||p.peakContactForceNewtons!<0)
      ||!Number.isInteger(p.heldOutTotal)||!Number.isInteger(p.heldOutPassed)||p.heldOutTotal<1||p.heldOutPassed!==p.heldOutTotal
      ||r.frames.some(({physicsPose:pose})=>!pose||!Array.isArray(pose.quaternion)||pose.quaternion.length!==4
        ||!pose.quaternion.every(Number.isFinite)||Math.abs(Math.hypot(...pose.quaternion)-1)>1e-4
        ||!Array.isArray(pose.joints)||pose.joints.length!==p.jointNames.length||!pose.joints.every(Number.isFinite))){
      throw new Error('Invalid recorded physics replay.');
    }
  }
  if(r.contact!==undefined){
    const c=r.contact;
    if(!c||!['drawer','counter'].includes(c.kind)||!vec3(c.point)||!vec3(c.travel)
      ||c.padSize!==undefined&&(!vec3(c.padSize)||c.padSize.some(value=>value<=0))
      ||![c.start,c.contact,c.release,c.end].every(Number.isFinite)
      ||!(0<=c.start&&c.start<c.contact&&c.contact<c.release&&c.release<c.end&&c.end<=r.duration)){
      throw new Error('Invalid contact trial.');
    }
    if(c.kind==='drawer'&&r.physics&&(!Array.isArray(c.drawerParts)||c.drawerParts.length!==4
      ||c.drawerParts.some(part=>!part||!vec3(part.position)||!vec3(part.size)||part.size.some(value=>value<=0))
      ||r.frames.some(frame=>typeof frame.drawerDisplacement!=='number'||!Number.isFinite(frame.drawerDisplacement)))){
      throw new Error('Invalid recorded drawer motion.');
    }
  }
  if(r.cloth!==undefined){
    const c=r.cloth;
    const count=c?.frames?.[0]?.points?.length??0, vertices=count/3;
    if(!r.physics||!c||!Array.isArray(c.marks)||c.marks.length>64||!c.marks.every(vec3)
      ||!Number.isInteger(vertices)||vertices<3||vertices>256
      ||!Array.isArray(c.faces)||!c.faces.length||c.faces.length>512
      ||c.faces.some(f=>!vec3(f)||f.some(i=>!Number.isInteger(i)||i<0||i>=vertices))
      ||!Array.isArray(c.frames)||c.frames.length<2||c.frames.length>4096
      ||c.frames[0].time!==0||c.frames.at(-1)!.time!==r.duration
      ||c.frames.some((f,i)=>!f||!Number.isFinite(f.time)||i>0&&f.time<=c.frames[i-1].time
        ||!Array.isArray(f.points)||f.points.length!==count||!f.points.every(Number.isFinite)
        ||!vec3(f.tabPosition)||!Array.isArray(f.tabQuaternion)||f.tabQuaternion.length!==4
        ||!f.tabQuaternion.every(Number.isFinite)||Math.abs(Math.hypot(...f.tabQuaternion)-1)>.001
        ||!Array.isArray(f.wiped)||f.wiped.some(mark=>!Number.isInteger(mark)||mark<0||mark>=c.marks.length))){
      throw new Error('Invalid recorded cloth motion.');
    }
  }
  if(r.transfers!==undefined){
    if(!Array.isArray(r.transfers))throw new Error('Expected object transfer tracks.');
    let previousEnd=0;
    for(const t of r.transfers){
      if(!t||!['glass','bowl'].includes(t.kind)||!vec3(t.pick)||!vec3(t.place))throw new Error('Invalid transfer object or pose.');
      const times=t.kind==='glass'?[t.reach,t.grasp,t.fillStart,t.fillEnd,t.lift,t.lower,t.release,t.end]:[t.reach,t.grasp,t.lift,t.lower,t.release,t.end];
      if(times.some((v,i)=>typeof v!=='number'||!Number.isFinite(v)||v<(i?times[i-1]!:previousEnd)||v>r.duration)
        ||t.reach>=t.grasp||t.lift+2>t.lower||t.lower>=t.release||t.release>=t.end||(t.kind==='glass'&&t.fillStart!>=t.fillEnd!))throw new Error('Invalid transfer timing.');
      previousEnd=t.end;
    }
  }
  if(r.wipes!==undefined && (!Array.isArray(r.wipes)||r.wipes.some((w,i)=>!w||typeof w.target!=='string'||!vec3(w.point)||!Number.isFinite(w.start)||!Number.isFinite(w.end)||w.start<0||w.end-w.start<8||w.end>r.duration||(i>0&&w.start<r.wipes![i-1].end))))throw new Error('Invalid cleaning contact track.');
  if(r.task!==undefined&&typeof r.task!=='string')throw new Error('Replay task must be text.');
  if(r.events!==undefined){if(!Array.isArray(r.events)||r.events.some(e=>!e||!Number.isFinite(e.time)||typeof e.label!=='string'))throw new Error('Replay events need finite times and text labels.');}
  if(r.verdict!==undefined){const v=r.verdict;if(!v||!['completed','partial','failed'].includes(v.status)||typeof v.summary!=='string'||!Array.isArray(v.steps)||v.steps.some(s=>!s||typeof s.name!=='string'||!['done','failed','skipped'].includes(s.status)||(s.note!==undefined&&typeof s.note!=='string'))||(v.planner!==undefined&&typeof v.planner!=='string'))throw new Error('Replay verdict is malformed.');}
  if(r.carried!==undefined){if(!Array.isArray(r.carried)||r.carried.some(c=>!c||typeof c.objectId!=='string'||!Number.isFinite(c.pickTime)||!Number.isFinite(c.placeTime)||c.pickTime<0||c.pickTime>=c.placeTime||!vec3(c.placeAt)||!vec3(c.carryOffset)||(c.pickAt!==undefined&&!vec3(c.pickAt))))throw new Error('Replay carried objects are malformed.');}
  if(r.landmarks!==undefined){if(!Array.isArray(r.landmarks)||r.landmarks.some(l=>!l||typeof l.id!=='string'||typeof l.label!=='string'||typeof l.kind!=='string'||!vec3(l.position)||!vec3(l.size)))throw new Error('Replay landmarks are malformed.');}
}
/** Latest event whose time is at or before `time`. */
export function currentEvent(replay:Pick<Replay,'events'>|null|undefined,time:number):ReplayEvent|null{
  let best:ReplayEvent|null=null;
  for(const e of replay?.events??[])if(e.time<=time+1e-6&&(!best||e.time>=best.time))best=e;
  return best;
}

/** Robot-local Stretch arm model (Z-up, +Y forward): arm group origin and gripper tip distance per unit arm scale. */
/** Stretch 3 arm (src/robots/stretch.json): mounted on the mast at local (-0.135, -0.02), telescoping along local +X (the robot's right side,
 * yaw 0 faces +Y); grasp centre at mount.x + link*(baseScale+ext) and tipDrop below the arm height. */
export const STRETCH_ARM={mount:[-.135,-.02] as [number,number],height:.98,link:.52,baseScale:1.058,tipDrop:.086,liftWindow:.8};
const smooth=(x:number)=>{const t=Math.max(0,Math.min(1,x));return t*t*(3-2*t)};
/** World position of the Stretch gripper tip for a robot pose, arm extension and lift height above the base. */
export function stretchTip(position:Vec3,yaw:number,armExtension:number,lift=STRETCH_ARM.height):Vec3{
  const x=STRETCH_ARM.mount[0]+STRETCH_ARM.link*(STRETCH_ARM.baseScale+armExtension),y=STRETCH_ARM.mount[1],c=Math.cos(yaw),s=Math.sin(yaw);
  return [position[0]+x*c-y*s,position[1]+x*s+y*c,position[2]+lift-STRETCH_ARM.tipDrop];
}
/** Arm lift height (above the base) so the gripper dips to each pick/place height around pickTime/placeTime. */
export function stretchLift(carried:CarriedObject[]|undefined,time:number,baseZ=0,origins?:Record<string,Vec3>):number{
  let lift=STRETCH_ARM.height;const w=STRETCH_ARM.liftWindow;
  for(const c of carried??[]){
    const pickZ=(c.pickAt??origins?.[c.objectId])?.[2];
    for(const [t,z] of [[c.pickTime,pickZ],[c.placeTime,c.placeAt[2]]] as [number,number|undefined][]){
      if(z===undefined)continue;
      const weight=time<=t?smooth((time-(t-w))/w):1-smooth((time-t)/w);
      if(weight>0)lift=lift+(z-c.carryOffset[2]-baseZ+STRETCH_ARM.tipDrop-lift)*weight;
    }
  }
  return lift;
}
/** Where a carried object is at `time`: at rest before pickTime, following the gripper tip until placeTime, then at placeAt. */
export function carriedPose(entry:CarriedObject,time:number,tip:Vec3,rest:Vec3):{phase:'rest'|'carried'|'placed';position:Vec3}{
  if(time<entry.pickTime)return {phase:'rest',position:[...rest] as Vec3};
  if(time>=entry.placeTime)return {phase:'placed',position:[...entry.placeAt] as Vec3};
  return {phase:'carried',position:tip.map((v,i)=>v+entry.carryOffset[i]) as Vec3};
}
