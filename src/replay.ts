import type { Vec3 } from './types';

export type RobotId = 'stretch' | 'go2' | 'g1';
export type ReplayFrame = { time: number; position: Vec3; yaw: number; armExtension: number };
export type ReplayEvent = { time: number; label: string };
export type StepStatus = 'done' | 'failed' | 'skipped';
export type Verdict = { status: 'completed' | 'partial' | 'failed'; summary: string; steps: { name: string; status: StepStatus; note?: string }[]; planner?: string };
export type Landmark = { id: string; label: string; position: Vec3; size: Vec3; kind: string };
/** A saved SimReady object the robot carries: picked at pickTime, set down at placeAt at placeTime. */
export type CarriedObject = { objectId: string; assetId?: string; pickTime: number; placeTime: number; placeAt: Vec3; carryOffset: Vec3; pickAt?: Vec3 };
export type Replay = { version: 1; duration: number; source: 'demo' | 'backend'; robot: RobotId; frames: ReplayFrame[]; task?: string; events?: ReplayEvent[]; verdict?: Verdict; landmarks?: Landmark[]; carried?: CarriedObject[] };
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
  return {time:t,position:a.position.map((v,i)=>v+(b.position[i]-v)*mix) as Vec3,yaw:a.yaw+angle*mix,armExtension:a.armExtension+(b.armExtension-a.armExtension)*mix};
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
export const STRETCH_ARM={mount:[-.13,.05] as [number,number],height:.98,link:.58,baseScale:.65,liftWindow:.8};
const smooth=(x:number)=>{const t=Math.max(0,Math.min(1,x));return t*t*(3-2*t)};
/** World position of the Stretch gripper tip for a robot pose, arm extension and lift height above the base. */
export function stretchTip(position:Vec3,yaw:number,armExtension:number,lift=STRETCH_ARM.height):Vec3{
  const x=STRETCH_ARM.mount[0],y=STRETCH_ARM.mount[1]+STRETCH_ARM.link*(STRETCH_ARM.baseScale+armExtension),c=Math.cos(yaw),s=Math.sin(yaw);
  return [position[0]+x*c-y*s,position[1]+x*s+y*c,position[2]+lift];
}
/** Arm lift height (above the base) so the gripper dips to each pick/place height around pickTime/placeTime. */
export function stretchLift(carried:CarriedObject[]|undefined,time:number,baseZ=0,origins?:Record<string,Vec3>):number{
  let lift=STRETCH_ARM.height;const w=STRETCH_ARM.liftWindow;
  for(const c of carried??[]){
    const pickZ=(c.pickAt??origins?.[c.objectId])?.[2];
    for(const [t,z] of [[c.pickTime,pickZ],[c.placeTime,c.placeAt[2]]] as [number,number|undefined][]){
      if(z===undefined)continue;
      const weight=time<=t?smooth((time-(t-w))/w):1-smooth((time-t)/w);
      if(weight>0)lift=lift+(z-c.carryOffset[2]-baseZ-lift)*weight;
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
