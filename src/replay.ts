import type { Vec3 } from './types';

export type RobotId = 'stretch' | 'go2' | 'soarm';
export type ReplayFrame = { time: number; position: Vec3; yaw: number; armExtension: number };
export type Replay = { version: 1; duration: number; source: 'demo' | 'backend'; robot: RobotId; frames: ReplayFrame[] };
export const ROBOTS = [
  { id: 'go2' as const, name: 'Go2', type: 'Quadruped', description: 'A mobile quadruped for exploring a room. This demo model has no manipulation arm.', reach: 'No arm', clearance: 'Review', drawer: 'Not tested' },
  { id: 'soarm' as const, name: 'SO-ARM100', type: 'Stationary arm', description: 'A compact arm for a fixed work area. The demo keeps its base stationary while the arm moves.', reach: 'Review', clearance: 'Stationary', drawer: 'Not tested' },
  { id: 'stretch' as const, name: 'Stretch', type: 'Mobile manipulator', description: 'A mobile base moves between work areas. A telescoping arm approaches the counter target.', reach: 'Reachable', clearance: 'Tight', drawer: 'Not tested' },
];
// Z-up coordinates in the aligned sample scene. Illustrative only; not a planned or collision-tested route.
export function createDemoReplay(robot: RobotId): Replay {
  const route: Vec3[] = [[1.95,-1.2,0],[1.95,-.6,0],[1.95,0,0],[1.95,.6,0],[1.95,1.2,0],[1.95,1.2,0]];
  return { version: 1, duration: 24, source: 'demo', robot, frames: route.map((position,i)=>({ time: i*4.8, position: robot==='soarm'?[1.95,1.2,0]:position, yaw: 0, armExtension: i<4?0:.25 })) };
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
  return r;
}
