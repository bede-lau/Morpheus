import type { Vec3 } from './types';

export type RobotId = 'stretch' | 'go2' | 'soarm' | 'g1';
export type ReplayFrame = { time: number; position: Vec3; yaw: number; armExtension: number };
export type Replay = { version: 1; duration: number; source: 'demo' | 'backend'; robot: RobotId; frames: ReplayFrame[] };
export const ROBOTS = [
  { id: 'g1' as const, name: 'Unitree G1', type: 'Humanoid', description: 'A humanoid with a pinned native MuJoCo locomotion controller. Navigation learning is evaluated separately from its gait.', reach: 'Not evaluated', clearance: 'Not evaluated', drawer: 'Not tested' },
  { id: 'go2' as const, name: 'Go2', type: 'Quadruped', description: 'A mobile quadruped for exploring a room. This demo model has no manipulation arm.', reach: 'No arm', clearance: 'Review', drawer: 'Not tested' },
  { id: 'soarm' as const, name: 'SO-ARM100', type: 'Stationary arm', description: 'A compact arm for a fixed work area. The demo keeps its base stationary while the arm moves.', reach: 'Review', clearance: 'Stationary', drawer: 'Not tested' },
  { id: 'stretch' as const, name: 'Stretch', type: 'Mobile manipulator', description: 'A mobile base moves between work areas. A telescoping arm approaches the counter target.', reach: 'Not evaluated', clearance: 'Not evaluated', drawer: 'Not tested' },
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
  if(r.robot==='soarm'&&r.frames.some(f=>f.position.some((v,i)=>Math.abs(v-r.frames[0].position[i])>1e-8)||Math.abs(f.yaw-r.frames[0].yaw)>1e-8))throw new Error('The fixed arm base must remain stationary.');
  return r;
}


export type PolicyRoute = {id:string;label:string;kind:'planned'|'learned';points:Vec3[];lengthM:number;clearanceM:number|null;status:string};
export type CleaningIntent = {id:string;label:string;kind:'wipe'|'sweep'|'inspect';points:Vec3[];status:'planned_tool_unverified'|'unsupported';reason:string};
export type PolicyRun = {
  version:1;robot:RobotId;readiness:string;
  scene:{sha256:string;scaleConfirmed:boolean;reviewed:boolean;synthetic:boolean;manifestUrl?:string;supportStatus?:'sampled_no_overlap';supportNote?:string};
  training:null|{algorithm:string;model:string;steps:number;seed:number;heldoutSuccessRate?:number;baselineSuccessRate?:number;warmStartedSuccessRate?:number|null;astarSuccessRate?:number;heldoutEpisodes?:number;evaluationScope?:string;checkpointSha256?:string};
  capability:{navigation:string;locomotion:string;cleaning:string};
  replay:Replay;routes:PolicyRoute[];cleaning:CleaningIntent[];caveats:string[];
  validation?:{model:string;status:string;completed:boolean;forbiddenContacts:number;reportUrl?:string};
  nativeReplayUrl?:string;nativeManifestUrl?:string;nativeVideoUrl?:string;
};
function finitePoints(value:unknown):value is Vec3[]{return Array.isArray(value)&&value.length>=2&&value.length<=100000&&value.every(p=>Array.isArray(p)&&p.length===3&&p.every(Number.isFinite));}
function stringField(value:unknown):value is string{return typeof value==='string'&&value.length>0&&value.length<5000;}
/** Data-only preview contract. Importing results never upgrades physics or hardware readiness. */
export function parsePolicyRun(data:unknown):PolicyRun{
  const r=data as PolicyRun;
  if(!r||r.version!==1||!ROBOTS.some(b=>b.id===r.robot)||!stringField(r.readiness)||!r.scene||!/^[a-f0-9]{64}$/i.test(r.scene.sha256)||!['scaleConfirmed','reviewed','synthetic'].every(k=>typeof r.scene[k as keyof typeof r.scene]==='boolean'))throw new Error('Policy run needs a supported robot and a hashed source scene.');
  parseReplay(r.replay);if(r.replay.robot!==r.robot||r.replay.source!=='backend')throw new Error('Policy run needs a matching backend replay.');
  if(!r.capability||!['navigation','locomotion','cleaning'].every(k=>stringField(r.capability[k as keyof typeof r.capability])))throw new Error('Policy capabilities are missing.');
  if(r.training!==null&&(!r.training||!stringField(r.training.algorithm)||!stringField(r.training.model)||!Number.isInteger(r.training.steps)||r.training.steps<1||!Number.isInteger(r.training.seed)))throw new Error('Training must contain actual steps and a seed, or be null.');
  if(r.training){for(const v of [r.training.heldoutSuccessRate,r.training.baselineSuccessRate,r.training.warmStartedSuccessRate,r.training.astarSuccessRate])if(v!==undefined&&v!==null&&(!Number.isFinite(v)||v<0||v>1))throw new Error('Success rates must be between 0 and 1.');if(r.training.heldoutEpisodes!==undefined&&(!Number.isInteger(r.training.heldoutEpisodes)||r.training.heldoutEpisodes<1))throw new Error('Evaluation count must be positive.');if(r.training.evaluationScope!==undefined&&!stringField(r.training.evaluationScope))throw new Error('Evaluation scope must be text.');if(r.training.checkpointSha256!==undefined&&!/^[a-f0-9]{64}$/i.test(r.training.checkpointSha256))throw new Error('Training checkpoint digest must be SHA-256.');}
  if(r.scene.supportStatus!==undefined&&r.scene.supportStatus!=='sampled_no_overlap')throw new Error('Scene support status is invalid.');
  if(r.scene.supportNote!==undefined&&!stringField(r.scene.supportNote))throw new Error('Scene support note is invalid.');
  if(!Array.isArray(r.routes)||r.routes.length>20||r.routes.some(p=>!stringField(p.id)||!stringField(p.label)||!['planned','learned'].includes(p.kind)||!finitePoints(p.points)||!Number.isFinite(p.lengthM)||p.lengthM<0||(p.clearanceM!==null&&!Number.isFinite(p.clearanceM))||!stringField(p.status)))throw new Error('Routes need finite geometry and recorded metrics.');
  if(new Set(r.routes.map(p=>p.id)).size!==r.routes.length)throw new Error('Route IDs must be unique.');
  if(!Array.isArray(r.cleaning)||r.cleaning.length>20||r.cleaning.some(p=>!stringField(p.id)||!stringField(p.label)||!['wipe','sweep','inspect'].includes(p.kind)||!['planned_tool_unverified','unsupported'].includes(p.status)||!finitePoints(p.points)||!stringField(p.reason)))throw new Error('Cleaning overlays must be labelled planned or unsupported.');
  if(!Array.isArray(r.caveats)||r.caveats.length>50||!r.caveats.every(stringField))throw new Error('Policy run caveats are required.');
  if(r.validation&&(!stringField(r.validation.model)||!stringField(r.validation.status)||typeof r.validation.completed!=='boolean'||!Number.isInteger(r.validation.forbiddenContacts)||r.validation.forbiddenContacts<0))throw new Error('Native validation summary is malformed.');
  for(const url of [r.validation?.reportUrl,r.nativeReplayUrl,r.nativeManifestUrl,r.nativeVideoUrl,r.scene.manifestUrl])if(url!==undefined&&(typeof url!=='string'||!(/^\/(?!\/)/.test(url)||/^https:\/\//.test(url))))throw new Error('Artifact links must be local paths or HTTPS URLs.');
  return r;
}
