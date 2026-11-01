import test from 'node:test';
import assert from 'node:assert/strict';
import { createDemoReplay, parseReplay, sampleReplay } from './replay';

test('demo replay interpolates, clamps and reaches its final pose',()=>{
 const replay=createDemoReplay('stretch');
 assert.deepEqual(parseReplay(replay),replay);
 assert.deepEqual(sampleReplay(replay,-1).position,replay.frames[0].position);
 assert.ok(Math.abs(sampleReplay(replay,2.4).position[1]+.9)<1e-9);
 assert.deepEqual(sampleReplay(replay,100).position,replay.frames.at(-1)!.position);
 assert.equal(sampleReplay(replay,24).armExtension,.25);
});
test('stationary arm keeps its base fixed throughout the replay',()=>{
 const replay=createDemoReplay('soarm');
 assert.ok(replay.frames.every(f=>f.position.every((v,i)=>v===replay.frames[0].position[i])));
});
test('backend replay rejects invalid times, coordinates and unsupported robots',()=>{
 for(const mutate of [
  (r:any)=>r.frames[1].time=0,
  (r:any)=>r.frames[1].position[0]=Infinity,
  (r:any)=>r.frames[1].position=[0,0],
  (r:any)=>r.frames[0].time=1,
  (r:any)=>r.duration=100,
  (r:any)=>r.robot='unknown',
  (r:any)=>r.frames[2].armExtension=2,
 ]) {const replay=createDemoReplay('stretch');mutate(replay);assert.throws(()=>parseReplay(replay));}
});
test('heading interpolation takes the short route across the angle boundary',()=>{
 const replay=createDemoReplay('stretch');
 replay.frames[0].yaw=3;replay.frames[1].yaw=-3;
 assert.ok(Math.abs(sampleReplay(replay,2.4).yaw-Math.PI)<.001);
});

import { parsePolicyRun, type PolicyRun, type RobotId } from './replay';
function policyFixture(robot:RobotId='g1'):PolicyRun{
 const replay=createDemoReplay(robot);replay.source='backend';
 return {version:1,robot,readiness:'PREVIEW_ONLY',scene:{sha256:'a'.repeat(64),scaleConfirmed:false,reviewed:false,synthetic:true},training:robot==='soarm'?null:{algorithm:'PPO',model:'reduced_order_static_footprint',steps:1024,seed:42,heldoutSuccessRate:.5,baselineSuccessRate:.25},capability:{navigation:'Footprint navigation',locomotion:'Not verified by this replay',cleaning:'Tool unverified'},replay,routes:robot==='soarm'?[]:[{id:'route-a',label:'Shortest',kind:'planned',points:[[0,0,0],[1,0,0]],lengthM:1,clearanceM:.2,status:'footprint checked'}],cleaning:[{id:'wipe',label:'Wipe path',kind:'wipe',points:[[0,0,1],[.2,0,1]],status:'planned_tool_unverified',reason:'Requires a calibrated tool'}],caveats:['Unconfirmed scale']};
}
test('policy-run contract supports each registered footprint and fixed-arm capabilities',()=>{
 for(const robot of ['stretch','go2','g1','soarm'] as RobotId[]){const run=policyFixture(robot);assert.equal(parsePolicyRun(run),run);}
});
test('policy-run contract rejects invented training, malformed geometry and unsafe evidence links',()=>{
 for(const mutate of [
  (r:PolicyRun)=>r.replay.robot='go2',
  (r:PolicyRun)=>r.training!.steps=0,
  (r:PolicyRun)=>r.training!.heldoutSuccessRate=100,
  (r:PolicyRun)=>r.routes[0].points[0][0]=NaN,
  (r:PolicyRun)=>r.routes.push({...r.routes[0]}),
  (r:PolicyRun)=>r.scene.sha256='unverified',
  (r:PolicyRun)=>r.nativeReplayUrl='javascript:alert(1)',
  (r:PolicyRun)=>(r.cleaning[0] as any).status='completed',
 ]){const run=policyFixture();mutate(run);assert.throws(()=>parsePolicyRun(run));}
});
test('policy-run contract validates optional native-video, support, and training identities',()=>{
 const run=policyFixture();run.training!.checkpointSha256='b'.repeat(64);run.nativeVideoUrl='/policies/g1-native/demo.mp4';run.scene.supportStatus='sampled_no_overlap';run.scene.supportNote='No sampled floor overlap; inspect supplied collision boxes.';
 assert.equal(parsePolicyRun(run),run);
 for(const mutate of [(r:PolicyRun)=>r.training!.checkpointSha256='not-a-hash',(r:PolicyRun)=>r.nativeVideoUrl='javascript:alert(1)',(r:PolicyRun)=>(r.scene as any).supportStatus='supported',(r:PolicyRun)=>(r.scene as any).supportNote=3]){const invalid=policyFixture();mutate(invalid);assert.throws(()=>parsePolicyRun(invalid));}
});
test('import cannot turn a stationary SO-ARM100 into a moving base',()=>{
 const run=policyFixture('soarm');run.replay.frames[1].position=[2,1,0];assert.throws(()=>parsePolicyRun(run),/stationary/);
});
