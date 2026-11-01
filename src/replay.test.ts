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
