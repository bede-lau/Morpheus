import test from 'node:test';
import assert from 'node:assert/strict';
import { boxDistance, buildPhases, classifyEvent, computeRunMetrics, mapBounds, NEAR_MISS_M } from './run-metrics';
import type { Replay } from './replay';

const landmarks=[{id:'table',label:'Table',kind:'table',position:[2,0,.45] as [number,number,number],size:[1,.6,.9] as [number,number,number]},{id:'wall',label:'Wall',kind:'tall',position:[0,0,1] as [number,number,number],size:[10,10,2] as [number,number,number]}];
function replay(extra:Partial<Replay>={}):Replay{
 return {version:1,duration:10,source:'backend',robot:'stretch',task:'wipe the table',landmarks,
  frames:[{time:0,position:[0,0,0],yaw:0,armExtension:0},{time:5,position:[0,2,0],yaw:0,armExtension:0},{time:6,position:[0,2,0],yaw:-Math.PI/2,armExtension:0},{time:8,position:[0,2,0],yaw:-Math.PI/2,armExtension:.8},{time:10,position:[0,2,0],yaw:-Math.PI/2,armExtension:0}],
  events:[{time:0,label:'Navigating to the table'},{time:5,label:'Arrived at the table'},{time:6,label:'Wiping the table'},{time:9,label:'Task completed'}],
  verdict:{status:'completed',summary:'ok',planner:'rules',steps:[{name:'Navigate to table',status:'done'},{name:'Wipe table',status:'done'}]},...extra};
}
test('box distance is zero inside and euclidean outside the landmark footprint',()=>{
 assert.equal(boxDistance(2,0,landmarks[0]),0);assert.ok(Math.abs(boxDistance(2,1.3,landmarks[0])-1)<1e-9);assert.ok(Math.abs(boxDistance(3.5,0,landmarks[0])-1)<1e-9);
});
test('events map to phases and the planner vocabulary classifies correctly',()=>{
 assert.equal(classifyEvent('Go2 has no manipulator: cannot wipe the table'),'blocked');assert.equal(classifyEvent('Navigating to the sink counter'),'navigate');assert.equal(classifyEvent('Picked up Bowl'),'manipulate');assert.equal(classifyEvent('Spawned at the kitchen door'),'idle');
 const phases=buildPhases(replay());assert.deepEqual(phases.map(p=>p.kind),['navigate','idle','manipulate','idle']);assert.equal(phases.at(-1)!.end,10);
});
test('kinematics, clearance and score come out of a simple straight run',()=>{
 const m=computeRunMetrics(replay());
 assert.ok(Math.abs(m.pathLength-2)<.05);assert.ok(m.pathEfficiency>.98);assert.ok(Math.abs(m.meanSpeed-.4)<.05);assert.ok(m.totalTurn>Math.PI/2-.05&&m.totalTurn<Math.PI/2+.05);
 // Nearest obstacle (the table, walls excluded) at y=2 is 1.5 m from its edge at x=1.5, minus the 0.20 m Stretch body radius.
 assert.ok(m.minClearance!==null&&Math.abs(m.minClearance-(1.5-.20))<.02);assert.equal(m.nearMisses,0);
 assert.ok(m.armDuty>.15&&m.armDuty<.4);assert.equal(m.steps.length,2);assert.equal(m.steps[0].start,0);assert.equal(m.steps[1].start,6);
 assert.ok(m.score>=85,`score ${m.score}`);assert.equal(m.feedback[0].kind,'keep');assert.equal(m.firstFailure,null);
});
test('failures turn into feedback actions and a blocked point on the map',()=>{
 const m=computeRunMetrics(replay({robot:'go2',events:[{time:0,label:'Navigating to the table'},{time:5,label:'Arrived at the table'},{time:6,label:'Go2 has no manipulator: cannot wipe the table'}],verdict:{status:'partial',summary:'x',steps:[{name:'Navigate to table',status:'done'},{name:'Wipe table',status:'failed',note:'Go2 has no arm.'}]}}));
 assert.equal(m.firstFailure?.time,6);assert.equal(m.feedback[0].kind,'switch');assert.ok(/Stretch|G1/.test(m.feedback[0].detail));assert.ok(m.score<80);
 assert.equal(m.capabilities.find(c=>c.axis==='Manipulation')!.score,0);
});
test('near misses count contiguous intervals under the threshold while moving',()=>{
 const m=computeRunMetrics(replay({frames:[{time:0,position:[1.3,-1.5,0],yaw:0,armExtension:0},{time:10,position:[1.3,1.5,0],yaw:0,armExtension:0}],events:[{time:0,label:'Navigating to the table'}]}));
 assert.ok(m.minClearance!==null&&m.minClearance<NEAR_MISS_M);assert.equal(m.nearMisses,1);assert.ok(m.nearMissTime>0);
 assert.ok(m.feedback.some(f=>/near-miss/.test(f.title)));
});
test('map bounds cover the route and obstacle boxes with padding',()=>{
 const b=mapBounds(computeRunMetrics(replay()));assert.ok(b.minX<=-.3&&b.maxX>=2.8&&b.minY<=-.6&&b.maxY>=2.3);
});
