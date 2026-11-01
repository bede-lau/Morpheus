import test from 'node:test';
import assert from 'node:assert/strict';
import { directionFromPose, moveVector, poseFromDirection } from './fly';

test('fly pose round-trips through a look direction in the Z-up frame',()=>{
  for(const [yaw,pitch] of [[0,0],[1.2,-.4],[-2.5,.7],[3,0]]){
    const d=directionFromPose({yaw,pitch});const p=poseFromDirection({x:d[0],y:d[1],z:d[2]});
    assert.ok(Math.abs(Math.atan2(Math.sin(p.yaw-yaw),Math.cos(p.yaw-yaw)))<1e-9);assert.ok(Math.abs(p.pitch-pitch)<1e-9);
  }
  assert.deepEqual(directionFromPose({yaw:0,pitch:0}).map(v=>Math.round(v*1e6)/1e6+0),[0,1,0]);
});
test('movement keys map to the horizontal look frame and vertical axis',()=>{
  const fwd=moveVector(new Set(['KeyW']),0,2);assert.deepEqual(fwd.map(v=>Math.round(v*1e6)/1e6+0),[0,2,0]);
  const right=moveVector(new Set(['KeyD']),Math.PI/2,1);assert.deepEqual(right.map(v=>Math.round(v*1e6)/1e6+0),[0,1,0]);
  const up=moveVector(new Set(['Space']),.3,1);assert.deepEqual(up,[0,0,1]);
  const down=moveVector(new Set(['ShiftLeft','KeyW']),0,1);assert.ok(Math.abs(Math.hypot(...down)-1)<1e-9&&down[2]<0&&down[1]>0);
  assert.deepEqual(moveVector(new Set(['KeyW','KeyS']),0,1),[0,0,0]);
});
