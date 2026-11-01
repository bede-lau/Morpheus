import test from 'node:test';
import assert from 'node:assert/strict';
import { carriedPose, parseReplay, stretchLift, stretchTip, STRETCH_ARM, type CarriedObject } from './replay';

const entry:CarriedObject={objectId:'bowl',pickTime:10,placeTime:20,placeAt:[1,2,.7],carryOffset:[0,0,.05],pickAt:[3,-2,.6]};
const close=(a:number[],b:number[])=>a.forEach((v,i)=>assert.ok(Math.abs(v-b[i])<1e-9,`${a} vs ${b}`));

test('carried object rests, follows the gripper tip, then stays at placeAt',()=>{
  const tip:[number,number,number]=[5,6,1];
  assert.deepEqual(carriedPose(entry,9.99,tip,[3,-2,.6]),{phase:'rest',position:[3,-2,.6]});
  const mid=carriedPose(entry,15,tip,[3,-2,.6]);assert.equal(mid.phase,'carried');close(mid.position,[5,6,1.05]);
  assert.equal(carriedPose(entry,10,tip,[3,-2,.6]).phase,'carried');
  assert.deepEqual(carriedPose(entry,20,tip,[3,-2,.6]),{phase:'placed',position:[1,2,.7]});
});

test('stretch tip is forward along yaw with the arm mount offset',()=>{
  // yaw 0 faces +Y: tip x = mount x, y = mount y + link*(base+ext)
  close(stretchTip([0,0,0],0,1),[-.13,.05+.58*1.65,STRETCH_ARM.height]);
  // yaw pi/2 faces -X
  close(stretchTip([1,1,0],Math.PI/2,0,.5),[1-(.05+.58*.65),1-.13,.5]);
});

test('lift dips to the pick and place heights only around those times',()=>{
  assert.equal(stretchLift([entry],0),STRETCH_ARM.height);
  assert.ok(Math.abs(stretchLift([entry],10)-(.6-.05))<1e-9);
  assert.ok(Math.abs(stretchLift([entry],20)-(.7-.05))<1e-9);
  assert.equal(stretchLift([entry],15),STRETCH_ARM.height);
  assert.equal(stretchLift(undefined,10),STRETCH_ARM.height);
});

test('replay carried entries are validated but optional',()=>{
  const base={version:1,source:'backend',robot:'stretch',duration:30,frames:[{time:0,position:[0,0,0],yaw:0,armExtension:0},{time:30,position:[0,0,0],yaw:0,armExtension:0}]};
  assert.doesNotThrow(()=>parseReplay(base));
  assert.doesNotThrow(()=>parseReplay({...base,carried:[entry]}));
  assert.throws(()=>parseReplay({...base,carried:[{...entry,placeTime:5}]}),/carried/);
  assert.throws(()=>parseReplay({...base,carried:[{...entry,placeAt:[1,2]}]}),/carried/);
});
