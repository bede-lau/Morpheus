import test from 'node:test';
import assert from 'node:assert/strict';
import { EYE_HEIGHT, eyePose, forward, insetRect } from './pov';
import type { RobotId } from './replay';

const close=(a:number[],b:number[],eps=1e-9)=>a.forEach((v,i)=>assert.ok(Math.abs(v-b[i])<eps,`${a} vs ${b}`));
test('eye sits at the profile height, 0.12 m forward along yaw',()=>{
 const expected:Record<RobotId,number>={stretch:1.30,go2:.45,soarm:.35};
 for(const robot of Object.keys(expected) as RobotId[]){
  assert.equal(EYE_HEIGHT[robot],expected[robot]);
  close(eyePose(robot,[1,2,.1],0).eye,[1,2.12,.1+expected[robot]]);
  close(eyePose(robot,[0,0,0],Math.PI/2).eye,[-.12,0,expected[robot]]);
 }
});
test('view direction follows yaw and pitches down about 12 degrees',()=>{
 close(forward(0),[0,1,0]);close(forward(Math.PI),[0,-1,0]);
 const {direction,eye,target}=eyePose('go2',[0,0,0],0);
 assert.ok(Math.abs(Math.atan2(-direction[2],Math.hypot(direction[0],direction[1]))*180/Math.PI-12)<1e-9);
 assert.ok(Math.abs(Math.hypot(...direction)-1)<1e-9);close(target,eye.map((v,i)=>v+direction[i]));
});
test('inset is bottom-right, 16:10 and about a third of the width',()=>{
 const r=insetRect(1000,700);assert.equal(r.width,340);assert.equal(r.height,213);assert.equal(r.x,1000-340-14);assert.ok(r.y>=0&&r.y+r.height<=700);
});
