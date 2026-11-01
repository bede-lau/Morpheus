import test from 'node:test';
import assert from 'node:assert/strict';
import { EYE_HEIGHT, INSET_MAX_FRACTION, INSET_MIN_FRACTION, POV_INSET, clampInset, eyePose, forward, insetRect, moveInset, resizeInset } from './pov';
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
test('inset layout stays inside the viewport while moving and resizing',()=>{
 const W=1000,H=700;
 const moved=moveInset(POV_INSET,-200,-100,W,H);assert.equal(moved.right,214);assert.equal(moved.bottom,204);
 const far=moveInset(POV_INSET,5000,5000,W,H);assert.equal(far.right,0);assert.equal(far.bottom,0);
 const pinned=insetRect(W,H,moveInset(POV_INSET,-5000,-5000,W,H));assert.equal(pinned.x,0);assert.equal(pinned.y+pinned.height,H);
 const grown=resizeInset(POV_INSET,-100,W,H);assert.ok(Math.abs(grown.widthFraction-.44)<1e-9);assert.equal(grown.right,14);assert.equal(grown.bottom,104);
 assert.equal(resizeInset(POV_INSET,1000,W,H).widthFraction,INSET_MIN_FRACTION);
 const huge=insetRect(W,H,resizeInset(POV_INSET,-5000,W,H));assert.equal(huge.width,INSET_MAX_FRACTION*W);assert.ok(huge.x>=0&&huge.y>=0&&huge.y+huge.height<=H);
 const small=clampInset({...POV_INSET,right:900,bottom:650},W,H);const r=insetRect(W,H,small);assert.ok(r.x>=0&&r.y>=0&&r.x+r.width<=W&&r.y+r.height<=H);
});
