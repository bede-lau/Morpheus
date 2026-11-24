import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import * as T from 'three';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import {ContactScene} from './contact-scene';
import {Cleaning} from './cleaning';
import {ArticulatedRobot} from './robot-model';
import {parseReplay,sampleReplay} from './replay';

for(const id of ['g1','stretch','go2'] as const){
  test(id+' completes each remaining kitchen action',async()=>{
    const geometry=fs.readFileSync(new URL('./robots/'+id+'.glb',import.meta.url));
    const {scene}=await new GLTFLoader().parseAsync(geometry.buffer.slice(geometry.byteOffset,geometry.byteOffset+geometry.byteLength),'');
    const model=new ArticulatedRobot(scene,JSON.parse(fs.readFileSync(new URL('./robots/'+id+'.json',import.meta.url),'utf8')));
    const robot=new T.Group();robot.add(scene);
    const finger=(side:string)=>scene.getObjectByName('link_SG3_gripper_'+side+'_finger_aruco')!.getWorldPosition(new T.Vector3());
    for(const slug of ['drawer-opening','counter-reach','aisle-clearance','clean-the-sink','move-the-bowl-to-the-sink']){
      const file=new URL('../rollouts/kitchen/'+slug+'/'+id+'.json',import.meta.url);
      const replay=parseReplay(JSON.parse(fs.readFileSync(file,'utf8')).replay);
      assert.equal(replay.verdict?.status,'completed',slug);
      if(slug==='clean-the-sink')assert.equal(replay.wipes?.[0].target,'sink');
      if(slug==='move-the-bowl-to-the-sink')assert.equal(replay.transfers?.[0].kind,'bowl');
      if(slug==='aisle-clearance'){
        assert.match(replay.events!.at(-1)!.label,/physics run complete/);
        assert.ok(replay.physics?.minimumClearance && replay.physics.heldOutPassed===12);
        assert.ok(new T.Vector3(...replay.frames[0].position).distanceTo(new T.Vector3(...replay.frames.at(-1)!.position))>2);
      }
      if(!replay.contact)continue;
      if(replay.physics){
        const start=replay.frames[0];
        const end=replay.frames.at(-1)!;
        assert.equal(start.contactForce,0);
        assert.equal(end.contactForce,0);
        assert.ok(replay.frames.some(frame=>(frame.contactForce??0)>1));
        continue; // Measured poses are checked in physics-replay.test.ts.
      }
      const trial=replay.contact;
      const fixture=new ContactScene(trial);
      const rig=new Cleaning(replay);
      function update(time:number){
        const f=sampleReplay(replay,time);
        robot.position.fromArray(f.position);robot.rotation.z=f.yaw;
        fixture.update(time,robot,model,rig);
      }
      let maxGap=0;
      for(let t=trial.contact;t<=trial.release+.00001;t+=.05){
        update(t);
        const hand=id==='g1'?scene.getObjectByName('right_wrist_yaw_link')!.localToWorld(new T.Vector3(.105,0,0)):
          id==='stretch'?finger('left').lerp(finger('right'),.5):rig.root.children[2].localToWorld(new T.Vector3(0,.5,0));
        maxGap=Math.max(maxGap, hand.distanceTo(fixture.handle));
      }
      assert.ok(maxGap<.035,slug+' hand gap '+maxGap.toFixed(4)+' m');
      update(replay.duration);
      assert.ok(fixture.fixture.position.distanceTo(new T.Vector3(...trial.travel))<1e-6);
      update(0);
      assert.equal(fixture.fixture.position.length(),0);
    }
  });
}
