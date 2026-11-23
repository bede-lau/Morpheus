import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import * as T from 'three';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import {DishTransfer} from './dish-transfer';
import {Cleaning} from './cleaning';
import {ArticulatedRobot} from './robot-model';
import {parseReplay,sampleReplay} from './replay';

for(const id of ['g1','go2','stretch'] as const){
  for(const slug of ['fill-a-glass-of-water','wipe-the-table-and-move-the-bowl-to-the-sink','wipe-the-table']){
    test(id+' completes '+slug+' with visible object contact',async()=>{
      const read=(path:string)=>fs.readFileSync(new URL(path,import.meta.url));
      const replay=parseReplay(JSON.parse(read('../rollouts/kitchen/'+slug+'/'+id+'.json').toString()).replay);
      assert.equal(replay.verdict?.status,'completed');
      const bytes=read('./robots/'+id+'.glb');
      const {scene}=await new GLTFLoader().parseAsync(bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength),'');
      const model=new ArticulatedRobot(scene,JSON.parse(read('./robots/'+id+'.json').toString()));
      const robot=new T.Group();robot.add(scene);
      const rig=new Cleaning(replay);
      const track=replay.transfers?.[0];
      const dish=track&&new DishTransfer(replay,track);
      function pose(t:number){
        const f=sampleReplay(replay,t);
        robot.position.fromArray(f.position);robot.rotation.z=f.yaw;
        rig.update(t,robot,model);
        const target=dish?.update(t,robot,model,rig,null)??rig.cloth.children[2].localToWorld(new T.Vector3(0,.5,0));
        const finger=(side:string)=>scene.getObjectByName('link_SG3_gripper_'+side+'_finger_aruco')!.getWorldPosition(new T.Vector3());
        const hand=id==='g1'?scene.getObjectByName('right_wrist_yaw_link')!.localToWorld(new T.Vector3(.105,0,0)):
          id==='stretch'?finger('left').lerp(finger('right'),.5):rig.root.children[2].localToWorld(new T.Vector3(0,.5,0));
        return hand.distanceTo(target);
      }
      const wipe=replay.wipes?.[0];
      if(wipe){
        assert.equal(wipe.target,'table');assert.ok(replay.landmarks?.some(l=>l.fixture==='practice-table'));
        for(let t=wipe.start+2;t<wipe.start+8;t+=.125){
          // The cloth owns the hand during a wipe preceding a bowl pickup.
          if(!dish)assert.ok(pose(t)<.035);else pose(t);
          assert.ok(Math.abs(rig.cloth.position.z-.711)<1e-5);
        }
      }
      if(!track||!dish)return;
      let gap=0;
      for(let t=track.grasp;t<=track.release;t+=.125)gap=Math.max(gap,pose(t));
      assert.ok(gap<.035,'worst grasp error '+gap.toFixed(4)+' m');
      pose(track.lift+1.9);assert.ok(dish.vessel.position.z>track.pick[2]+.09);
      pose(replay.duration);assert.ok(dish.vessel.position.distanceTo(new T.Vector3(...track.place))<1e-5);
      if(track.kind==='glass'){
        assert.ok(dish.water.visible&&!dish.stream.visible);
        const full=dish.water.scale.y;
        pose((track.fillStart!+track.fillEnd!)/2);
        assert.ok(dish.stream.visible&&Math.abs(dish.water.scale.y-full*.5)<1e-5);
      }
      pose(0);assert.ok(!dish.water.visible&&!dish.stream.visible);
      assert.ok(dish.vessel.position.distanceTo(new T.Vector3(...track.pick))<1e-5);
    });
  }
}
