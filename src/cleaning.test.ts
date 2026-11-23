import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import * as T from 'three';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import {Cleaning} from './cleaning';
import {ArticulatedRobot} from './robot-model';
import {sampleReplay, type Replay} from './replay';

for(const id of ['g1','stretch','go2'] as const){
  test(id+' holds the cloth while it wipes both kitchen surfaces',async()=>{
    const replay:Replay=JSON.parse(fs.readFileSync(new URL('../rollouts/kitchen/wipe-the-table-and-clean-the-sink/'+id+'.json',import.meta.url),'utf8')).replay;
    const geometry=fs.readFileSync(new URL('./robots/'+id+'.glb',import.meta.url));
    const {scene}=await new GLTFLoader().parseAsync(geometry.buffer.slice(geometry.byteOffset,geometry.byteOffset+geometry.byteLength),'');
    const model=new ArticulatedRobot(scene,JSON.parse(fs.readFileSync(new URL('./robots/'+id+'.json',import.meta.url),'utf8')));
    const robot=new T.Group();robot.add(scene);
    const cleaning=new Cleaning(replay);
    assert.equal(replay.wipes?.length,2);
    function poseAt(t:number){
      const pose=sampleReplay(replay,t);
      robot.position.fromArray(pose.position);robot.rotation.z=pose.yaw;
      cleaning.update(t,robot,model);
      const grip=cleaning.cloth.children[2].localToWorld(new T.Vector3(0,.5,0));
      let hand:T.Vector3;
      if(id==='g1')hand=scene.getObjectByName('right_wrist_yaw_link')!.localToWorld(new T.Vector3(.105,0,0));
      else if(id==='stretch'){
        const finger=(side:string)=>scene.getObjectByName('link_SG3_gripper_'+side+'_finger_aruco')!.getWorldPosition(new T.Vector3());
        hand=finger('left').lerp(finger('right'),.5);
      }else hand=cleaning.root.children[2].localToWorld(new T.Vector3(0,.5,0));
      return {gap:hand.distanceTo(grip),pad:cleaning.cloth.position.clone()};
    }
    const [table,sink]=replay.wipes!;
    const initial=poseAt(0).pad;
    assert.ok(poseAt(table.start+1.5).pad.z>initial.z+.10,'cloth lifts after grasp');
    for(let t=table.start+1;t<sink.end-1.5;t+=.125){
      const {gap}=poseAt(t);
      assert.ok(gap<.035,id+' hand gap '+gap.toFixed(3)+' m at '+t.toFixed(2)+' s');
    }
    for(const step of [table,sink]){
      const positions:T.Vector3[]=[];
      for(let t=step.start+2;t<=step.start+8;t+=.125){
        const {pad}=poseAt(t);
        assert.ok(Math.abs(pad.z-step.point[2])<1e-5);
        positions.push(pad);
      }
      assert.ok(positions.some(p=>p.distanceTo(positions[0])>.15));
    }
    const released=poseAt(sink.end).pad;
    assert.ok(poseAt(replay.duration).pad.distanceTo(released)<1e-6);
    assert.ok(poseAt(0).pad.distanceTo(initial)<1e-6);
  });
}
