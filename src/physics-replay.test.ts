import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import * as T from 'three';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import {ArticulatedRobot} from './robot-model';
import {parseReplay,sampleReplay} from './replay';

for(const [id,slug] of [['g1','aisle-clearance'],['go2','aisle-clearance'],['stretch','aisle-clearance'],['stretch','counter-reach']]){
  test(id+' '+slug+' uses recorded MuJoCo state',async()=>{
    const asset=id==='g1'?'g1-12dof':id;
    const replay=parseReplay(JSON.parse(fs.readFileSync(new URL('../rollouts/kitchen/'+slug+'/'+id+'.json',import.meta.url),'utf8')).replay);
    const data=fs.readFileSync(new URL('./robots/'+asset+'.glb',import.meta.url));
    const {scene}=await new GLTFLoader().parseAsync(data.buffer.slice(data.byteOffset,data.byteOffset+data.byteLength),'');
    const meta=JSON.parse(fs.readFileSync(new URL('./robots/'+asset+'.json',import.meta.url),'utf8'));
    const model=new ArticulatedRobot(scene,meta);
    const parent=new T.Group();
    parent.add(scene);
    const physics=replay.physics!;
    assert.ok(physics.engine.startsWith('MuJoCo'));
    assert.equal(physics.heldOutPassed,12);
    const frame=sampleReplay(replay,slug==='counter-reach'?16.123:.523);
    parent.position.fromArray(frame.position);
    parent.rotation.z=frame.yaw;
    model.applyPhysicsPose(frame.physicsPose!,frame.yaw,physics.jointNames);
    const free=Object.keys(meta.bodies).find(body=>meta.bodies[body].free)!;
    const base=scene.getObjectByName(free)!;
    const [w,x,y,z]=frame.physicsPose!.quaternion;
    assert.ok(base.getWorldPosition(new T.Vector3()).distanceTo(parent.position)<1e-6);
    assert.ok(base.getWorldQuaternion(new T.Quaternion()).angleTo(new T.Quaternion(x,y,z,w))<1e-6);
    const first=replay.frames[0].physicsPose!.joints;
    assert.ok(frame.physicsPose!.joints.some((value,i)=>Math.abs(value-first[i])>.02));
    assert.throws(()=>parseReplay({...replay,frames:[{...replay.frames[0],physicsPose:undefined},...replay.frames.slice(1)]}),/Invalid recorded physics/);
  });
}
