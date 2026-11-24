import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import * as T from 'three';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import {ArticulatedRobot} from './robot-model';
import {ContactScene} from './contact-scene';
import {parseReplay,sampleReplay} from './replay';

for(const [id,slug] of [
  ['g1','aisle-clearance'],['go2','aisle-clearance'],['stretch','aisle-clearance'],
  ['g1','counter-reach'],['go2','counter-reach'],['stretch','counter-reach'],
  ['go2','drawer-opening'],['stretch','drawer-opening'],
]){
  test(id+' '+slug+' uses recorded MuJoCo state',async()=>{
    const replay=parseReplay(JSON.parse(fs.readFileSync(new URL('../rollouts/kitchen/'+slug+'/'+id+'.json',import.meta.url),'utf8')).replay);
    const physics=replay.physics!;
    const asset=physics.visualModel??id;
    const data=fs.readFileSync(new URL('./robots/'+asset+'.glb',import.meta.url));
    const {scene}=await new GLTFLoader().parseAsync(data.buffer.slice(data.byteOffset,data.byteOffset+data.byteLength),'');
    const meta=JSON.parse(fs.readFileSync(new URL('./robots/'+asset+'.json',import.meta.url),'utf8'));
    const model=new ArticulatedRobot(scene,meta);
    const parent=new T.Group();
    parent.add(scene);
    assert.ok(physics.engine.startsWith('MuJoCo'));
    assert.equal(physics.heldOutPassed,12);
    const frame=sampleReplay(replay,replay.contact?replay.contact.contact+.023:.523);
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
    if(replay.contact){
      assert.ok(physics.contactDwellSeconds!>=.25);
      assert.ok(physics.peakContactForceNewtons!>1);
      assert.ok(replay.frames.some(f=>(f.contactForce??0)>1&&f.time<replay.contact!.release));
      assert.equal(replay.frames.at(-1)!.contactForce,0);
    }
    if(replay.contact?.kind==='drawer'){
      const drawer=new ContactScene(replay.contact);
      const last=replay.frames.at(-1)!;
      assert.ok(last.drawerDisplacement!>=.12);
      const opening=sampleReplay(replay,replay.contact.release-2.03);
      assert.ok(opening.drawerDisplacement!>0 && opening.drawerDisplacement!<last.drawerDisplacement!);
      drawer.update(opening.time,parent,model,null,opening);
      assert.equal(drawer.fixture.position.x,opening.drawerDisplacement);
      drawer.update(last.time,parent,model,null,last);
      assert.equal(drawer.fixture.position.x,last.drawerDisplacement);
      assert.throws(()=>parseReplay({...replay,frames:[{...replay.frames[0],drawerDisplacement:undefined},...replay.frames.slice(1)]}),/Invalid recorded drawer/);
    }
    assert.throws(()=>parseReplay({...replay,frames:[{...replay.frames[0],physicsPose:undefined},...replay.frames.slice(1)]}),/Invalid recorded physics/);
  });
}
