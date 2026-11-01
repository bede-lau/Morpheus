import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { apiPath, getJson, parseBackendReplay, parseCatalogue, parseEnvironment } from './backend';
import { RobotScene } from './robot-scene';
import type { RoomViewer } from './viewer';
import type { Replay } from './replay';

function environment(){return {version:1,id:'kitchen',revision:'scene-revision-1',scene:{version:1,name:'Kitchen',units:'m',upAxis:'Z',source:{scaleConfirmed:false},objects:[{id:'floor',name:'Floor',type:'box',position:[0,0,-.05],size:[5,5,.1],role:'floor'}]},validation:{compile:'passed',dropProbe:{status:'passed'}},visual:{splatUrl:'/api/environments/kitchen/scan.spz',meshUrl:null,sourceUpAxis:'Y-down',transform:{matrixOrder:'column-major',matrix:new THREE.Matrix4().toArray()}},replays:{go2:{status:'unavailable',url:null,reason:'Not published'},soarm:{status:'unavailable',url:null},stretch:{status:'available',url:'/api/environments/kitchen/replays/stretch'}}};}
function envelope(){return {environmentId:'kitchen',environmentRevision:'scene-revision-1',replay:{version:1,source:'backend',robot:'stretch',duration:1,frames:[{time:0,position:[2,1,.1],yaw:0,armExtension:0},{time:1,position:[2,2,.1],yaw:0,armExtension:.3}]} as Replay};}

test('backend scene retains exact geometry, transform and availability',()=>{
 const data=environment(),scene=parseEnvironment(data);
 assert.deepEqual(scene.scene.objects,data.scene.objects);assert.deepEqual(scene.visual.transform.matrix,data.visual.transform.matrix);
 assert.equal(scene.replays.go2.url,null);assert.equal(scene.replays.stretch.status,'available');assert.equal(scene.scene.source.scaleConfirmed,false);
 assert.equal(parseCatalogue({version:1,environments:[{id:'kitchen',name:'Kitchen',revision:'scene-revision-1',url:'/api/environments/kitchen'}]})[0].id,'kitchen');
});
test('invalid scene alignment and missing replay URLs fail explicitly',()=>{
 const singular=environment();singular.visual.transform.matrix.fill(0);assert.throws(()=>parseEnvironment(singular),/transform/);
 const noUrl=environment();(noUrl.replays.stretch as any).url=null;assert.throws(()=>parseEnvironment(noUrl),/no URL/);
 const wrongAxis=environment();wrongAxis.visual.sourceUpAxis='unknown';assert.throws(()=>parseEnvironment(wrongAxis),/axis/);
 assert.throws(()=>apiPath('https://other.example/api/replays'));
 assert.throws(()=>apiPath('/api/../secrets'));
});
test('real replay must match scene revision, robot and backend provenance',()=>{
 const env=parseEnvironment(environment());assert.deepEqual(parseBackendReplay(envelope(),env,'stretch'),envelope().replay);
 for(const change of [(e:any)=>e.environmentRevision='old',(e:any)=>e.environmentId='other',(e:any)=>e.replay.robot='go2',(e:any)=>e.replay.source='demo']){const e=envelope();change(e);assert.throws(()=>parseBackendReplay(e,env,'stretch'));}
});
test('HTTP failures are not replaced with demo data',async t=>{
 t.mock.method(globalThis,'fetch',async()=>new Response(JSON.stringify({error:'No teammate rollout supplied'}),{status:404}));
 await assert.rejects(()=>getJson('/api/environments/kitchen/replays/stretch'),/No teammate rollout/);
});
test('backend poses render without procedural movement and clear removes stale robot',()=>{
 const viewer={scene:new THREE.Scene(),camera:new THREE.PerspectiveCamera(),orbit:{target:new THREE.Vector3()}} as RoomViewer;
 const scene=new RobotScene(viewer);scene.clearReplay();assert.equal(scene.root.visible,false);
 const replay=envelope().replay;replay.robot='go2';scene.setReplay(replay);scene.update(.5);
 assert.deepEqual(scene.robot.position.toArray(),[2,1.5,.1]);assert.ok(scene.legs.every(leg=>leg.rotation.x===0));
 scene.clearReplay();scene.update(.8);assert.equal(scene.root.visible,false);assert.deepEqual(scene.robot.position.toArray(),[2,1.5,.1]);scene.dispose();
});
