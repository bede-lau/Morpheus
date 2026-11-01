import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { groundDemoReplay, createPlacedDemoReplay } from './demo-placement';
import { createDemoReplay, type Replay } from './replay';
import { RobotScene } from './robot-scene';
import type { RoomViewer } from './viewer';

test('sample placement follows floor height and preserves backend trajectories',()=>{
 const floor=new THREE.Mesh(new THREE.PlaneGeometry(10,10),new THREE.MeshBasicMaterial({side:THREE.DoubleSide}));floor.position.z=-.12;
 const replay=groundDemoReplay(createDemoReplay('stretch'),floor);
 // Wheel bottom = 0.01; retain 0.03 m over rough reference geometry.
 assert.ok(replay.frames.every(f=>Math.abs(f.position[2]+.1)<1e-6));
 const backend={...replay,source:'backend' as const};assert.equal(groundDemoReplay(backend,floor),backend);
 assert.throws(()=>groundDemoReplay(createDemoReplay('stretch'),new THREE.Group()),/floor support/);
});

const meshUrl=new URL('../public/samples/rustic-kitchen-collider.glb',import.meta.url);
const manifestUrl=new URL('../output/kitchen/scene.json',import.meta.url);
test('sample robot volumes clear the actual kitchen mesh throughout their replays', {skip:!fs.existsSync(meshUrl)||!fs.existsSync(manifestUrl)},async()=>{
 const data=fs.readFileSync(meshUrl);
 const {scene:room}=await new GLTFLoader().parseAsync(data.buffer.slice(data.byteOffset,data.byteOffset+data.byteLength),'');
 const manifest=JSON.parse(fs.readFileSync(manifestUrl,'utf8'));
 room.applyMatrix4(new THREE.Matrix4().fromArray(manifest.source.transform.matrix));room.updateMatrixWorld(true);
 // Index world triangles, including cabinets and the uncut roof. Render cutaway is irrelevant to clearance.
 const grid=new Map<string,Set<THREE.Triangle>>();
 function cells(bounds:THREE.Box3){const keys:string[]=[];for(let x=Math.floor(bounds.min.x*2);x<=Math.floor(bounds.max.x*2);x++)for(let y=Math.floor(bounds.min.y*2);y<=Math.floor(bounds.max.y*2);y++)for(let z=Math.floor(bounds.min.z*2);z<=Math.floor(bounds.max.z*2);z++)keys.push(`${x},${y},${z}`);return keys;}
 room.traverse(o=>{
   if(!(o instanceof THREE.Mesh))return;
   (o.material as THREE.Material).side=THREE.DoubleSide;
   const p=o.geometry.attributes.position,ix=o.geometry.index;
   for(let i=0;i<(ix?.count??p.count);i+=3){
     const points=[0,1,2].map(j=>new THREE.Vector3().fromBufferAttribute(p,ix?ix.getX(i+j):i+j).applyMatrix4(o.matrixWorld));
     const triangle=new THREE.Triangle(...points as [THREE.Vector3,THREE.Vector3,THREE.Vector3]);
     for(const key of cells(new THREE.Box3().setFromPoints(points))){if(!grid.has(key))grid.set(key,new Set());grid.get(key)!.add(triangle)}
   }
 });
 const viewer={scene:new THREE.Scene(),camera:new THREE.PerspectiveCamera(),orbit:{target:new THREE.Vector3()}} as RoomViewer;
 const simulation=new RobotScene(viewer);
 function firstIntersection(replay:Replay){
   simulation.setReplay(replay);
   for(let i=0;i<=480;i++){
     const time=i*.05;simulation.update(time);simulation.robot.updateMatrixWorld(true);
     let hit:string|undefined;
     simulation.robot.traverse(o=>{
       if(hit||!(o instanceof THREE.Mesh)||o.userData.decoration)return;
       const bounds=new THREE.Box3().setFromObject(o).expandByScalar(.005);
       for(const key of cells(bounds)){for(const triangle of grid.get(key)??[]){if(bounds.intersectsTriangle(triangle)){hit=`${replay.robot} at ${time.toFixed(2)}s (${o.geometry.type})`;return}}}
     });
     if(hit)return hit;
   }
 }
 // Regression proof: the old center-only route passes through a cabinet.
 const legacy=createDemoReplay('stretch');legacy.frames=legacy.frames.map((f,i)=>({...f,position:[1.5,[-1.5,-.7,.1,.8,1.5,1.5][i],.06]}));
 assert.ok(firstIntersection(legacy),'The reference test must detect the original clipping route.');
 for(const robot of ['stretch','go2','soarm'] as const){assert.equal(firstIntersection(createPlacedDemoReplay(robot,room)),undefined);}
 simulation.dispose();
});
