import {test} from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import {ObjectScene,fitScale} from './object-scene';
import type {RoomViewer} from './viewer';

test('fitting preserves the library asset proportions',()=>{
 assert.equal(fitScale([.4,.4,.1],[.2,.2,.05]),2);
 assert.throws(()=>fitScale([1,1,1],[0,0,0]));
});

test('replacement hides the selected world box and undo preserves existing edits',()=>{
 const existing={};const scene=new THREE.Scene();
 const fake={scene,splat:{edits:[existing],updateVersion(){}},renderer:{domElement:{addEventListener(){},removeEventListener(){}}}} as unknown as RoomViewer;
 const controller=new ObjectScene(fake,()=>{});
 const selection={position:[1,2,3] as [number,number,number],size:[.4,.3,.1] as [number,number,number]};
 controller.setSelection(selection);
 controller.preview=new THREE.Group();controller.preview.position.set(1,2,3);controller.preview.scale.setScalar(2);
 controller.previewAsset={id:'plate',name:'plate',category:'plate',sourceUrl:'source',revision:'revision',dimensions:[.2,.2,.05],triangles:20,license:'CC-BY-4.0'};
 const result=controller.apply('manual');
 assert.deepEqual(result.selection,selection);assert.equal(result.scale,2);
 assert.equal(fake.splat!.edits!.length,2);
 const edit=fake.splat!.edits![1];assert.deepEqual(edit.children[0].position.toArray(),[1,2,3]);
 assert.deepEqual(edit.children[0].scale.toArray(),[.2,.15,.05]);
 controller.undo();assert.deepEqual(fake.splat!.edits,[existing]);assert.equal(controller.replacements.length,0);
 controller.dispose();
});
