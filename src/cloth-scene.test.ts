import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import * as T from 'three';
import {ClothScene} from './cloth-scene';
import {parseReplay} from './replay';

test('the Stretch table wipe displays measured cloth lift and wiped marks',()=>{
  const file=new URL('../rollouts/kitchen/wipe-the-table/stretch.json',import.meta.url);
  const replay=parseReplay(JSON.parse(fs.readFileSync(file,'utf8')).replay);
  assert.equal(replay.physics?.heldOutPassed,12);
  const cloth=replay.cloth!;
  const scene=new ClothScene(cloth);
  const fabric=scene.root.children[0] as T.Mesh<T.BufferGeometry>;
  const lifted=cloth.frames.find(frame=>frame.points.every((value,i)=>i%3!==2||value>.73));
  assert.ok(lifted);
  scene.update(lifted.time);
  const measured=fabric.geometry.getAttribute('position').array;
  assert.ok(lifted.points.every((value,i)=>Math.abs(value-measured[i])<1e-6));

  scene.update(replay.duration);
  const final=cloth.frames.at(-1)!;
  assert.ok(final.wiped.length>=8);
  assert.equal(scene.root.children.slice(2).filter(mark=>!mark.visible).length,final.wiped.length);
  scene.update(0);
  assert.ok(scene.root.children.slice(2).every(mark=>mark.visible));
});
