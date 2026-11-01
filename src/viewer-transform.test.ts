import assert from 'node:assert/strict';
import test from 'node:test';
import * as THREE from 'three';
import { RoomViewer } from './viewer';

test('backend matrix replaces provisional alignment exactly, is idempotent, and keeps later scaling coherent',()=>{
  // No renderer is needed: exercise the public transform on a source root and
  // already-aligned samples, as produced by load().
  const root=new THREE.Group();root.position.set(7,-4,2);root.rotation.x=-Math.PI/2;root.updateMatrixWorld(true);
  const raw=[new THREE.Vector3(1,2,3),new THREE.Vector3(-2,1,-1)];
  const points=new Float32Array(raw.flatMap(p=>p.clone().applyMatrix4(root.matrixWorld).toArray()));
  const view=Object.assign(Object.create(RoomViewer.prototype),{sourceRoot:root,points,basePoints:points,baseRotation:new THREE.Quaternion(),offset:new THREE.Vector3(),scale:1,info:null,cutaway:null,clearMeasure(){},home(){}}) as RoomViewer;
  const matrix=new THREE.Matrix4().compose(new THREE.Vector3(2,3,4),new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1,0,0),-Math.PI/2),new THREE.Vector3(2,2,2));
  for(let iteration=0;iteration<2;iteration++){
    view.applyWorldTransform(matrix.toArray());
    const actual=view.getGeometryPoints(true);
    raw.forEach((p,i)=>assert.ok(new THREE.Vector3().fromArray(actual,i*3).distanceTo(p.clone().applyMatrix4(matrix))<1e-5));
    view.getTransform().matrix.forEach((value,i)=>assert.ok(Math.abs(value-matrix.elements[i])<1e-8));
  }
  view.setScale(4);
  const doubled=view.getGeometryPoints(true);
  raw.forEach((p,i)=>assert.ok(new THREE.Vector3().fromArray(doubled,i*3).distanceTo(p.clone().applyMatrix4(matrix).multiplyScalar(2))<1e-5));
  assert.throws(()=>view.applyWorldTransform(new THREE.Matrix4().makeScale(1,2,1).toArray()));
});
