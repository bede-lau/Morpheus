import test from 'node:test';
import assert from 'node:assert/strict';
import { Matrix4 } from 'three';
import { parseSavedObjects } from './saved-objects';
import type { BackendEnvironment } from './backend';
const matrix=new Matrix4().makeTranslation(1,2,3).toArray();
const environment={visual:{sourceSha256:'abc',transform:{matrix}}} as unknown as unknown as BackendEnvironment;
function saved(){return {version:1,kind:'simready-replacements',revision:1,units:'m',upAxis:'Z',interactionValidated:false,source:{sha256:'abc',transform:{matrixOrder:'column-major',matrix:[...matrix]}},replacements:[{id:'object-1',assetId:'mug',asset:{id:'mug'},position:[2,3,4],quaternion:[1,0,0,0],scale:1,selection:{position:[2,3,4],size:[.3,.2,.1]},chosenBy:'manual'}]};}
test('saved replacements preserve world poses and require matching source and alignment',()=>{
 const data=saved();assert.deepEqual(parseSavedObjects(data,environment).replacements,data.replacements);
 const wrongScan=saved();wrongScan.source.sha256='other';assert.throws(()=>parseSavedObjects(wrongScan,environment),/source scan/);
 const wrongFrame=saved();wrongFrame.source.transform.matrix[12]+=1;assert.throws(()=>parseSavedObjects(wrongFrame,environment),/alignment/);
 assert.throws(()=>parseSavedObjects(data,{visual:{transform:{matrix}}} as unknown as unknown as BackendEnvironment),/unverified/);
});
test('unsafe saved poses and masks fail before any source scan is hidden',()=>{
 for(const change of [(v:any)=>v.replacements[0].scale=-1,(v:any)=>v.replacements[0].quaternion=[0,0,0,0],(v:any)=>v.replacements[0].selection.size=[0,1,1],(v:any)=>v.replacements[0].position=[NaN,0,0],(v:any)=>v.replacements.push(v.replacements[0])]){const data=saved();change(data);assert.throws(()=>parseSavedObjects(data,environment));}
 assert.deepEqual(parseSavedObjects({...saved(),source:null,replacements:[],revision:0},environment),{revision:0,replacements:[]});
});
