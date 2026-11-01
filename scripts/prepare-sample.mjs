// Create an explicit, assumed-scale fixture from the downloaded reference mesh.
// No learned geometry or semantics are claimed. Run with Node 24 strip-types.
import {readFile,mkdir,writeFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import {computeBounds,estimateFloorZ,transformPoints,proposeColliders} from '../src/geometry.ts';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const buffer=await readFile(path.join(root,'public/samples/rustic-kitchen-collider.glb'));
if(buffer.readUInt32LE(0)!==0x46546c67)throw new Error('Expected GLB');
const jsonLength=buffer.readUInt32LE(12),gltf=JSON.parse(buffer.subarray(20,20+jsonLength).toString());
const binary=buffer.subarray(20+jsonLength+8);
function accessor(index){const a=gltf.accessors[index],view=gltf.bufferViews[a.bufferView],offset=(view.byteOffset||0)+(a.byteOffset||0);const components={SCALAR:1,VEC3:3}[a.type],bytes={5123:2,5125:4,5126:4}[a.componentType],stride=view.byteStride||bytes*components;const result=[];for(let i=0;i<a.count;i++){const row=[];for(let j=0;j<components;j++){const o=offset+i*stride+j*bytes;row.push(a.componentType===5126?binary.readFloatLE(o):a.componentType===5125?binary.readUInt32LE(o):binary.readUInt16LE(o));}result.push(components===1?row[0]:row)}return result}
const primitive=gltf.meshes[0].primitives[0],vertices=accessor(primitive.attributes.POSITION).map(([x,y,z])=>[x,z,-y]),indices=accessor(primitive.indices),triangles=[];let total=0;
for(let i=0;i<indices.length;i+=3){const[a,b,c]=[vertices[indices[i]],vertices[indices[i+1]],vertices[indices[i+2]]],u=b.map((v,j)=>v-a[j]),v=c.map((v,j)=>v-a[j]);const area=Math.hypot(u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0])/2;if(area>0){triangles.push({a,b,c,area});total+=area}}
const points=[];let budget=0,index=0;
for(const t of triangles){budget+=t.area/total*90000;const n=Math.floor(budget);budget-=n;for(let j=0;j<n;j++){index++;const u=Math.sqrt(index*.618033988749895%1),v=index*.414213562373095%1;for(let axis=0;axis<3;axis++)points.push(t.a[axis]*(1-u)+t.b[axis]*u*(1-v)+t.c[axis]*u*v)}}
const raw=new Float32Array(points),bounds=computeBounds(raw,.008),floor=estimateFloorZ(raw),translation=[-bounds.center[0],-bounds.center[1],-floor];
const aligned=transformPoints(raw,1,translation),normalized=computeBounds(aligned,.008),objects=proposeColliders(aligned,{cellSize:.22,floorZ:0,ceilingZ:normalized.max[2],maxBoxes:140,minPoints:3,trimFraction:.008});
const manifest={version:1,name:'Rustic kitchen — assumed-scale fixture',units:'m',upAxis:'Z',source:{kind:'sample',synthetic:true,fileName:'rustic-kitchen-500k.spz',url:'https://docs.worldlabs.ai/marble/export/specs',attribution:'World Labs — Rustic kitchen with natural light',scaleConfirmed:false,scaleStatus:'assumed',reviewed:false,geometrySource:'companion-mesh',approximation:'Occupied columns; space below elevated surfaces is filled. Not semantic objects.',cellSizeMeters:.22,transform:{matrix:[1,0,0,0,0,0,-1,0,0,1,0,0,...translation,1],matrixOrder:'column-major',unitsPerSourceUnit:1}},objects};
await mkdir(path.join(root,'output/kitchen'),{recursive:true});await writeFile(path.join(root,'output/kitchen/scene.json'),JSON.stringify(manifest,null,2));
console.log(JSON.stringify({scene:path.join(root,'output/kitchen/scene.json'),shapes:objects.length,bounds:normalized.size,scale:'assumed',reviewed:false}));
