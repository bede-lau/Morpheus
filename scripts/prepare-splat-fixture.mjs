// Exercise the same conservative geometry algorithm using SPZ centres alone.
// This fixture decoder deliberately supports only our verified SPZ-v2 sample.
import {readFile,mkdir,writeFile} from 'node:fs/promises';
import {gunzipSync} from 'node:zlib';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import {computeBounds,estimateFloorZ,transformPoints,proposeColliders} from '../src/geometry.ts';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const data=gunzipSync(await readFile(path.join(root,'public/samples/rustic-kitchen-500k.spz')));
if(data.readUInt32LE(0)!==0x5053474e||data.readUInt32LE(4)!==2)throw new Error('This fixture expects SPZ v2. Use the browser importer for other formats.');
const count=data.readUInt32LE(8),fractionalBits=data[13],factor=2**-fractionalBits,opacityStart=16+count*9,scaleStart=16+count*13;
const out=[],stride=Math.max(1,Math.ceil(count/90000));
function fixed(index){const offset=16+index*3;let v=data[offset]|data[offset+1]<<8|data[offset+2]<<16;if(v&0x800000)v-=0x1000000;return v*factor}
for(let i=0;i<count;i+=stride){if(data[opacityStart+i]/255<.3)continue;let maxScale=0;for(let axis=0;axis<3;axis++)maxScale=Math.max(maxScale,Math.exp(data[scaleStart+i*3+axis]/16-10));if(maxScale>.3)continue;out.push(fixed(i*3),fixed(i*3+2),-fixed(i*3+1));}
const raw=new Float32Array(out),bounds=computeBounds(raw,.008),floor=estimateFloorZ(raw),translation=[-bounds.center[0],-bounds.center[1],-floor],points=transformPoints(raw,1,translation),b=computeBounds(points,.008);
const objects=proposeColliders(points,{cellSize:.22,floorZ:0,ceilingZ:b.max[2],maxBoxes:140,minPoints:3,trimFraction:.008});
const scene={version:1,name:'Rustic kitchen — splat-only fixture',units:'m',upAxis:'Z',source:{kind:'sample',synthetic:true,fileName:'rustic-kitchen-500k.spz',url:'https://docs.worldlabs.ai/marble/export/specs',attribution:'World Labs — Rustic kitchen with natural light',scaleConfirmed:false,scaleStatus:'assumed',reviewed:false,geometrySource:'splat-centers',sampledPoints:points.length/3,approximation:'Occupied columns; unseen free space and under-table gaps are not established.',transform:{matrix:[1,0,0,0,0,0,-1,0,0,1,0,0,...translation,1],matrixOrder:'column-major',unitsPerSourceUnit:1}},objects};
await mkdir(path.join(root,'output/splat-only'),{recursive:true});await writeFile(path.join(root,'output/splat-only/scene.json'),JSON.stringify(scene,null,2));
console.log(JSON.stringify({path:'output/splat-only/scene.json',shapes:objects.length,sampledPoints:points.length/3,bounds:b.size,scale:'assumed'}));
