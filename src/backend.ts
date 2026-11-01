import { Matrix4 } from 'three';
import { validateManifest } from './export';
import { parseReplay, ROBOTS, type Replay, type RobotId } from './replay';
import type { SceneManifest } from './types';

export type EnvironmentSummary={id:string;name:string;revision:string;url:string};
export type ReplayAvailability={status:string;url:string|null;reason?:string};
export type BackendEnvironment={version:1;id:string;revision:string;scene:SceneManifest;validation:Record<string,unknown>;visual:{sourceSha256?:string;splatUrl:string|null;meshUrl:string|null;sourceUpAxis:string;transform:{matrix:number[];matrixOrder:'column-major'}};replays:Record<RobotId,ReplayAvailability>};
const object=(v:unknown):v is Record<string,any>=>!!v&&typeof v==='object'&&!Array.isArray(v);
export function apiPath(value:unknown):string{
 if(typeof value!=='string'||!/^\/api\/[A-Za-z0-9_/?=&.%-]+$/.test(value)||value.includes('..'))throw new Error('Backend returned an invalid API path.');
 return value;
}
function assetPath(value:unknown):string|null{
 if(value===null||value===undefined)return null;
 if(typeof value!=='string'||(!/^https:\/\//.test(value)&&!/^\/(?!\/)/.test(value)))throw new Error('Backend returned an invalid visual asset URL.');
 return value;
}
export function parseCatalogue(data:unknown):EnvironmentSummary[]{
 if(!object(data)||data.version!==1||!Array.isArray(data.environments))throw new Error('Unsupported environment catalogue.');
 return data.environments.map((item:unknown)=>{if(!object(item)||typeof item.id!=='string'||typeof item.name!=='string'||typeof item.revision!=='string')throw new Error('Invalid environment entry.');return {id:item.id,name:item.name,revision:item.revision,url:apiPath(item.url)};});
}
export function parseEnvironment(data:unknown):BackendEnvironment{
 if(!object(data)||data.version!==1||typeof data.id!=='string'||typeof data.revision!=='string'||!data.revision)throw new Error('Unsupported backend environment.');
 const errors=validateManifest(data.scene);if(errors.length)throw new Error(`Invalid backend scene: ${errors.join(' ')}`);
 const visual=data.visual;if(!object(visual)||!object(visual.transform))throw new Error('The environment is missing its visual alignment.');
 const matrix=visual.transform.matrix;
 if(visual.transform.matrixOrder!=='column-major'||!Array.isArray(matrix)||matrix.length!==16||!matrix.every(Number.isFinite)||Math.abs(new Matrix4().fromArray(matrix).determinant())<1e-12)throw new Error('Invalid scene transform. Expected an invertible column-major matrix.');
 if(!['Y-down','Y-up','Z-up'].includes(visual.sourceUpAxis))throw new Error('Unknown visual source axis.');
 if(!object(data.replays)||!object(data.validation))throw new Error('The environment is missing its replay or validation status.');
 const replays={} as BackendEnvironment['replays'];
 for(const {id}of ROBOTS){const item=data.replays[id];if(!object(item)||typeof item.status!=='string')throw new Error(`Missing replay status for ${id}.`);replays[id]={status:item.status,url:item.url===null||item.url===undefined?null:apiPath(item.url),reason:typeof item.reason==='string'?item.reason:undefined};if(item.status==='available'&&!replays[id].url)throw new Error(`Ready replay ${id} has no URL.`);}
 return {...data,visual:{...visual,splatUrl:assetPath(visual.splatUrl),meshUrl:assetPath(visual.meshUrl)},replays} as BackendEnvironment;
}
export function parseBackendReplay(data:unknown,environment:Pick<BackendEnvironment,'id'|'revision'>,robot:RobotId):Replay{
 if(!object(data)||data.environmentId!==environment.id||data.environmentRevision!==environment.revision)throw new Error('Replay belongs to a different environment revision. Refresh or regenerate the rollout.');
 const replay=parseReplay(data.replay);
 if(replay.source!=='backend'||replay.robot!==robot)throw new Error('Backend replay does not match the selected robot or has demo provenance.');
 return replay;
}
export async function getJson(path:string,signal?:AbortSignal):Promise<unknown>{
 const response=await fetch(apiPath(path),{signal:signal?AbortSignal.any([signal,AbortSignal.timeout(15000)]):AbortSignal.timeout(15000),cache:'no-store'});
 const data=await response.json().catch(()=>null);
 if(!response.ok)throw new Error(object(data)&&typeof data.error==='string'?data.error:`Backend request failed (${response.status}).`);
 if(data===null)throw new Error('Backend returned an empty response.');return data;
}
