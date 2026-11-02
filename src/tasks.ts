import { apiPath, getJson, parseBackendReplay, type BackendEnvironment } from './backend';
import { ROBOTS, type Replay, type RobotId } from './replay';

export type TaskSummary={slug:string;task:string;robots:Partial<Record<RobotId,string>>};
export type TaskEnvelope={task:string;taskSlug:string;replay:Replay};
export type PlanResult={slug:string;task:string;results:Partial<Record<RobotId,unknown>>};
export const PRESET_TASKS=['Wipe the table','Fill a glass of water'];
export const MAX_TASK_LENGTH=200;
const object=(v:unknown):v is Record<string,any>=>!!v&&typeof v==='object'&&!Array.isArray(v);
const SLUG=/^[a-z0-9-]{1,120}$/;
export function slugify(task:string){return task.toLowerCase().replace(/[^a-z0-9]+/g,'-').replace(/^-+|-+$/g,'').slice(0,80)||'task'}
/** `?env=<id>` selects a non-kitchen environment; the kitchen is the server default. */
export function envQuery(env?:string){return env&&env!=='kitchen'?`?env=${encodeURIComponent(env)}`:''}
export function taskUrl(slug:string,robot:RobotId,env?:string){if(!SLUG.test(slug))throw new Error('Invalid task slug.');return `/api/tasks/${slug}/${robot}${envQuery(env)}`}
export function parseTaskList(data:unknown):TaskSummary[]{
 if(!object(data)||data.version!==1||!Array.isArray(data.tasks))throw new Error('Unsupported task list.');
 return data.tasks.filter((t:unknown)=>object(t)&&typeof t.slug==='string'&&SLUG.test(t.slug)&&typeof t.task==='string').map((t:Record<string,any>)=>{const robots:TaskSummary['robots']={};if(object(t.robots))for(const {id}of ROBOTS){const url=t.robots[id];if(typeof url==='string')try{robots[id]=apiPath(url)}catch{/* skip invalid */}}return {slug:t.slug,task:t.task,robots};});
}
/** Parse a task envelope: the READ_API replay envelope plus top-level `task`/`taskSlug`. */
export function parseTaskEnvelope(data:unknown,environment:Pick<BackendEnvironment,'id'|'revision'>,robot:RobotId):TaskEnvelope{
 const replay=parseBackendReplay(data,environment,robot);const d=data as Record<string,unknown>;
 const task=typeof d.task==='string'?d.task:replay.task??'';
 const taskSlug=typeof d.taskSlug==='string'?d.taskSlug:slugify(task);
 return {task,taskSlug,replay:{...replay,task:replay.task??task}};
}
export function parsePlanResult(data:unknown):PlanResult{
 if(!object(data)||data.version!==1||typeof data.slug!=='string'||!object(data.results))throw new Error('Unsupported plan response.');
 return {slug:data.slug,task:typeof data.task==='string'?data.task:'',results:data.results};
}
export async function fetchTasks(signal?:AbortSignal,env?:string){return parseTaskList(await getJson(`/api/tasks${envQuery(env)}`,signal))}
export async function fetchTaskEnvelope(slug:string,robot:RobotId,environment:Pick<BackendEnvironment,'id'|'revision'>,signal?:AbortSignal){return parseTaskEnvelope(await getJson(taskUrl(slug,robot,environment.id),signal),environment,robot)}
export async function planTask(task:string,robot?:RobotId,signal?:AbortSignal,env?:string):Promise<PlanResult>{
 const text=task.trim();if(!text)throw new Error('Describe a task first.');if(text.length>MAX_TASK_LENGTH)throw new Error(`Tasks are limited to ${MAX_TASK_LENGTH} characters.`);
 const timeout=AbortSignal.timeout(120000);
 const response=await fetch(apiPath('/api/tasks/plan'),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({task:text,...(robot?{robot}:{}),...(env&&env!=='kitchen'?{env}:{})}),signal:signal?AbortSignal.any([signal,timeout]):timeout,cache:'no-store'});
 const data=await response.json().catch(()=>null);
 if(!response.ok)throw new Error(object(data)&&typeof data.error==='string'?data.error:`Planning failed (${response.status}).`);
 return parsePlanResult(data);
}
