import test from 'node:test';
import assert from 'node:assert/strict';
import { parsePlanResult, parseTaskEnvelope, parseTaskList, planTask, slugify, taskUrl } from './tasks';
import { currentEvent, parseReplay } from './replay';

const env={id:'kitchen',revision:'rev-1'};
function envelope(robot='stretch'){return {environmentId:'kitchen',environmentRevision:'rev-1',task:'wipe the table',taskSlug:'wipe-the-table',replay:{version:1,source:'backend',robot,duration:2,frames:[{time:0,position:[0,0,0],yaw:0,armExtension:0},{time:1,position:[0,1,0],yaw:0,armExtension:0},{time:2,position:[0,1,0],yaw:0,armExtension:.8}],task:'wipe the table',events:[{time:0,label:'Spawned at the kitchen door'},{time:1,label:'Navigating to the table'},{time:1.5,label:'Wiping'}],verdict:{status:'completed',summary:'Done.',steps:[{name:'Navigate to table',status:'done'},{name:'Wipe',status:'done',note:'0.6 m patch'}],planner:'rules'},landmarks:[{id:'table',label:'Table',position:[0,1.5,.4],size:[1,.8,.8],kind:'table'}]}};}

test('task envelope keeps task fields, verdict, events and landmarks',()=>{
 const parsed=parseTaskEnvelope(envelope(),env,'stretch');
 assert.equal(parsed.task,'wipe the table');assert.equal(parsed.taskSlug,'wipe-the-table');
 assert.equal(parsed.replay.verdict?.status,'completed');assert.equal(parsed.replay.events?.length,3);assert.equal(parsed.replay.landmarks?.[0].kind,'table');
 assert.throws(()=>parseTaskEnvelope(envelope('go2'),env,'stretch'));
 assert.throws(()=>parseTaskEnvelope({...envelope(),environmentRevision:'old'},env,'stretch'),/revision/);
});
test('replays without task fields still parse; malformed task fields fail',()=>{
 const plain=envelope().replay as any;delete plain.events;delete plain.verdict;delete plain.landmarks;delete plain.task;assert.equal(parseReplay(plain).robot,'stretch');
 for(const change of [(r:any)=>r.events=[{time:NaN,label:'x'}],(r:any)=>r.events[0].label=3,(r:any)=>r.verdict.status='validated',(r:any)=>r.verdict.steps[0].status='ok',(r:any)=>r.landmarks[0].position=[0,0],(r:any)=>r.task=5]){const r=envelope().replay as any;change(r);assert.throws(()=>parseReplay(r));}
});
test('currentEvent returns the latest event at or before the time',()=>{
 const r=envelope().replay as any;
 assert.equal(currentEvent(r,0)?.label,'Spawned at the kitchen door');assert.equal(currentEvent(r,1.2)?.label,'Navigating to the table');assert.equal(currentEvent(r,9)?.label,'Wiping');
 assert.equal(currentEvent(r,-1),null);assert.equal(currentEvent({},1),null);assert.equal(currentEvent(null,1),null);
});
test('task list and plan responses are validated',()=>{
 const list=parseTaskList({version:1,tasks:[{slug:'wipe-the-table',task:'wipe the table',robots:{stretch:'/api/tasks/wipe-the-table/stretch',go2:'https://evil/x'}},{slug:'../bad',task:'x',robots:{}}]});
 assert.equal(list.length,1);assert.equal(list[0].robots.stretch,'/api/tasks/wipe-the-table/stretch');assert.equal(list[0].robots.go2,undefined);
 assert.throws(()=>parseTaskList({tasks:[]}));
 assert.equal(parsePlanResult({version:1,slug:'s',task:'t',results:{stretch:{}}}).slug,'s');
 assert.equal(slugify('Fill a glass of water!'),'fill-a-glass-of-water');assert.equal(taskUrl('wipe-the-table','go2'),'/api/tasks/wipe-the-table/go2');assert.throws(()=>taskUrl('../x','go2'));
});
test('planning posts the task and surfaces backend errors',async t=>{
 const calls:any[]=[];t.mock.method(globalThis,'fetch',async(url:string,init:any)=>{calls.push([url,JSON.parse(init.body)]);return new Response(JSON.stringify({error:'Planner offline'}),{status:503})});
 await assert.rejects(()=>planTask('wipe the table','go2'),/Planner offline/);assert.deepEqual(calls[0],['/api/tasks/plan',{task:'wipe the table',robot:'go2'}]);
 await assert.rejects(()=>planTask('x'.repeat(201)),/200/);
});
