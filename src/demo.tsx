import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { RoomViewer } from './viewer';
import { RobotScene } from './robot-scene';
import { RobotPov } from './pov';
import { SavedObjectLayer, parseSavedObjects, type SavedObjects } from './saved-objects';
import { getJson, parseEnvironment, type BackendEnvironment } from './backend';
import { fetchTasks, parseTaskEnvelope, taskUrl } from './tasks';
import { currentEvent, ROBOTS, type Replay, type RobotId, type Verdict } from './replay';
import { computeBounds, roomFootprintBounds } from './geometry';
import { suggestedCutawayHeight } from './cutaway';
import { AMBER, GREEN, backOut, captureAround, createFlash, createRing, createScanPlane, createSphereReveal, easeInOutCubic, easeOutCubic, fadeColliders, loadAsset, progress, project } from './demo-fx';
import './demo.css';

type View={view:number;file:string;width:number;height:number};
type Detection={view:number;label:string;x:number;y:number;confidence:number};
type ReportObject={label:string;view:number;pixel:[number,number];position:[number,number,number];confidence:number;result:string;asset?:string;decisions?:string};
type Prob={value:string|number;label?:string;probability:number};
type Candidate={asset:{id:string;name:string;category:string;dimensions:number[];triangles:number;sourceUrl:string;license:string};score:number;confidence:number;probabilities:Prob[]};
type Answer={type:string;name:string;choice?:string;probability?:number;probabilities?:Prob[];score?:number;confidence?:number};
type Decision={category:string;candidates:Candidate[];trace:{stage:string;answers:Answer[];requestId:string|null}[]};
type Replacement={id:string;assetId:string;position:number[];quaternion:number[];scale:number;selection:{position:number[];size:number[]};chosenBy:string;decision?:Decision;match?:ReportObject};
type Data={tasks:{slug:string;task:string}[];env:BackendEnvironment;views:View[];detections:Detection[];source:string;report:ReportObject[];replacements:Replacement[];task:{slug:string;task:string};replay:Replay;numSplats:number;samples:number;floor:number;top:number;cut:number;size:THREE.Vector3};
type Log={id:number;at:string;text:string;tone:'ok'|'warn'|'dim'|'hi'};
type RobotRun={robot:RobotId;replay:Replay;start:number;speed:number};
type UI={stage:'empty'|'loading'|'ready'|'morph';status:string;error:string;data:Data|null;phase:number;now:number;paused:boolean;logs:Log[];crops:Record<string,string>;robot:{id:RobotId;time:number;replay:Replay}|null;verdicts:Partial<Record<RobotId,{verdict:Verdict|'missing';task:string}>>;panel:boolean;running:boolean;pick:{robot:RobotId;slug:string}};

const PHASES=['SCAN','TERRA VISION','LUNA DECISIONS','SIMREADY SWAP','PHYSICS','ROBOTS'];
const START=[0,7,21,37,42,47];
const SHEET_T=7.5, VIEW_DT=.7, RINGS_T=19.2, LUNA_SLOT=5.2, SWAP_T=37.7, SWAP_DT=1.4, RANK_AT=2, DECIDE_AT=3.7;
const ROBOT_NAME:Record<RobotId,string>=Object.fromEntries(ROBOTS.map(r=>[r.id,r.name])) as Record<RobotId,string>;
const pad=(n:number)=>String(n).padStart(2,'0');
const stamp=(t:number)=>`${pad(Math.floor(t/60))}:${pad(Math.floor(t%60))}.${String(Math.floor((t%1)*1000)).padStart(3,'0')}`;
const fmt=(n:number)=>n.toLocaleString('en-US');
const pct=(p:number)=>p.toFixed(2);
const bandOf=(c:Candidate)=>[...c.probabilities].sort((a,b)=>b.probability-a.probability)[0]?.label??'';

/** Imperative 3D director: owns the viewer, effects, robot playback and the morph clock. React renders the overlays from `UI`. */
class Show {
  viewer:RoomViewer; data:Data|null=null; t=0; paused=false; morphing=false; disposed=false;
  private events:{at:number;fn:()=>void;done:boolean}[]=[]; private raf=0; private last=0; private publish=0; private logId=0;
  private home={pos:new THREE.Vector3(),target:new THREE.Vector3()}; private tween:{from:[THREE.Vector3,THREE.Vector3];to:[THREE.Vector3,THREE.Vector3];start:number;dur:number}|null=null;
  private reveal:ReturnType<typeof createSphereReveal>|null=null; private revealStart=0; private revealRadius=1; private center=new THREE.Vector3();
  private scan:ReturnType<typeof createScanPlane>|null=null; private rings:ReturnType<typeof createRing>[]=[];
  private assets=new Map<string,Promise<THREE.Group>>(); private swaps:{flash:ReturnType<typeof createFlash>;item:Replacement}[]=[];
  private saved:SavedObjects|null=null; layer:SavedObjectLayer|null=null; private layerReady=false;
  private collidersOn=false; robotScene:RobotScene|null=null; pov:RobotPov|null=null; run:RobotRun|null=null; private onRunEnd:(()=>void)|null=null;
  constructor(private host:HTMLElement,private set:(fn:(u:UI)=>UI)=>void,private labels:()=>HTMLElement|null){
    this.viewer=new RoomViewer(host,{onSelect:()=>{},onChange:()=>{},onMeasure:()=>{},onStatus:status=>set(u=>({...u,status}))},{crop:true});
    this.whiten();this.viewer.sourceRoot.visible=false;
    this.raf=requestAnimationFrame(this.frame);
  }
  private whiten(){this.viewer.scene.background=new THREE.Color('#ffffff');this.viewer.scene.children.filter(o=>o instanceof THREE.GridHelper).forEach(o=>o.visible=false)}
  log(text:string,tone:Log['tone']='dim'){const id=++this.logId,at=stamp(this.t);this.set(u=>({...u,logs:[...u.logs.slice(-160),{id,at,text,tone}]}))}
  async loadKitchen(){
    this.set(u=>({...u,stage:'loading',status:'GET /api/environments/kitchen'}));
    const [envRaw,views,detections,report,objects,tasks]=await Promise.all([getJson('/api/environments/kitchen'),fetch('/discovery/views.json').then(r=>r.json()),fetch('/discovery/detections.json').then(r=>r.json()),fetch('/discovery/report.json').then(r=>r.json()),getJson('/api/objects/scene'),fetchTasks()]);
    const env=parseEnvironment(envRaw);if(!env.visual.splatUrl)throw new Error('The kitchen has no visual scan.');
    const chosen=tasks.find(t=>t.slug.includes('bowl'))??tasks.find(t=>t.slug==='wipe-the-table')??tasks[0];if(!chosen)throw new Error('No robot tasks have been published.');
    const replay=parseTaskEnvelope(await getJson(taskUrl(chosen.slug,'stretch')),env,'stretch').replay;
    if(this.disposed)return;
    const v=this.viewer,axis=env.visual.sourceUpAxis==='Z-up'?'z-up':env.visual.sourceUpAxis==='Y-up'?'y-up':'y-down';
    this.set(u=>({...u,status:`streaming ${env.visual.splatUrl!.split('/').pop()}`}));
    await v.load(env.visual.splatUrl,env.visual.meshUrl??undefined,env.scene.name,'upload',env.visual.splatUrl.split('/').pop()!.split('?')[0],axis);
    if(this.disposed)return;
    v.applyWorldTransform(env.visual.transform.matrix);v.setCrop(true);v.setCutaway(true);v.setLayers(true,false,false);
    const b=computeBounds(v.points,.008),min=new THREE.Vector3(...b.min),max=new THREE.Vector3(...b.max);
    const reps=((objects as {replacements?:Replacement[]}).replacements??[]).map(r=>({...r,match:(report.objects as ReportObject[]).map(o=>({o,d:Math.hypot(o.position[0]-r.position[0],o.position[1]-r.position[1])})).filter(x=>x.d<.35).sort((a,b)=>a.d-b.d)[0]?.o}));
    this.data={tasks:tasks.map(t=>({slug:t.slug,task:t.task})),env,views:views.views,detections:detections.detections,source:detections.source,report:report.objects,replacements:reps,task:{slug:chosen.slug,task:chosen.task},replay,numSplats:v.splat?.numSplats??0,samples:v.points.length/3,floor:min.z,top:max.z,cut:suggestedCutawayHeight(min.z,max.z),size:max.clone().sub(min)};
    this.center.set((min.x+max.x)/2,(min.y+max.y)/2,min.z);this.revealRadius=Math.hypot(max.x-min.x,max.y-min.y,max.z-min.z)*.62;
    this.scan=createScanPlane(v.scene,min,max);
    try{this.saved=parseSavedObjects(objects,env)}catch(e){console.warn('Saved SimReady objects unavailable',e)}
    // Warm the HTTP cache so the SavedObjectLayer load in phase 04 is instant.
    for(const r of reps)if(!this.assets.has(r.assetId)){const p=loadAsset(r.assetId);p.catch(()=>{});this.assets.set(r.assetId,p)}
    // Render-in: hidden until the reveal mask is in place, then a growing sphere and a slow dolly toward the home view.
    this.reveal=createSphereReveal(v.splat!,this.center);this.revealStart=performance.now();v.sourceRoot.visible=true;
    // Frame the cropped room footprint (not exterior samples) and keep it clear of the right-hand console.
    const fb=roomFootprintBounds(v.points,.008),shift=new THREE.Vector3((fb.min[0]+fb.max[0])/2,(fb.min[1]+fb.max[1])/2,v.orbit.target.z).sub(v.orbit.target);
    this.home.target.copy(v.orbit.target).add(shift);
    this.home.pos.copy(this.home.target).add(v.camera.position.clone().sub(v.orbit.target).multiplyScalar(.82));
    const offset=this.home.pos.clone().sub(this.home.target).applyAxisAngle(new THREE.Vector3(0,0,1),-.55).multiplyScalar(1.75);offset.z*=1.25;
    this.cameraTo(this.home.pos,this.home.target,4.6,[this.home.target.clone().add(offset),this.home.target.clone().add(new THREE.Vector3(0,0,-.3))]);
    this.envelopes.set(`${chosen.slug}/stretch`,replay);this.set(u=>({...u,stage:'ready',data:this.data,status:'',pick:{robot:'stretch',slug:chosen.slug}}));
  }
  cameraTo(pos:THREE.Vector3,target:THREE.Vector3,dur:number,from?:[THREE.Vector3,THREE.Vector3]){const c=this.viewer.camera,o=this.viewer.orbit;if(from){c.position.copy(from[0]);o.target.copy(from[1])}
    this.tween={from:[c.position.clone(),o.target.clone()],to:[pos.clone(),target.clone()],start:performance.now(),dur:dur*1000}}
  private lookAt(p:number[],dist:number,dur=1.1){const h=this.home.pos.clone().sub(this.home.target).normalize(),dir=new THREE.Vector3(h.x*.55,h.y*.55,1).normalize(),target=new THREE.Vector3(...p);this.cameraTo(target.clone().add(dir.multiplyScalar(dist)),target,dur)}
  startMorph(){if(!this.data||this.morphing)return;this.morphing=true;this.t=0;this.build();this.set(u=>({...u,stage:'morph',phase:0,logs:[]}))}
  togglePause(){this.paused=!this.paused;this.set(u=>({...u,paused:this.paused}))}
  next(){if(!this.morphing)return;const n=START.find(s=>s>this.t+.01);if(n!==undefined)this.t=n}
  private at(at:number,fn:()=>void){this.events.push({at,fn,done:false})}
  private phase(i:number){this.at(START[i],()=>{this.set(u=>({...u,phase:i}));this.log(`── ${pad(i+1)} ${PHASES[i]} ${'─'.repeat(Math.max(2,26-PHASES[i].length))}`,'hi')})}
  private build(){
    const d=this.data!,v=this.viewer,env=d.env,val=env.validation as Record<string,any>;
    // 01 SCAN
    this.phase(0);
    this.at(.3,()=>this.log(`spz parsed · ${fmt(d.numSplats)} splats`,'ok'));
    this.at(1.4,()=>this.log(`companion mesh sampled · ${fmt(d.samples)} surface points`,'ok'));
    this.at(2.6,()=>this.log(`floor plane fitted · z≈${d.floor.toFixed(3)} m`,'ok'));
    this.at(3.8,()=>this.log(`room bounds ${d.size.x.toFixed(2)} × ${d.size.y.toFixed(2)} × ${d.size.z.toFixed(2)} m`));
    this.at(5,()=>this.log(`display cutaway @ ${d.cut.toFixed(2)} m · room crop on`));
    this.at(6.1,()=>this.log(`world transform applied · ${env.visual.sourceUpAxis} → Z-up`));
    // 02 TERRA VISION
    this.phase(1);
    this.at(START[1]+.15,()=>this.log(d.source,'dim'));
    this.at(START[1]+.3,()=>this.log(`${d.views.length} orbit views rendered · ${d.views[0]?.width}×${d.views[0]?.height}`));
    d.views.forEach((view,i)=>this.at(SHEET_T+i*VIEW_DT+.2,()=>{const found=d.detections.filter(x=>x.view===view.view);this.log(found.length?`gpt-6.1-terra · view ${pad(view.view)} → ${found.map(x=>`${x.label} ${pct(x.confidence)}`).join(', ')}`:`gpt-6.1-terra · view ${pad(view.view)} → no tableware`,found.length?'ok':'dim')}));
    this.at(RINGS_T,()=>{for(const o of d.report)this.rings.push(createRing(v.scene,o.position,o.result==='replaced'?GREEN:o.result==='unknown'?AMBER:0x8a8f8c));
      const n=(r:string)=>d.report.filter(o=>o.result===r).length;this.log(`${d.detections.length} detections → ${d.report.length} lifted to 3D · ${n('replaced')} replaced · ${n('weak')} weak · ${n('unknown')} unknown`,'ok')});
    // 03 LUNA DECISIONS
    this.phase(2);
    d.replacements.forEach((r,i)=>{const t0=START[2]+i*LUNA_SLOT,dec=r.decision,cat=dec?.trace[0]?.answers.find(a=>a.type==='choice');
      this.at(t0,()=>{this.lookAt(r.position,2.6);const top=[...(cat?.probabilities??[])].sort((a,b)=>b.probability-a.probability)[0];this.log(`POST /v1/decisions gpt-6-luna · category=${dec?.category??'?'}${top?` p=${pct(top.probability)}`:''}`,'ok')});
      if(!r.match)this.at(t0+1.2,()=>{let url:string|null=null;try{url=captureAround(v,new THREE.Vector3(...r.position),190)}catch(e){console.warn('live crop failed',e)}if(url)this.set(u=>({...u,crops:{...u.crops,[r.id]:url!}}));else console.warn('live crop unavailable')});
      this.at(t0+RANK_AT,()=>this.log(`candidate_comparison · ${dec?.candidates.map(c=>`${c.asset.id} ${c.score.toFixed(2)}`).join(' · ')}`));
      this.at(t0+DECIDE_AT,()=>{const win=dec?.candidates.find(c=>c.asset.id===r.assetId);this.log(`→ ${r.assetId} ${win?`score ${win.score.toFixed(2)}/3 (${bandOf(win)})`:''} · ${win?.asset.license??''} · ${r.chosenBy}`,'ok')});
    });
    // 04 SIMREADY SWAP
    this.phase(3);
    d.replacements.forEach((r,i)=>this.at(SWAP_T-.55+i*SWAP_DT,()=>this.lookAt(r.position,2,.6)));
    // SavedObjectLayer owns the SimReady meshes and their splat masks (robot-scene carries them); the demo only stages the reveal.
    this.at(START[3]+.05,()=>{if(!this.saved||!v.splat){this.log('no saved SimReady scene for this scan','warn');return}const layer=new SavedObjectLayer(v);this.layer=layer;layer.root.visible=false;
      void layer.load(this.saved).then(()=>{if(this.disposed)return;this.saved!.replacements.forEach(item=>layer.getMesh(item.id)?.scale.setScalar(.0001));layer.root.visible=true;this.layerReady=true}).catch(e=>this.log(`SimReady layer failed: ${e instanceof Error?e.message:String(e)}`,'warn'))});
    d.replacements.forEach((r,i)=>this.at(SWAP_T+i*SWAP_DT,()=>{this.swaps.push({flash:createFlash(v.scene,r.position),item:r});
      this.log(`${r.assetId} · scale ${r.scale.toFixed(3)} · masked ${r.selection.size.map(n=>n.toFixed(2)).join('×')} m`,'ok')}));
    this.at(SWAP_T+d.replacements.length*SWAP_DT+.2,()=>this.log(`${d.replacements.length} objects replaced · splats masked · pose/scale from reviewed fit`,'ok'));
    // 05 PHYSICS
    this.phase(4);
    this.at(START[4]+.05,()=>{const objects=env.scene.objects.filter(o=>(o as typeof o&{enabled?:boolean}).enabled!==false);v.setBoxes(objects);v.select(null);fadeColliders(v,0);v.setLayers(true,false,true);this.collidersOn=true;this.cameraTo(this.home.pos,this.home.target,1.2);
      const roles=objects.reduce<Record<string,number>>((a,o)=>{const k=o.role??'box';a[k]=(a[k]??0)+1;return a},{});this.log(`scene.json · ${objects.length} box colliders · ${Object.entries(roles).map(([k,n])=>`${n} ${k}`).join(' · ')}`,'ok')});
    this.at(START[4]+1.8,()=>this.log(`landmarks · ${(d.replay.landmarks??[]).filter(l=>l.kind!=='tall'&&l.kind!=='counter').map(l=>l.label).join(' · ')}`));
    this.at(START[4]+2.8,()=>this.log(`${val.engine} ${val.engineVersion} · compile ${val.compile} · ${val.compiledGeomCount} geoms`,val.compile==='passed'?'ok':'warn'));
    this.at(START[4]+3.4,()=>{const p=val.dropProbe??{};this.log(`drop probe ${p.status} · rest z ${Number(p.actualRestZ).toFixed(3)} m (expected ${p.expectedRestZ}) · floor contact ${p.floorContact?'✓':'✗'}`,p.status==='passed'?'ok':'warn')});
    this.at(START[4]+4,()=>{for(const w of (val.warnings as string[]|undefined)??[])this.log(w,'warn')});
    // 06 ROBOTS
    this.phase(5);
    this.at(START[5]+.05,()=>{this.robotScene=new RobotScene(v);this.whiten();this.robotScene.hasEnvironmentColliders=true;
      try{this.pov=new RobotPov(v,this.robotScene);this.pov.setLayout({widthFraction:.22,aspect:16/10,right:384,bottom:96})}catch(e){console.warn('Robot POV unavailable',e)}
      this.robotScene.clearReplay();this.robotScene.setCarriedObjects(this.layer);this.cameraTo(this.home.pos,this.home.target,1);
      this.log(`environment ready · ${d.tasks.length} tasks published · choose a robot and run`,'ok');this.set(u=>({...u,panel:true}))});
  }
  play(robot:RobotId,replay:Replay,onEnd:()=>void,speed=1){if(!this.robotScene)return;this.robotScene.setReplay(replay);this.run={robot,replay,start:this.t+.4,speed};this.onRunEnd=onEnd;this.firedRobotEvents=new Set();this.log(`spawn ${ROBOT_NAME[robot]} · ${replay.duration.toFixed(1)} s rollout${speed!==1?` · ${speed}× playback`:''}`,'hi')}
  private firedRobotEvents=new Set<number>();
  private envelopes=new Map<string,Replay>();
  /** Load (or reuse) the chosen robot's published rollout for a task and play it; nothing moves before this is called. */
  async runTask(slug:string,robot:RobotId){const d=this.data;if(!d||!this.robotScene||this.onRunEnd)return;const task=d.tasks.find(t=>t.slug===slug);if(!task)return;
    this.set(u=>({...u,running:true}));const key=`${slug}/${robot}`;
    let replay=this.envelopes.get(key);
    if(!replay){this.log(`GET ${taskUrl(slug,robot)}`);try{replay=parseTaskEnvelope(await getJson(taskUrl(slug,robot)),d.env,robot).replay;this.envelopes.set(key,replay)}catch(e){if(this.disposed)return;this.log(`${ROBOT_NAME[robot]} · “${task.task}” has no rollout: ${e instanceof Error?e.message:String(e)}`,'warn');this.set(u=>({...u,running:false,verdicts:{...u.verdicts,[robot]:{verdict:'missing',task:task.task}}}));return}}
    if(this.disposed)return;const ready=replay;
    this.log(`task “${task.task}” · ${ROBOT_NAME[robot]} · ${ready.frames.length} frames · planner ${ready.verdict?.planner??'n/a'}`,'ok');
    this.play(robot,ready,()=>this.set(u=>({...u,running:false,verdicts:{...u.verdicts,[robot]:{verdict:ready.verdict??'missing',task:task.task}}})));
  }
  private frame=(now:number)=>{if(this.disposed)return;this.raf=requestAnimationFrame(this.frame);
    const dt=this.last?Math.min((now-this.last)/1000,.1):0;this.last=now;const v=this.viewer;
    const el=v.renderer.domElement,W=el.clientWidth,H=el.clientHeight,shiftX=W>900?Math.min(190,W*.13):0;if(W&&(v.camera.view?.fullWidth!==W||v.camera.view?.fullHeight!==H||v.camera.view?.offsetX!==shiftX)){v.camera.setViewOffset(W,H,shiftX,0,W,H)}
    if(this.tween){const k=easeInOutCubic((now-this.tween.start)/this.tween.dur);v.camera.position.lerpVectors(this.tween.from[0],this.tween.to[0],k);v.orbit.target.lerpVectors(this.tween.from[1],this.tween.to[1],k);if(k>=1)this.tween=null}
    if(this.reveal){const k=(now-this.revealStart)/2600;this.reveal.setRadius(easeOutCubic(k)*this.revealRadius);if(k>=1){this.reveal.dispose();this.reveal=null}}
    if(this.morphing&&!this.paused)this.t+=dt;const t=this.t;
    if(this.morphing){for(const e of this.events)if(!e.done&&t>=e.at){e.done=true;e.fn()}
      const d=this.data!;
      if(this.scan){const k=progress(t,.3,6),a=Math.min(progress(t,.1,.3),1-progress(t,6.4,.5));this.scan.set(d.floor+(d.cut-d.floor)*easeInOutCubic(k),t<START[1]?a:0)}
      const ringVis=Math.min(progress(t,RINGS_T,.5),1-progress(t,START[3],.5));this.rings.forEach(r=>r.update(now/1000,ringVis));
      for(const [i,s] of this.swaps.entries()){const p=progress(t,SWAP_T+i*SWAP_DT,.9);s.flash.update(progress(t,SWAP_T+i*SWAP_DT,1.1));const mesh=this.layerReady?this.layer?.getMesh(s.item.id):null;if(mesh)mesh.scale.setScalar(p>=1?s.item.scale:Math.max(.0001,backOut(p)*s.item.scale))}
      if(this.collidersOn)fadeColliders(v,t<START[5]?easeOutCubic(progress(t,START[4]+.1,1.4)):1-.8*progress(t,START[5],1));
      if(this.run&&this.robotScene){const rt=Math.max(0,(t-this.run.start)*this.run.speed);this.robotScene.update(Math.min(rt,this.run.replay.duration));
        (this.run.replay.events??[]).forEach((e,i)=>{if(rt>=e.time&&!this.firedRobotEvents.has(i)){this.firedRobotEvents.add(i);this.log(`[${this.run!.robot}] t=${e.time.toFixed(1)}s ${e.label}`,/cannot|fail|unable|no manipulator|partial/i.test(e.label)?'warn':'ok')}});
        if(rt>=this.run.replay.duration&&this.onRunEnd){const end=this.onRunEnd,r=this.run;this.onRunEnd=null;const vd=r.replay.verdict;if(vd)this.log(`verdict ${ROBOT_NAME[r.robot]} · ${vd.status} · ${vd.steps.filter(s=>s.status==='done').length}/${vd.steps.length} steps`,vd.status==='completed'?'ok':'warn');end()}}
    }
    // HTML labels pinned to world points.
    const root=this.labels();if(root)for(const el of Array.from(root.children) as HTMLElement[]){const p=project(v,new THREE.Vector3(+el.dataset.x!,+el.dataset.y!,+el.dataset.z!));if(!p){el.style.opacity='0';continue}el.style.transform=`translate(${p.x.toFixed(1)}px,${p.y.toFixed(1)}px)`;el.style.opacity=''}
    if(this.morphing&&now-this.publish>50){this.publish=now;const run=this.run;this.set(u=>({...u,now:t,robot:run?{id:run.robot,time:Math.max(0,Math.min(run.replay.duration,(t-run.start)*run.speed)),replay:run.replay}:null}))}
  };
  dispose(){if(this.disposed)return;this.disposed=true;cancelAnimationFrame(this.raf);this.reveal?.dispose();this.scan?.dispose();this.rings.forEach(r=>r.dispose());
    for(const s of this.swaps)s.flash.dispose();this.robotScene?.setCarriedObjects(null);this.layer?.dispose();
    this.pov?.dispose();this.robotScene?.dispose();this.viewer.dispose()}
}

const INITIAL:UI={stage:'empty',status:'',error:'',data:null,phase:-1,now:0,paused:false,logs:[],crops:{},robot:null,verdicts:{},panel:false,running:false,pick:{robot:'stretch',slug:''}};
const ROOMS=[{name:'Rustic kitchen',meta:'500k gaussians',active:true,image:'/rooms/rustic-kitchen.jpg'},{name:'Hackney flat',meta:'soon',active:false,image:'/rooms/hackney-flat.jpg'},{name:'Studio',meta:'soon',active:false,image:'/rooms/studio.jpg'}];

function Glyph({id}:{id:RobotId}){const c={fill:'none',stroke:'currentColor',strokeWidth:1.4,strokeLinecap:'square' as const};
  return <svg viewBox="0 0 64 44" width="64" height="44" aria-hidden="true">{id==='go2'?<g {...c}><rect x="14" y="14" width="30" height="9"/><rect x="44" y="11" width="9" height="8"/><path d="M18 23l-3 9 3 7M26 23l2 9-2 7M36 23l-3 9 3 7M42 23l2 9-2 7"/><path d="M4 41h56" strokeDasharray="2 3"/></g>
    :id==='g1'?<g {...c}><circle cx="32" cy="7" r="4"/><path d="M26 13h12v14H26z"/><path d="M26 15l-7 10M38 15l7 10"/><path d="M28 27l-3 14M36 27l3 14"/><path d="M4 41h56" strokeDasharray="2 3"/></g>
    :<g {...c}><rect x="10" y="31" width="18" height="8"/><circle cx="13" cy="39" r="2"/><circle cx="25" cy="39" r="2"/><path d="M22 31V4"/><rect x="19" y="15" width="7" height="5"/><path d="M26 17h24M50 15v5M53 15v5"/><path d="M4 41h56" strokeDasharray="2 3"/></g>}</svg>}

function Crop({view,pixel,size,span=230}:{view:number;pixel:[number,number];size:number;span?:number}){const s=size/span;
  return <div className="d-crop" style={{width:size,height:size,backgroundImage:`url(/discovery/view${pad(view)}.jpg)`,backgroundSize:`${850*s}px ${874*s}px`,backgroundPosition:`${-(pixel[0]*s-size/2)}px ${-(pixel[1]*s-size/2)}px`}}><i/><i/><i/><i/></div>}

function Sheet({data,now}:{data:Data;now:number}){
  const active=Math.floor((now-SHEET_T)/VIEW_DT),cw=104,ch=Math.round(cw*874/850),focus=data.views[Math.max(0,Math.min(data.views.length-1,active))],fw=262,fs=fw/850,out=now>RINGS_T+.4;
  return <div className={`d-sheet ${out?'out':''}`}>
    <div className="d-grid">{data.views.map((view,i)=>{const s=cw/850,dets=data.detections.filter(d=>d.view===view.view);
      return <div key={view.view} className={`d-thumb ${i<active?'seen':''} ${i===active?'live':''} ${i>active?'wait':''}`} style={{width:cw,height:ch,animationDelay:`${i*22}ms`}}>
        <img src={`/discovery/${view.file}`} alt={`Discovery view ${view.view}`}/>{i===active&&<span className="d-scan"/>}
        {i<=active&&dets.map((d,j)=><span key={j} className={`d-box ${d.label}`} style={{left:d.x*s-11,top:d.y*s-11}}/>)}
        <small>{pad(view.view)}</small></div>})}</div>
    {focus&&active>=0&&<div className="d-focus" style={{width:fw}}><div className="d-focus-img" style={{height:Math.round(fw*874/850)}}><img src={`/discovery/${focus.file}`} alt=""/><span className="d-scan big" key={focus.view}/>
      {data.detections.filter(d=>d.view===focus.view).map((d,j)=><span key={`${focus.view}-${j}`} className={`d-bracket ${d.label}`} style={{left:d.x*fs-30,top:d.y*fs-30}}><b>{d.label} {pct(d.confidence)}</b></span>)}</div>
      <div className="d-focus-meta"><span>VIEW {pad(focus.view)}</span><span>{focus.width}×{focus.height}</span><span>{data.detections.filter(d=>d.view===focus.view).length} hits</span></div></div>}
  </div>;
}

function LunaCard({data,item,crop,local}:{data:Data;item:Replacement;crop?:string;local:number}){
  const dec=item.decision,choice=dec?.trace[0]?.answers.find(a=>a.type==='choice'),preds=dec?.trace[0]?.answers.filter(a=>a.type==='predicate')??[];
  const probs=[...(choice?.probabilities??[])].sort((a,b)=>b.probability-a.probability).slice(0,4);const ranked=[...(dec?.candidates??[])].sort((a,b)=>b.score-a.score);const showRank=local>RANK_AT,decided=local>DECIDE_AT;
  return <div className="d-luna glass">
    <div className="d-luna-head"><span>{item.id.slice(0,8)}</span><span>POST /v1/decisions · gpt-6-luna</span></div>
    <div className="d-luna-top">{item.match?<Crop view={item.match.view} pixel={item.match.pixel} size={128}/>:crop?<div className="d-crop" style={{width:128,height:128,backgroundImage:`url(${crop})`,backgroundSize:'cover'}}><i/><i/><i/><i/></div>:<div className="d-crop empty" style={{width:128,height:128}}>live crop</div>}
      <div className="d-cat"><div className="d-k">CATEGORY · {choice?.choice?.toUpperCase()}</div>
        {probs.map((p,i)=><div key={String(p.value)} className={`d-bar ${p.value===choice?.choice?'win':''}`} style={{animationDelay:`${.3+i*.25}s`}}><span>{String(p.value)}</span><em><i style={{width:`${p.probability*100}%`}}/></em><b>{pct(p.probability)}</b></div>)}
        <div className="d-preds">{preds.map(p=><span key={p.name}>{p.name} p={pct(p.probability??0)}</span>)}</div>
        <div className="d-src">{item.match?`terra · view ${pad(item.match.view)} · (${item.match.pixel.join(', ')}) · ${pct(item.match.confidence)}`:'operator selection · live splat crop'}</div></div></div>
    <div className="d-k" style={{marginTop:12}}>CANDIDATE COMPARISON · simready catalogue</div>
    {ranked.map((c,i)=>{const win=decided&&c.asset.id===item.assetId;return <div key={c.asset.id} className={`d-cand ${win?'win':''} ${showRank?'on':''}`} style={{transitionDelay:`${i*450}ms`}}>
      <img src={`/api/objects/assets/${encodeURIComponent(c.asset.id)}/preview.jpg`} alt={c.asset.name}/>
      <div><div className="d-cand-name">{c.asset.id}<small>{fmt(c.asset.triangles)} tris · {c.asset.dimensions.map(n=>n.toFixed(2)).join('×')} m</small></div>
        <div className="d-score"><em><i style={{width:showRank?`${c.score/3*100}%`:0}}/>{[1,2].map(k=><s key={k} style={{left:`${k/3*100}%`}}/>)}</em><b>{c.score.toFixed(2)}/3</b><span className={`band ${bandOf(c).toLowerCase()}`}>{bandOf(c)}</span></div>
        {win&&<div className="d-attrib">nvidia/simready-assets · {c.asset.id} · {c.asset.license}</div>}</div></div>})}
    <div className="d-luna-foot">{data.replacements.indexOf(item)+1} / {data.replacements.length}</div>
  </div>;
}

// After loading, the veil holds until the quote has finished typing plus this long to read it.
const VEIL_READ_MS=1600;
const QUOTE:[string,''|'blue'|'red'][]=[["This is your last chance. Take the ",''],["blue pill",'blue'],[" and you buy the robot the brochure promised. Take the ",''],["red pill",'red'],[" and I show you how it really moves through your kitchen.",'']];
const QUOTE_LEN=QUOTE.reduce((n,[t])=>n+t.length,0);

function Hand({pill,x,flip}:{pill:'blue'|'red';x:number;flip?:boolean}){
  return <g transform={flip?`translate(${x+200},0) scale(-1,1)`:`translate(${x},0)`}>
    <g className="rp-hand">
      <path d="M0 64H20V112H0"/>
      <path d="M20 68C36 67 48 64 58 60C66 49 72 38 81 32C88 28 94 33 90 41C86 50 81 58 80 66C96 72 118 72 136 66C150 61 165 55 177 51C186 48 191 56 183 61C174 66 163 70 153 74C165 74 177 73 186 72C194 72 195 81 186 83C176 85 165 86 155 88C164 90 173 91 180 92C187 94 185 102 177 101C150 99 118 106 90 108C62 110 40 109 20 108"/>
      <path d="M141 77C152 79 166 79 176 77M136 86C146 88 158 89 168 89" className="rp-crease"/>
      <path d="M84 70C98 78 120 78 134 72" className="rp-crease"/>
    </g>
    <g className={`rp-pill ${pill}`}><g transform="translate(108 62) rotate(-14)"><rect x="-15" y="-6.5" width="30" height="13" rx="6.5"/><path d="M0-6.5V6.5"/><rect x="-11" y="-4" width="9" height="2.4" rx="1.2" className="rp-shine"/></g></g>
  </g>;
}

function Rain({on}:{on:boolean}){
  const ref=useRef<HTMLCanvasElement>(null);
  useEffect(()=>{if(!on)return;const c=ref.current!,g=c.getContext('2d')!,cell=16,glyphs='ｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄ0123456789MORPHEUS<>/*';
    let drops:number[]=[],raf=0,last=0;
    const size=()=>{c.width=innerWidth;c.height=innerHeight;drops=Array.from({length:Math.ceil(c.width/cell)},()=>Math.random()*c.height/cell-12);g.font=`${cell-3}px 'JetBrains Mono',monospace`};
    const tick=(t:number)=>{raf=requestAnimationFrame(tick);if(t-last<55)return;last=t;
      g.fillStyle='rgba(43,46,45,.16)';g.fillRect(0,0,c.width,c.height);
      drops.forEach((y,i)=>{const ch=glyphs[Math.floor(Math.random()*glyphs.length)];g.fillStyle=Math.random()<.04?'#d8ffe9':'#19c37d';g.fillText(ch,i*cell,y*cell);
        drops[i]=y*cell>c.height&&Math.random()>.97?Math.random()*-20:y+1})};
    size();addEventListener('resize',size);raf=requestAnimationFrame(tick);
    return()=>{cancelAnimationFrame(raf);removeEventListener('resize',size)}},[on]);
  return <canvas ref={ref} className="rp-rain"/>;
}

// Grey loading veil: up as soon as loading starts, held until the quote lands, then dissolves into the room reveal.
function RedPill({loading,abort,status}:{loading:boolean;abort:boolean;status:string}){
  const [typed,setTyped]=useState(0),[on,setOn]=useState(false),[live,setLive]=useState(false);
  useEffect(()=>{if(loading){setOn(true);setLive(true)}},[loading]);
  useEffect(()=>{if(!on)return;setTyped(0);const id=setInterval(()=>setTyped(n=>n>=QUOTE_LEN?(clearInterval(id),n):n+1),17);return()=>clearInterval(id)},[on]);
  useEffect(()=>{if(abort){setOn(false);return}if(!on||loading||typed<QUOTE_LEN)return;
    const t=setTimeout(()=>setOn(false),VEIL_READ_MS);return()=>clearTimeout(t)},[on,loading,abort,typed>=QUOTE_LEN]);
  useEffect(()=>{if(on)return;const t=setTimeout(()=>setLive(false),1400);return()=>clearTimeout(t)},[on]);
  if(!live&&!on)return null;
  let left=typed;
  return <div className={`rp ${on?'on':''}`} aria-live="polite">
    <Rain on={live}/>
    <div className="rp-body">
      <svg className="rp-hands" viewBox="0 0 440 120" aria-hidden="true"><Hand pill="blue" x={0}/><Hand pill="red" x={240} flip/></svg>
      <blockquote className="rp-quote">{QUOTE.map(([text,tone],i)=>{const shown=text.slice(0,Math.max(0,left));left-=text.length;return <span key={i} className={tone}>{shown}</span>})}<span className={`rp-caret ${typed>=QUOTE_LEN?'done':''}`}/></blockquote>
      <cite className="rp-cite">— MORPHEUS</cite>
      <div className="rp-status"><span className="d-spin"/>{status||'loading'}</div>
    </div>
  </div>;
}

export default function Demo(){
  const host=useRef<HTMLDivElement>(null),labels=useRef<HTMLDivElement>(null),show=useRef<Show|null>(null),logEnd=useRef<HTMLDivElement>(null);
  const [ui,setUi]=useState<UI>(INITIAL),[gen,setGen]=useState(0),[auto,setAuto]=useState(false);
  useEffect(()=>{const s=new Show(host.current!,setUi,()=>labels.current);show.current=s;setUi(INITIAL);
    if(auto)void s.loadKitchen().catch(e=>{console.error(e);setUi(u=>({...u,stage:'empty',error:e instanceof Error?e.message:String(e)}))});
    return()=>{s.dispose();show.current=null}},[gen]);
  useEffect(()=>{logEnd.current?.scrollIntoView({block:'end'})},[ui.logs.length]);
  const start=()=>{if(ui.stage!=='empty')return;setUi(u=>({...u,error:''}));void show.current?.loadKitchen().catch(e=>{console.error(e);setUi(u=>({...u,stage:'empty',error:e instanceof Error?e.message:String(e)}))})};
  const restart=()=>{setAuto(true);setGen(g=>g+1)};
  useEffect(()=>{const key=(e:KeyboardEvent)=>{if(e.target instanceof HTMLInputElement)return;const s=show.current;if(!s)return;
      if(e.code==='Space'){e.preventDefault();if(ui.stage==='ready')s.startMorph();else if(ui.stage==='morph')s.togglePause();else if(ui.stage==='empty')start()}
      else if(e.key==='ArrowRight'){s.next()}else if(e.key.toLowerCase()==='r'&&ui.stage!=='empty'&&ui.stage!=='loading')restart()};
    addEventListener('keydown',key);return()=>removeEventListener('keydown',key)});
  const d=ui.data,now=ui.now,phase=ui.phase;
  const lunaIndex=d?Math.floor((now-START[2])/LUNA_SLOT):-1,luna=d&&phase===2&&lunaIndex>=0&&lunaIndex<d.replacements.length?d.replacements[lunaIndex]:null;
  const floor=d?.env.scene.objects.find(o=>o.role==='floor');const val=d?.env.validation as Record<string,any>|undefined;
  const scanK=easeOutCubic(progress(now,.1,5.5));
  const event=ui.robot?currentEvent(ui.robot.replay,ui.robot.time):null;
  const landmarkList=(d?.replay.landmarks??[]).filter(l=>['table','sink','door'].includes(l.kind));
  return <div className={`demo stage-${ui.stage}`}>
    <div className="d-canvas" ref={host}/>
    <div className="d-noise"/>
    <div className={`d-vignette ${ui.stage==='ready'||ui.stage==='morph'?'gone':''}`}/>
    <div className="d-labels" ref={labels}>
      {d&&phase>=1&&phase<=3&&now>=RINGS_T&&d.report.map((o,i)=><div key={`r${i}`} className={`d-pin ${o.result}`} data-x={o.position[0]} data-y={o.position[1]} data-z={o.position[2]}><span>{o.label} · v{pad(o.view)} · {pct(o.confidence)}<b>{o.result}</b></span></div>)}
      {d&&phase===4&&now>START[4]+1.8&&landmarkList.map(l=><div key={l.id} className="d-pin land" data-x={l.position[0]} data-y={l.position[1]} data-z={l.position[2]+l.size[2]/2}><span>{l.label}<b>{l.size.slice(0,2).map(n=>Math.abs(n).toFixed(2)).join('×')} m</b></span></div>)}
    </div>
    <header className="d-brand"><span className="d-mark"/>MORPHEUS{ui.stage==='ready'&&d&&<em>{d.env.scene.name} · rev {d.env.revision.slice(0,8)}</em>}</header>
    {ui.stage==='empty'&&<>
      <div className="d-rooms">{ROOMS.map(r=><button key={r.name} className={`d-room ${r.active?'':'off'}`} disabled={!r.active} onClick={start}><div className="d-tile"><img src={r.image} alt={r.name} draggable={false}/><span>{r.active?'scan ready':'scan pending'}</span></div><div className="d-room-meta"><b>{r.name}</b><small>{r.meta}</small></div></button>)}</div>
      {ui.error&&<div className="d-error">{ui.error}</div>}
      <nav className="d-links"><a href="/?view=workspace">workspace</a>·<a href="/?view=objects">objects</a>·<a href="/?view=builder">builder</a></nav>
    </>}
    <RedPill loading={ui.stage==='loading'} abort={ui.stage==='empty'} status={ui.stage==='loading'?ui.status:'wake up · entering the kitchen'}/>
    {ui.stage==='ready'&&<div className="d-cta"><button onClick={()=>show.current?.startMorph()}>MORPH TO SIM <span>→</span></button><small>scan → objects → physics → robots</small></div>}
    {ui.stage==='morph'&&d&&<>
      <nav className="d-phases">{PHASES.map((p,i)=><span key={p} className={i===phase?'on':i<phase?'past':''}><b>{pad(i+1)}</b>{p}</span>)}</nav>
      <aside className="d-console glass"><div className="d-console-head"><span>BUILD LOG</span><span className={ui.paused?'warn':'ok'}>{ui.paused?'❚❚ paused':'● live'} · {stamp(now)}</span></div>
        <div className="d-log">{ui.logs.map(l=><div key={l.id} className={`l ${l.tone}`}><time>{l.at}</time><span>{l.text}</span></div>)}<div ref={logEnd}/></div>
        <div className="d-console-foot"><span>space pause</span><span>→ next</span><span>r restart</span></div></aside>
      {phase===0&&<div className="d-counters"><div><b>{fmt(Math.round(d.numSplats*scanK))}</b><small>gaussians</small></div><div><b>{fmt(Math.round(d.samples*easeOutCubic(progress(now,1.2,2.2))))}</b><small>surface samples</small></div><div><b>{now>2.6?`z≈${d.floor.toFixed(3)}`:'—'}</b><small>floor plane fitted</small></div><div><b>{floor?`${floor.size[0].toFixed(2)}×${floor.size[1].toFixed(2)}`:'—'}</b><small>floor collider · m</small></div><div><b>{(d.cut).toFixed(2)} m</b><small>cutaway</small></div></div>}
      {phase===1&&<Sheet data={d} now={now}/>}
      {phase===2&&<div className="d-luna-wrap">{d.replacements.slice(0,Math.max(0,lunaIndex)).map(r=><div key={r.id} className="d-done-chip glass">✓ {r.decision?.category} → {r.assetId}</div>)}{luna&&<LunaCard key={luna.id} data={d} item={luna} crop={ui.crops[luna.id]} local={now-START[2]-lunaIndex*LUNA_SLOT}/>}</div>}
      {phase===3&&<div className="d-swap glass">{d.replacements.map((r,i)=><div key={r.id} className={now>=SWAP_T+i*SWAP_DT?'on':''}><img src={`/api/objects/assets/${encodeURIComponent(r.assetId)}/preview.jpg`} alt=""/><span>{r.assetId}<small>×{r.scale.toFixed(2)} · [{r.position.map(n=>n.toFixed(2)).join(', ')}]</small></span></div>)}</div>}
      {phase===4&&val&&now>START[4]+3.3&&<div className="d-stamp"><b>{val.engine} {String(val.engineVersion).split('.').slice(0,2).join('.')}</b><span>compiled {val.compile==='passed'?'✓':'✗'}</span><span>floor-contact probe {val.dropProbe?.status==='passed'?'✓':'✗'}</span><small>{val.compiledGeomCount} geoms · rest z {Number(val.dropProbe?.actualRestZ).toFixed(3)} m</small></div>}
      {phase===5&&<>
        {ui.robot&&<div className="d-task glass"><small>TASK · {ROBOT_NAME[ui.robot.id]}</small><b>{ui.robot.replay.task??''}</b><div className="d-prog"><i style={{width:`${ui.robot.time/ui.robot.replay.duration*100}%`}}/></div></div>}
        {event&&ui.running&&<div className="d-caption" key={event.label+ui.robot?.id}>{event.label}</div>}
        {Object.keys(ui.verdicts).length>0&&<div className="d-verdicts">{(['go2','g1','stretch'] as RobotId[]).filter(id=>ui.verdicts[id]).map(id=>{const r=ui.verdicts[id]!,v=r.verdict;return <div key={id} className={`d-verdict glass ${v!=='missing'?v.status:''}`}>
            <small>{ROBOT_NAME[id]} · {r.task}</small>{v!=='missing'?<><b>{v.status}</b><span>{v.steps.filter(s=>s.status==='done').length}/{v.steps.length} steps</span><ul>{v.steps.map(s=><li key={s.name} className={s.status}>{s.status==='done'?'✓':s.status==='failed'?'✗':'–'} {s.name}</li>)}</ul></>:<b>no rollout</b>}</div>})}</div>}
        {ui.panel&&<div className="d-robots glass">
          <div className="d-k">06 · SELECT ROBOT</div>
          <div className="d-robot-row">{ROBOTS.map(r=><button key={r.id} className={`d-robot ${ui.pick.robot===r.id?'sel':''}`} disabled={ui.running} onClick={()=>setUi(u=>({...u,pick:{...u.pick,robot:r.id}}))}><Glyph id={r.id}/><b>{r.name}</b><small>{r.type}</small>{ui.verdicts[r.id]&&ui.verdicts[r.id]!.verdict!=='missing'&&<i className={(ui.verdicts[r.id]!.verdict as Verdict).status}/>}</button>)}</div>
          <div className="d-k" style={{marginTop:12}}>TASK · GET /api/tasks</div>
          <div className="d-task-list">{d.tasks.map(t=><button key={t.slug} className={ui.pick.slug===t.slug?'sel':''} disabled={ui.running} onClick={()=>setUi(u=>({...u,pick:{...u.pick,slug:t.slug}}))}><span>{ui.pick.slug===t.slug?'◉':'○'}</span>{t.task}</button>)}</div>
          <div className="d-run-row"><button className="d-run" disabled={ui.running||!ui.pick.slug} onClick={()=>void show.current?.runTask(ui.pick.slug,ui.pick.robot)}>{ui.running?<>RUNNING<span className="d-spin light"/></>:<>RUN TASK <span>→</span></>}</button>
            <div className="d-run-links"><button className="ghost" onClick={restart}>↺ replay</button><a href="/?view=workspace">Open workspace →</a></div></div>
        </div>}
      </>}
    </>}
  </div>;
}
