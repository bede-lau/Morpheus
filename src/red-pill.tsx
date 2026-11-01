import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import './red-pill.css';

// Opening screen: Morpheus' shades, two 3D hands offering the pills. Only the red pill lets you in.
// Hands: @webxr-input-profiles/assets generic-hand (MIT), bundled under public/brand/hands.

type Pill='red'|'blue';
type Phase='intro'|'blue'|'exit';
const QUOTE:[string,''|Pill][]=[["This is your last chance. Take the ",''],["blue pill",'blue'],[" and you buy the robot the brochure promised. Take the ",''],["red pill",'red'],[" and I show you how it really moves through your kitchen.",'']];
const QUOTE_LEN=QUOTE.reduce((n,[t])=>n+t.length,0);
const LINES:Record<Phase,string>={intro:'choose',blue:'the story ends. you wake up in your bed and buy whatever the brochure says.',exit:'remember: all I am offering is the truth. nothing more.'};
const PILL_COLOR:Record<Pill,string>={red:'#ff2d3a',blue:'#2f6bff'};

function Rain({on}:{on:boolean}){
  const ref=useRef<HTMLCanvasElement>(null);
  useEffect(()=>{if(!on)return;const c=ref.current!,g=c.getContext('2d')!,cell=16,glyphs='ｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄ0123456789MORPHEUS<>/*';
    let drops:number[]=[],raf=0,last=0;
    const size=()=>{c.width=innerWidth;c.height=innerHeight;drops=Array.from({length:Math.ceil(c.width/cell)},()=>Math.random()*c.height/cell-12);g.font=`${cell-3}px 'JetBrains Mono',monospace`};
    const tick=(t:number)=>{raf=requestAnimationFrame(tick);if(t-last<55)return;last=t;
      g.fillStyle='rgba(2,4,3,.18)';g.fillRect(0,0,c.width,c.height);
      drops.forEach((y,i)=>{const ch=glyphs[Math.floor(Math.random()*glyphs.length)];g.fillStyle=Math.random()<.04?'#d8ffe9':'#19c37d';g.fillText(ch,i*cell,y*cell);
        drops[i]=y*cell>c.height&&Math.random()>.97?Math.random()*-20:y+1})};
    size();addEventListener('resize',size);raf=requestAnimationFrame(tick);
    return()=>{cancelAnimationFrame(raf);removeEventListener('resize',size)}},[on]);
  return <canvas ref={ref} className="rp-rain"/>;
}

type Stage3D={pick:(p:Pill)=>void;dispose:()=>void};

// Orient a relaxed XR hand so its palm faces up toward the camera and its fingers reach out toward the viewer.
function placeHand(root:THREE.Object3D,side:-1|1){
  const bone=(n:string)=>root.getObjectByName(n)!.position.clone();
  const wrist=bone('wrist'),tip=bone('middle-finger-tip'),prox=bone('middle-finger-phalanx-proximal'),mid=bone('middle-finger-phalanx-intermediate');
  const index=bone('index-finger-phalanx-proximal'),pinky=bone('pinky-finger-phalanx-proximal');
  const along=tip.clone().sub(wrist).normalize(),across=index.clone().sub(pinky).normalize();
  let normal=new THREE.Vector3().crossVectors(along,across).normalize();
  // The relaxed rest pose bends the fingertips off the proximal line on the back-of-hand side, so the palm faces the other way.
  const d=mid.clone().sub(prox).normalize(),dev=tip.clone().sub(mid);dev.sub(d.clone().multiplyScalar(dev.dot(d)));
  if(normal.dot(dev)>0)normal.negate();
  across.crossVectors(normal,along).normalize();
  const local=new THREE.Matrix4().makeBasis(along,normal,across);
  const wAlong=new THREE.Vector3(side*.06,-.12,1).normalize();
  const wNormal=new THREE.Vector3(-side*.22,1,.5);wNormal.sub(wAlong.clone().multiplyScalar(wNormal.dot(wAlong))).normalize();
  const wAcross=new THREE.Vector3().crossVectors(wNormal,wAlong);
  const world=new THREE.Matrix4().makeBasis(wAlong,wNormal,wAcross);
  const q=new THREE.Quaternion().setFromRotationMatrix(world.multiply(local.transpose()));
  const palm=wrist.clone().add(index).add(pinky).add(prox).multiplyScalar(.25);
  return {q,palm,normal:wNormal};
}

function buildStage(host:HTMLDivElement,hooks:{hover:(p:Pill|null)=>void;click:(p:Pill)=>void;ready:()=>void}):Stage3D{
  const renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});
  renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.05;renderer.outputColorSpace=THREE.SRGBColorSpace;
  host.appendChild(renderer.domElement);
  const scene=new THREE.Scene(),pmrem=new THREE.PMREMGenerator(renderer);
  scene.environment=pmrem.fromScene(new RoomEnvironment(),.04).texture;scene.environmentIntensity=.35;
  const camera=new THREE.PerspectiveCamera(28,1,.01,10),home=new THREE.Vector3(0,.3,.7),look=new THREE.Vector3(0,.07,0);
  camera.position.copy(home);camera.lookAt(look);
  scene.add(new THREE.HemisphereLight(0xcfe9dc,0x020403,.5));
  const key=new THREE.DirectionalLight(0xffffff,2.2);key.position.set(.2,.6,.5);scene.add(key);
  const rimL=new THREE.DirectionalLight(0x19c37d,3.2);rimL.position.set(-.6,.25,-.5);scene.add(rimL);
  const rimR=new THREE.DirectionalLight(0x19c37d,3.2);rimR.position.set(.6,.25,-.5);scene.add(rimR);
  const skin=new THREE.MeshPhysicalMaterial({color:0x4a3328,roughness:.56,metalness:0,clearcoat:.18,clearcoatRoughness:.6});
  const rig=new THREE.Group();scene.add(rig);
  const pills:{pill:Pill;mesh:THREE.Mesh;hit:THREE.Mesh;glow:THREE.PointLight;base:THREE.Vector3;mat:THREE.MeshPhysicalMaterial}[]=[];
  const hit:THREE.Object3D[]=[];
  const loader=new GLTFLoader();let disposed=false;
  Promise.all((['left','right'] as const).map(h=>loader.loadAsync(`/brand/hands/${h}.glb`))).then(([left,right])=>{
    if(disposed)return;
    ([[left,'blue',-1],[right,'red',1]] as const).forEach(([gltf,pill,side])=>{
      const root=gltf.scene;root.traverse(o=>{if((o as THREE.Mesh).isMesh){const m=o as THREE.Mesh;m.material=skin;m.frustumCulled=false}});
      ['xr_standard_trigger_pressed_max','xr_standard_trigger_pressed_min','xr_standard_trigger_pressed_value'].forEach(n=>{const o=root.getObjectByName(n);o?.parent?.remove(o)});
      const {q,palm,normal}=placeHand(root,side);
      const hand=new THREE.Group();hand.add(root);root.position.copy(palm).negate();
      const pivot=new THREE.Group();pivot.add(hand);hand.quaternion.copy(q);pivot.position.set(side*.108,0,0);rig.add(pivot);
      const mat=new THREE.MeshPhysicalMaterial({color:PILL_COLOR[pill],roughness:.16,clearcoat:1,clearcoatRoughness:.06,emissive:PILL_COLOR[pill],emissiveIntensity:.35});
      const mesh=new THREE.Mesh(new THREE.CapsuleGeometry(.0078,.019,8,24),mat);
      const base=pivot.position.clone().add(normal.clone().multiplyScalar(.019));mesh.position.copy(base);mesh.rotation.set(Math.PI/2,0,side*.5);
      const target=new THREE.Mesh(new THREE.SphereGeometry(.036,12,8),new THREE.MeshBasicMaterial({visible:false}));target.position.copy(base);target.userData.pill=pill;
      const glow=new THREE.PointLight(PILL_COLOR[pill],.06,.14,1.6);glow.position.copy(base).add(normal.clone().multiplyScalar(.01));
      rig.add(mesh,target,glow);hit.push(target);pills.push({pill,mesh,hit:target,glow,base,mat});
    });
    hooks.ready();
  }).catch(e=>{console.error('Hand models unavailable',e);hooks.ready()});

  const ray=new THREE.Raycaster(),ptr=new THREE.Vector2(9,9),par=new THREE.Vector2();let hovered:Pill|null=null,chosen:{pill:Pill;at:number}|null=null;
  const toNdc=(e:PointerEvent)=>{const r=renderer.domElement.getBoundingClientRect();ptr.set((e.clientX-r.left)/r.width*2-1,-(e.clientY-r.top)/r.height*2+1);par.copy(ptr)};
  const pickAt=()=>{ray.setFromCamera(ptr,camera);return (ray.intersectObjects(hit)[0]?.object.userData.pill as Pill|undefined)??null};
  const move=(e:PointerEvent)=>{toNdc(e);const p=chosen?null:pickAt();if(p!==hovered){hovered=p;hooks.hover(p);renderer.domElement.style.cursor=p?'pointer':''}};
  const down=(e:PointerEvent)=>{toNdc(e);const p=pickAt();if(p&&!chosen)hooks.click(p)};
  renderer.domElement.addEventListener('pointermove',move);renderer.domElement.addEventListener('pointerdown',down);
  const resize=()=>{const w=host.clientWidth,h=host.clientHeight;renderer.setSize(w,h);camera.aspect=w/h;
    // Keep both hands in frame on narrow screens.
    camera.position.copy(home).multiplyScalar(Math.max(1,1.05/camera.aspect));camera.updateProjectionMatrix()};
  resize();addEventListener('resize',resize);
  const clock=new THREE.Clock();let raf=0;
  const frame=()=>{raf=requestAnimationFrame(frame);const t=clock.getElapsedTime();
    rig.position.y=Math.sin(t*1.1)*.0025;rig.rotation.y+=((chosen?0:par.x*.025)-rig.rotation.y)*.05;rig.rotation.x+=((chosen?0:-par.y*.012)-rig.rotation.x)*.05;
    for(const p of pills){const on=hovered===p.pill||chosen?.pill===p.pill,gone=chosen?.pill==='blue'&&p.pill==='blue';
      const s=gone?Math.max(0,1-(t-chosen!.at)*2.2):on?1.22:1;p.mesh.scale.lerp(new THREE.Vector3(s,s,s),.15);
      p.mesh.position.y=p.base.y+Math.sin(t*2+(p.pill==='red'?1.6:0))*.0015+(on?.004:0);p.mesh.rotation.y=t*.6;
      p.mat.emissiveIntensity+=((on?1.2:.35)-p.mat.emissiveIntensity)*.12;p.glow.intensity+=((gone?0:on?.22:.06)-p.glow.intensity)*.12}
    if(chosen?.pill==='red'){const k=Math.min(1,(t-chosen.at)/1.6),e=k*k*(3-2*k),red=pills.find(p=>p.pill==='red');
      if(red){const to=red.mesh.getWorldPosition(new THREE.Vector3());camera.position.lerpVectors(home,to.clone().add(new THREE.Vector3(0,.035,.06)),e);camera.lookAt(look.clone().lerp(to,e))}}
    renderer.render(scene,camera)};
  frame();
  return {
    pick(p){chosen={pill:p,at:clock.getElapsedTime()};hovered=null;renderer.domElement.style.cursor='';if(p==='blue')setTimeout(()=>{if(chosen?.pill==='blue')chosen=null},2600)},
    dispose(){disposed=true;cancelAnimationFrame(raf);removeEventListener('resize',resize);renderer.domElement.removeEventListener('pointermove',move);renderer.domElement.removeEventListener('pointerdown',down);
      scene.traverse(o=>{const m=o as THREE.Mesh;if(m.geometry)m.geometry.dispose()});skin.dispose();pills.forEach(p=>p.mat.dispose());scene.environment?.dispose();pmrem.dispose();renderer.dispose();renderer.domElement.remove()}
  };
}

export default function RedPill(){
  const host=useRef<HTMLDivElement>(null),stage=useRef<Stage3D|null>(null);
  const [typed,setTyped]=useState(0),[phase,setPhase]=useState<Phase>('intro'),[hover,setHover]=useState<Pill|null>(null),[ready,setReady]=useState(false),[on,setOn]=useState(true),[live,setLive]=useState(true);
  const choose=(p:Pill)=>{if(phase==='exit')return;stage.current?.pick(p);setHover(null);
    if(p==='red'){setPhase('exit');setTimeout(()=>setOn(false),1500)}else{setPhase('blue');setTimeout(()=>setPhase(f=>f==='blue'?'intro':f),2600)}};
  const chooseRef=useRef(choose);chooseRef.current=choose;
  useEffect(()=>{const s=buildStage(host.current!,{hover:setHover,click:p=>chooseRef.current(p),ready:()=>setReady(true)});stage.current=s;return()=>{s.dispose();stage.current=null}},[]);
  useEffect(()=>{const id=setInterval(()=>setTyped(n=>n>=QUOTE_LEN?(clearInterval(id),n):n+1),17);return()=>clearInterval(id)},[]);
  useEffect(()=>{if(on)return;const t=setTimeout(()=>setLive(false),1200);return()=>clearTimeout(t)},[on]);
  if(!live)return null;
  let left=typed;
  const line=hover&&phase==='intro'?`take the ${hover} pill`:LINES[phase];
  return <div className={`rp ${on?'on':''} ${ready?'ready':''} phase-${phase}`}>
    <Rain on={live}/>
    <div className="rp-stage" ref={host}/>
    <div className="rp-top">
      <div className="rp-shades" role="img" aria-label="Morpheus"/>
      <blockquote className="rp-quote">{QUOTE.map(([text,tone],i)=>{const shown=text.slice(0,Math.max(0,left));left-=text.length;return <span key={i} className={tone}>{shown}</span>})}<span className={`rp-caret ${typed>=QUOTE_LEN?'done':''}`}/></blockquote>
      <cite className="rp-cite">— MORPHEUS</cite>
    </div>
    <div className={`rp-line ${hover??''}`}>{line}</div>
    <div className="rp-sr"><button onClick={()=>choose('blue')}>Take the blue pill</button><button onClick={()=>choose('red')}>Take the red pill and enter the demo</button></div>
  </div>;
}
