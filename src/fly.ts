import * as THREE from 'three';

/**
 * Minecraft-creative-style free camera for a Z-up scene.
 *
 * W/A/S/D move along the horizontal look direction, Space rises, Shift sinks,
 * mouse drag (or pointer lock after a click) looks around, wheel changes speed.
 * While enabled the orbit controls must not call update(), because OrbitControls
 * rewrites the camera pose every frame; the owner of the render loop checks
 * `fly.enabled` and calls `fly.update(dt)` instead.
 */
export type FlyPose = {yaw:number;pitch:number};

/** Yaw/pitch (Z-up; yaw 0 looks along +Y) from a world look direction. */
export function poseFromDirection(d:{x:number;y:number;z:number}):FlyPose{
  const len=Math.hypot(d.x,d.y,d.z)||1;
  return {yaw:Math.atan2(-d.x,d.y),pitch:Math.asin(Math.max(-1,Math.min(1,d.z/len)))};
}

/** World look direction for a yaw/pitch pose. */
export function directionFromPose(p:FlyPose):[number,number,number]{
  const c=Math.cos(p.pitch);
  return [-Math.sin(p.yaw)*c,Math.cos(p.yaw)*c,Math.sin(p.pitch)];
}

/** Velocity (m/s) in world space for the pressed keys at a yaw. Shift/Space are vertical, so pitch is ignored. */
export function moveVector(keys:Set<string>,yaw:number,speed:number):[number,number,number]{
  const forward=[-Math.sin(yaw),Math.cos(yaw)] as const,right=[Math.cos(yaw),Math.sin(yaw)] as const;
  let x=0,y=0,z=0;
  if(keys.has('KeyW')||keys.has('ArrowUp')){x+=forward[0];y+=forward[1]}
  if(keys.has('KeyS')||keys.has('ArrowDown')){x-=forward[0];y-=forward[1]}
  if(keys.has('KeyD')||keys.has('ArrowRight')){x+=right[0];y+=right[1]}
  if(keys.has('KeyA')||keys.has('ArrowLeft')){x-=right[0];y-=right[1]}
  if(keys.has('Space')||keys.has('KeyE'))z+=1;
  if(keys.has('ShiftLeft')||keys.has('ShiftRight')||keys.has('KeyQ'))z-=1;
  const n=Math.hypot(x,y,z);
  return n?[x/n*speed,y/n*speed,z/n*speed]:[0,0,0];
}

const MAX_PITCH=Math.PI/2-0.02;

export class FlyControls {
  enabled=false; speed=1.6; lookSpeed=0.0022; pose:FlyPose={yaw:0,pitch:0};
  private keys=new Set<string>(); private dragging=false; private last={x:0,y:0}; private disposed=false;
  onChange:(enabled:boolean)=>void=()=>{};
  constructor(private camera:THREE.PerspectiveCamera,private element:HTMLElement){
    element.addEventListener('pointerdown',this.pointerDown);
    element.addEventListener('pointermove',this.pointerMove);
    window.addEventListener('pointerup',this.pointerUp);
    element.addEventListener('wheel',this.wheel,{passive:false});
    element.addEventListener('click',this.click);
    window.addEventListener('keydown',this.keyDown);
    window.addEventListener('keyup',this.keyUp);
    document.addEventListener('pointerlockchange',this.lockChange);
    window.addEventListener('blur',()=>this.keys.clear());
  }
  get locked(){return document.pointerLockElement===this.element}
  /** Enter fly mode from the camera's current pose so there is no jump. */
  enable(){
    if(this.enabled||this.disposed)return;
    const d=this.camera.getWorldDirection(new THREE.Vector3());
    this.pose=poseFromDirection(d);this.enabled=true;this.keys.clear();this.apply();this.onChange(true);
  }
  disable(){
    if(!this.enabled)return;
    this.enabled=false;this.keys.clear();this.dragging=false;
    if(this.locked)document.exitPointerLock();
    this.onChange(false);
  }
  toggle(){this.enabled?this.disable():this.enable()}
  /** Point the camera at a target without changing position. */
  lookAt(target:THREE.Vector3){const d=target.clone().sub(this.camera.position);this.pose=poseFromDirection(d);this.apply()}
  /** Camera orientation from yaw/pitch: Rz(yaw) · Rx(π/2 + pitch) looks along +Y at yaw 0 with +Z up. */
  private apply(){this.camera.rotation.set(Math.PI/2+this.pose.pitch,0,this.pose.yaw,'ZYX');this.camera.updateMatrixWorld()}
  update(dt:number){
    if(!this.enabled)return;
    const v=moveVector(this.keys,this.pose.yaw,this.speed);
    if(v[0]||v[1]||v[2])this.camera.position.add(new THREE.Vector3(v[0]*dt,v[1]*dt,v[2]*dt));
    this.apply();
  }
  private look(dx:number,dy:number){
    this.pose.yaw-=dx*this.lookSpeed;this.pose.pitch=Math.max(-MAX_PITCH,Math.min(MAX_PITCH,this.pose.pitch-dy*this.lookSpeed));
  }
  private pointerDown=(e:PointerEvent)=>{if(!this.enabled||e.button!==0||this.locked)return;this.dragging=true;this.last={x:e.clientX,y:e.clientY}};
  private pointerMove=(e:PointerEvent)=>{
    if(!this.enabled)return;
    if(this.locked){this.look(e.movementX,e.movementY);return}
    if(!this.dragging)return;this.look(e.clientX-this.last.x,e.clientY-this.last.y);this.last={x:e.clientX,y:e.clientY};
  };
  private pointerUp=()=>{this.dragging=false};
  private click=()=>{if(this.enabled&&!this.locked)this.element.requestPointerLock?.()};
  private wheel=(e:WheelEvent)=>{if(!this.enabled)return;e.preventDefault();this.speed=Math.max(.2,Math.min(12,this.speed*(e.deltaY>0?.85:1.18)))};
  private keyDown=(e:KeyboardEvent)=>{
    if(!this.enabled)return;const t=e.target as HTMLElement|null;
    if(t&&(t.tagName==='INPUT'||t.tagName==='TEXTAREA'||t.tagName==='SELECT'||t.isContentEditable))return;
    if(e.code==='Escape'){this.disable();return}
    if(['KeyW','KeyA','KeyS','KeyD','KeyQ','KeyE','Space','ShiftLeft','ShiftRight','ArrowUp','ArrowDown','ArrowLeft','ArrowRight'].includes(e.code)){this.keys.add(e.code);e.preventDefault()}
  };
  private keyUp=(e:KeyboardEvent)=>{this.keys.delete(e.code)};
  private lockChange=()=>{if(!this.locked)this.dragging=false};
  dispose(){
    this.disposed=true;this.disable();
    this.element.removeEventListener('pointerdown',this.pointerDown);this.element.removeEventListener('pointermove',this.pointerMove);
    window.removeEventListener('pointerup',this.pointerUp);this.element.removeEventListener('wheel',this.wheel);this.element.removeEventListener('click',this.click);
    window.removeEventListener('keydown',this.keyDown);window.removeEventListener('keyup',this.keyUp);document.removeEventListener('pointerlockchange',this.lockChange);
  }
}
