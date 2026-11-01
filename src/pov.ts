import * as THREE from 'three';
import { FlyControls } from './fly';
import type { RoomViewer } from './viewer';
import type { RobotId } from './replay';
import type { Vec3 } from './types';

/** Eye height above the robot body origin (docs/TASKS.md robot profiles). */
export const EYE_HEIGHT:Record<RobotId,number>={stretch:1.30,go2:.45,soarm:.35};
export const EYE_FORWARD=.12, EYE_PITCH=12*Math.PI/180, POV_FOV=70;
/** Inset layout in CSS pixels / fractions; workspace.tsx mirrors these for the HTML frame. */
export const POV_INSET={widthFraction:.34,aspect:16/10,right:14,bottom:104};
/** Yaw 0 faces +Y (Z-up). */
export function forward(yaw:number):Vec3{return [-Math.sin(yaw),Math.cos(yaw),0]}
export function eyePose(robot:RobotId,position:Vec3,yaw:number){
  const f=forward(yaw),h=EYE_HEIGHT[robot];
  const eye:Vec3=[position[0]+f[0]*EYE_FORWARD,position[1]+f[1]*EYE_FORWARD,position[2]+h];
  const c=Math.cos(EYE_PITCH),s=Math.sin(EYE_PITCH);
  const direction:Vec3=[f[0]*c,f[1]*c,-s];
  return {eye,direction,target:[eye[0]+direction[0],eye[1]+direction[1],eye[2]+direction[2]] as Vec3};
}
export function insetRect(width:number,height:number,layout=POV_INSET){
  const w=Math.round(Math.max(120,width*layout.widthFraction)),h=Math.round(w/layout.aspect);
  return {x:Math.max(0,width-w-layout.right),y:Math.min(layout.bottom,Math.max(0,height-h)),width:w,height:h};
}
type PoseSource={getPose():{position:THREE.Vector3;yaw:number;robot:RobotId}|null;setPovPass(active:boolean):void};

/**
 * Picture-in-picture robot point of view. Takes over RoomViewer's animation loop.
 * The inset is rendered first into a render target, then the full-screen view last so
 * Spark keeps sorting splats for the on-screen camera; the inset is then composited.
 */
export class RobotPov {
  camera=new THREE.PerspectiveCamera(POV_FOV,POV_INSET.aspect,.03,120);
  enabled=true; swapped=false;
  private target=new THREE.WebGLRenderTarget(4,4,{depthBuffer:true});
  private quadScene=new THREE.Scene(); private quadCamera=new THREE.OrthographicCamera(-1,1,1,-1,0,1);
  private quad:THREE.Mesh; private size=new THREE.Vector2(); private disposed=false; private frame=0; private lastTime=0;
  /** Free-fly camera; while enabled it replaces orbit.update() in the loop. */
  fly:FlyControls;
  constructor(private viewer:RoomViewer,private source:PoseSource){
    this.camera.up.set(0,0,1);
    this.quad=new THREE.Mesh(new THREE.PlaneGeometry(2,2),new THREE.ShaderMaterial({uniforms:{map:{value:this.target.texture}},depthTest:false,depthWrite:false,
      // Raw texel copy: Spark writes display-ready colour, so no colour-space conversion here.
      vertexShader:'varying vec2 vUv;void main(){vUv=uv;gl_Position=vec4(position.xy,0.,1.);}',
      fragmentShader:'uniform sampler2D map;varying vec2 vUv;void main(){gl_FragColor=vec4(texture2D(map,vUv).rgb,1.);}'}));
    this.quadScene.add(this.quad);
    this.fly=new FlyControls(viewer.camera,viewer.renderer.domElement);
    viewer.renderer.setAnimationLoop(null);
    viewer.renderer.setAnimationLoop(this.render);
  }
  setSwap(value:boolean){this.swapped=value}
  setEnabled(value:boolean){this.enabled=value}
  /** Returns false when there is no robot pose to look from. */
  updateCamera(){const pose=this.source.getPose();if(!pose)return false;
    const p=eyePose(pose.robot,pose.position.toArray() as Vec3,pose.yaw);
    this.camera.position.set(...p.eye);this.camera.lookAt(...p.target);this.camera.updateMatrixWorld();return true;}
  private render=(now:number=performance.now())=>{
    if(this.disposed)return;const {renderer,scene,camera,orbit}=this.viewer;
    const dt=this.lastTime?Math.min((now-this.lastTime)/1000,.1):0;this.lastTime=now;
    if(this.fly.enabled){orbit.enabled=false;this.fly.update(dt)}else{orbit.enabled=true;orbit.update()}renderer.getSize(this.size);const W=this.size.x,H=this.size.y;
    const active=this.enabled&&W>0&&H>0&&this.updateCamera();
    if(!active){renderer.setScissorTest(false);renderer.setViewport(0,0,W,H);renderer.render(scene,camera);return;}
    const rect=insetRect(W,H),pr=renderer.getPixelRatio();
    const full=this.swapped?this.camera:camera,inset=this.swapped?camera:this.camera;
    const fullAspect=W/H,insetAspect=rect.width/rect.height,mainAspect=camera.aspect;
    this.frame++;
    // 1. inset into a render target (accepting the on-screen camera's splat sort).
    const tw=Math.max(2,Math.round(rect.width*pr)),th=Math.max(2,Math.round(rect.height*pr));
    if(this.target.width!==tw||this.target.height!==th)this.target.setSize(tw,th);
    this.setAspect(inset,insetAspect);
    this.source.setPovPass(inset===this.camera);
    renderer.setRenderTarget(this.target);renderer.setScissorTest(false);renderer.clear();renderer.render(scene,inset);renderer.setRenderTarget(null);
    // 2. full-screen view last so its splat sort wins.
    this.setAspect(full,fullAspect);
    this.source.setPovPass(full===this.camera);
    renderer.setViewport(0,0,W,H);renderer.render(scene,full);
    this.source.setPovPass(false);
    // 3. composite the inset.
    const autoClear=renderer.autoClear;renderer.autoClear=false;
    renderer.setScissorTest(true);renderer.setViewport(rect.x,rect.y,rect.width,rect.height);renderer.setScissor(rect.x,rect.y,rect.width,rect.height);
    renderer.render(this.quadScene,this.quadCamera);
    renderer.setScissorTest(false);renderer.setViewport(0,0,W,H);renderer.autoClear=autoClear;
    if(camera.aspect!==mainAspect)this.setAspect(camera,mainAspect);
  };
  private setAspect(cam:THREE.PerspectiveCamera,aspect:number){if(Math.abs(cam.aspect-aspect)>1e-6){cam.aspect=aspect;cam.updateProjectionMatrix()}}
  dispose(){if(this.disposed)return;this.disposed=true;const v=this.viewer;this.fly.dispose();v.orbit.enabled=true;
    try{v.renderer.setAnimationLoop(()=>{v.orbit.update();v.renderer.render(v.scene,v.camera)})}catch{/* viewer already disposed */}
    this.target.dispose();this.quad.geometry.dispose();(this.quad.material as THREE.Material).dispose();}
}
