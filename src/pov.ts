import * as THREE from 'three';
import { SparkRenderer } from '@sparkjsdev/spark';
import { FlyControls } from './fly';
import type { RoomViewer } from './viewer';
import type { RobotId } from './replay';
import type { Vec3 } from './types';

/** Eye height above the robot body origin (docs/TASKS.md robot profiles). */
export const EYE_HEIGHT:Record<RobotId,number>={stretch:1.30,go2:.33,g1:1.22};
/** Camera offset forward of the body origin: in front of each model's head (Go2's head reaches 0.33 m ahead of its origin). */
export const EYE_FORWARD:Record<RobotId,number>={stretch:.12,go2:.36,g1:.12};
export const EYE_PITCH=12*Math.PI/180, POV_FOV=70;
export type InsetLayout={widthFraction:number;aspect:number;right:number;bottom:number};
/** Default inset layout in CSS pixels / fractions; the user can drag and resize it (workspace.tsx). */
export const POV_INSET:InsetLayout={widthFraction:.34,aspect:16/10,right:14,bottom:104};
export const INSET_MIN_FRACTION=.18, INSET_MAX_FRACTION=.7;
/** Yaw 0 faces +Y (Z-up). */
export function forward(yaw:number):Vec3{return [-Math.sin(yaw),Math.cos(yaw),0]}
export function eyePose(robot:RobotId,position:Vec3,yaw:number){
  const f=forward(yaw),h=EYE_HEIGHT[robot],d=EYE_FORWARD[robot];
  const eye:Vec3=[position[0]+f[0]*d,position[1]+f[1]*d,position[2]+h];
  const c=Math.cos(EYE_PITCH),s=Math.sin(EYE_PITCH);
  const direction:Vec3=[f[0]*c,f[1]*c,-s];
  return {eye,direction,target:[eye[0]+direction[0],eye[1]+direction[1],eye[2]+direction[2]] as Vec3};
}
/** GL-style rect (y from the bottom) of the inset inside a width×height viewport. */
export function insetRect(width:number,height:number,layout:InsetLayout=POV_INSET){
  const w=Math.round(Math.max(120,width*layout.widthFraction)),h=Math.round(w/layout.aspect);
  return {x:Math.max(0,width-w-layout.right),y:Math.min(layout.bottom,Math.max(0,height-h)),width:w,height:h};
}
/** Keep the inset fully inside the viewport and within the allowed size range. */
export function clampInset(layout:InsetLayout,width:number,height:number):InsetLayout{
  const widthFraction=Math.min(INSET_MAX_FRACTION,Math.max(INSET_MIN_FRACTION,layout.widthFraction));
  const w=Math.max(120,width*widthFraction),h=w/layout.aspect;
  const right=Math.min(Math.max(0,width-w),Math.max(0,layout.right));
  const bottom=Math.min(Math.max(0,height-h),Math.max(0,layout.bottom));
  return {...layout,widthFraction,right,bottom};
}
/** Drag the inset by a screen delta in CSS pixels (+x right, +y down). */
export function moveInset(layout:InsetLayout,dx:number,dy:number,width:number,height:number){
  return clampInset({...layout,right:layout.right-dx,bottom:layout.bottom-dy},width,height);
}
/** Resize from the top-left corner by a horizontal screen delta; the bottom-right corner stays anchored. */
export function resizeInset(layout:InsetLayout,dx:number,width:number,height:number){
  const w=Math.max(120,width*layout.widthFraction)-dx;
  const maxWidth=Math.min(width-layout.right,(height-layout.bottom)*layout.aspect);
  return clampInset({...layout,widthFraction:Math.min(maxWidth,w)/width},width,height);
}
type PoseSource={getPose():{position:THREE.Vector3;yaw:number;robot:RobotId}|null;setPovPass(active:boolean):void};

/**
 * Picture-in-picture robot point of view. Takes over RoomViewer's animation loop.
 *
 * Spark treats every renderer.render() call as a new frame and re-sorts its splats for
 * whichever camera that call used. Rendering two cameras through one SparkRenderer makes
 * the shared sort order flip between viewpoints every frame, which shows up as flicker in
 * both views. The inset therefore has its own SparkRenderer with an offscreen target, so
 * each view keeps a stable sort for its own camera; the inset is composited afterwards.
 */
export class RobotPov {
  camera=new THREE.PerspectiveCamera(POV_FOV,POV_INSET.aspect,.03,120);
  enabled=true; swapped=false; layout:InsetLayout={...POV_INSET};
  private spark:SparkRenderer;
  private quadScene=new THREE.Scene(); private quadCamera=new THREE.OrthographicCamera(-1,1,1,-1,0,1);
  private quad:THREE.Mesh; private size=new THREE.Vector2(); private disposed=false; private lastTime=0;
  /** Free-fly camera; while enabled it replaces orbit.update() in the loop. */
  fly:FlyControls;
  constructor(private viewer:RoomViewer,private source:PoseSource){
    this.camera.up.set(0,0,1);
    this.spark=new SparkRenderer({renderer:viewer.renderer,target:{width:4,height:4}});
    this.quad=new THREE.Mesh(new THREE.PlaneGeometry(2,2),new THREE.ShaderMaterial({uniforms:{map:{value:this.spark.target!.texture}},depthTest:false,depthWrite:false,
      // The offscreen target holds linear colour (three renders every non-canvas target in the working
      // colour space), so convert to the canvas output colour space while compositing.
      vertexShader:'varying vec2 vUv;void main(){vUv=uv;gl_Position=vec4(position.xy,0.,1.);}',
      fragmentShader:'uniform sampler2D map;varying vec2 vUv;void main(){gl_FragColor=vec4(texture2D(map,vUv).rgb,1.);\n#include <colorspace_fragment>\n}'}));
    this.quadScene.add(this.quad);
    this.fly=new FlyControls(viewer.camera,viewer.renderer.domElement);
    viewer.renderer.setAnimationLoop(null);
    viewer.renderer.setAnimationLoop(this.render);
  }
  setSwap(value:boolean){this.swapped=value}
  setEnabled(value:boolean){this.enabled=value}
  setLayout(layout:InsetLayout){this.layout={...layout}}
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
    const rect=insetRect(W,H,this.layout),pr=renderer.getPixelRatio();
    const full=this.swapped?this.camera:camera,inset=this.swapped?camera:this.camera;
    const fullAspect=W/H,insetAspect=rect.width/rect.height,mainAspect=camera.aspect;
    // 1. inset through its own SparkRenderer into the offscreen target (own splat sort).
    const target=this.spark.target!,tw=Math.max(2,Math.round(rect.width*pr)),th=Math.max(2,Math.round(rect.height*pr));
    if(target.width!==tw||target.height!==th)target.setSize(tw,th);
    this.setAspect(inset,insetAspect);
    this.source.setPovPass(inset===this.camera);
    renderer.setScissorTest(false);this.spark.renderTarget({scene,camera:inset});
    // 2. full-screen view through the scene's SparkRenderer.
    this.setAspect(full,fullAspect);
    this.source.setPovPass(full===this.camera);
    renderer.setRenderTarget(null);renderer.setViewport(0,0,W,H);renderer.render(scene,full);
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
    this.spark.dispose();this.quad.geometry.dispose();(this.quad.material as THREE.Material).dispose();}
}
