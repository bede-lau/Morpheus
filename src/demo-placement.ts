import * as THREE from 'three';
import { createDemoReplay, type Replay, type RobotId } from './replay';

// Model origin to the lowest physical part. The floor ring is decoration.
const CONTACT_Z: Record<RobotId,number> = {stretch:.01,go2:.005,soarm:.025};
const FOOTPRINT: Record<RobotId,[number,number]> = {stretch:[.295,.265],go2:[.23,.4],soarm:[.16,.16]};

/** Ground only our sample animation. Never rewrite an imported trajectory. */
export function groundDemoReplay(replay:Replay, mesh:THREE.Object3D):Replay {
  if(replay.source!=='demo')return replay;
  mesh.updateWorldMatrix(true,true);
  const ray=new THREE.Raycaster();
  const [width,length]=FOOTPRINT[replay.robot];
  return {...replay,frames:replay.frames.map(frame=>{
    const heights:number[]=[];
    // Probe below the base, not from above counters or ceilings.
    for(const [dx,dy] of [[0,0],[-width,0],[width,0],[0,-length],[0,length],[-width,-length],[-width,length],[width,-length],[width,length]]){
      const x=frame.position[0]+dx*Math.cos(frame.yaw)-dy*Math.sin(frame.yaw);
      const y=frame.position[1]+dx*Math.sin(frame.yaw)+dy*Math.cos(frame.yaw);
      ray.set(new THREE.Vector3(x,y,.25),new THREE.Vector3(0,0,-1));ray.far=1;
      const hit=ray.intersectObject(mesh,true)[0];
      if(hit)heights.push(hit.point.z);
    }
    if(heights.length!==9)throw new Error('The demo route has no complete floor support.');
    return {...frame,position:[frame.position[0],frame.position[1],Math.max(...heights)-CONTACT_Z[replay.robot]+.03]};
  })};
}
export function createPlacedDemoReplay(robot:RobotId,mesh?:THREE.Object3D|null):Replay {
  const replay=createDemoReplay(robot);
  return mesh?groundDemoReplay(replay,mesh):replay;
}
