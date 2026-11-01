import * as THREE from 'three';
import { createReplayBridge, validateViewerEvidence, type PolicyReplayBundle } from '../web/policy-bridge';
import type { PolicyRun } from './replay';

const G1_POLICY_SHA='cf668f75b90d1abf73d2b87612a6e76bccc61ff7e083b63582d3f6aaa3c1759d';
export async function digest(text:string){return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(text))),b=>b.toString(16).padStart(2,'0')).join('');}
/** Checks actual file bytes, source scene and pinned controller before displaying exported poses. */
export async function validateNativeArtifacts(replayText:string,manifestText:string,reportText:string,sceneSha:string,expectedHighLevelSha?:string):Promise<PolicyReplayBundle>{
 const manifest=JSON.parse(manifestText),report=JSON.parse(reportText),raw=JSON.parse(replayText);
 if(manifest.schema_version!=='1.0'||manifest.hardware_readiness!=='UNVERIFIED'||await digest(replayText)!==manifest.replay_sha256||await digest(reportText)!==manifest.report_sha256)throw new Error('Native replay/report integrity check failed.');
 validateViewerEvidence(report);
 if(report.hashes?.scene_json_sha256!==sceneSha||report.hashes?.model_mjz_sha256!==manifest.model_mjz_sha256||report.hashes?.policy_sha256!==G1_POLICY_SHA)throw new Error('Native evidence does not match this scene or the pinned G1 controller.');
 const trainedNavigation=report.navigation_policy?.trained;
 if(trainedNavigation!==undefined&&typeof trainedNavigation!=='boolean')throw new Error('Native navigation policy status is invalid.');
 if(trainedNavigation){
  if(!expectedHighLevelSha||report.hashes?.navigation_policy_sha256!==expectedHighLevelSha||raw.hashes?.navigation_policy_sha256!==expectedHighLevelSha)throw new Error('Native learned-navigation evidence requires the matching loaded PPO checkpoint.');
 }
 const bundle=createReplayBridge(raw,{scene_json_sha256:sceneSha,mjcf_sha256:report.hashes.mjcf_sha256,model_mjz_sha256:manifest.model_mjz_sha256,policy_sha256:G1_POLICY_SHA,robot_id:'unitree_g1_12dof',nq:19}).bundle;
 if(bundle.samples.length<2||bundle.samples.length>100000||bundle.samples[0].time_s!==0||bundle.samples.some(s=>!s.body_poses?.['robot/pelvis']))throw new Error('Native replay requires bounded timed body poses.');
 for(const sample of bundle.samples)for(const pose of Object.values(sample.body_poses!)){if(Math.abs(Math.hypot(...pose.slice(3))-1)>.01)throw new Error('Native body quaternion is invalid.');}
 return bundle;
}
export async function loadNativeReplay(run:PolicyRun):Promise<PolicyReplayBundle>{
 const urls=[run.nativeReplayUrl,run.nativeManifestUrl,run.validation?.reportUrl];
 if(run.robot!=='g1'||urls.some(url=>!url||!/^\/policies\/g1-native(?:-baseline)?\//.test(url)||url.includes('..')))throw new Error('Native playback requires this workspace’s exported G1 evidence.');
 const texts=await Promise.all(urls.map(async url=>{const response=await fetch(url!);if(!response.ok)throw new Error('Native evidence is unavailable. Export the G1 native run first.');const text=await response.text();if(text.length>30*1024*1024)throw new Error('Native evidence exceeds the preview limit.');return text;}));
 return validateNativeArtifacts(texts[0],texts[1],texts[2],run.scene.sha256,run.training?.checkpointSha256);
}
const CHAINS=[['pelvis','left_hip_pitch_link','left_hip_roll_link','left_hip_yaw_link','left_knee_link','left_ankle_pitch_link','left_ankle_roll_link'],['pelvis','right_hip_pitch_link','right_hip_roll_link','right_hip_yaw_link','right_knee_link','right_ankle_pitch_link','right_ankle_roll_link']];
/** Diagnostic geometry only; positions are exact recorded samples, never a new simulated rollout. */
export class NativeReplayView{
 root=new THREE.Group();private joints=new Map<string,THREE.Mesh>();private links:{from:string;to:string;mesh:THREE.Mesh}[]=[];
 private torso:THREE.Mesh;private material=new THREE.MeshStandardMaterial({color:0x96b6ae,roughness:.38,metalness:.45});
 constructor(readonly bundle:PolicyReplayBundle){
  for(const name of new Set(CHAINS.flat())){const joint=new THREE.Mesh(new THREE.SphereGeometry(name==='pelvis'?.065:.028,16,12),this.material);this.joints.set('robot/'+name,joint);this.root.add(joint);}
  for(const chain of CHAINS)for(let i=1;i<chain.length;i++){const mesh=new THREE.Mesh(new THREE.CylinderGeometry(.026,.033,1,12),this.material);this.links.push({from:'robot/'+chain[i-1],to:'robot/'+chain[i],mesh});this.root.add(mesh);}
  this.torso=new THREE.Mesh(new THREE.BoxGeometry(.16,.27,.34),new THREE.MeshStandardMaterial({color:0x3f6156,roughness:.5,transparent:true,opacity:.7}));this.root.add(this.torso);
  const points=bundle.samples.map(s=>new THREE.Vector3(s.body_poses!['robot/pelvis'][0],s.body_poses!['robot/pelvis'][1],.045));const path=new THREE.Line(new THREE.BufferGeometry().setFromPoints(points),new THREE.LineBasicMaterial({color:0x05654f,depthTest:false}));path.renderOrder=25;this.root.add(path);this.update(0);
 }
 update(time:number){
  let low=0,high=this.bundle.samples.length-1;while(low<high){const mid=Math.ceil((low+high)/2);if(this.bundle.samples[mid].time_s<=time)low=mid;else high=mid-1;}
  const poses=this.bundle.samples[low].body_poses!;
  for(const[name,node]of this.joints){const p=poses[name];node.visible=!!p;if(p)node.position.set(p[0],p[1],p[2]);}
  for(const link of this.links){const a=poses[link.from],b=poses[link.to];link.mesh.visible=!!a&&!!b;if(!a||!b)continue;const start=new THREE.Vector3(a[0],a[1],a[2]),end=new THREE.Vector3(b[0],b[1],b[2]),delta=end.clone().sub(start);link.mesh.position.copy(start).add(end).multiplyScalar(.5);link.mesh.scale.y=delta.length();if(delta.length()>1e-8)link.mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),delta.normalize());}
  const p=poses['robot/pelvis'];this.torso.quaternion.set(p[4],p[5],p[6],p[3]);this.torso.position.set(p[0],p[1],p[2]).add(new THREE.Vector3(0,0,.19).applyQuaternion(this.torso.quaternion));
  return new THREE.Vector3(p[0],p[1],p[2]);
 }
 dispose(){this.root.traverse(o=>{if(o instanceof THREE.Mesh||o instanceof THREE.Line){o.geometry.dispose();if(!Array.isArray(o.material))o.material.dispose();}});this.root.clear();}
}
