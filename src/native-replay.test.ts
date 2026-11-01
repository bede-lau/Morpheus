import test from 'node:test';
import assert from 'node:assert/strict';
import {digest,validateNativeArtifacts,NativeReplayView} from './native-replay';
const hash='a'.repeat(64),policy='cf668f75b90d1abf73d2b87612a6e76bccc61ff7e083b63582d3f6aaa3c1759d';
async function fixture({trained=false,navigationSha='d'.repeat(64)}={}){
 const hashes={scene_json_sha256:hash,mjcf_sha256:hash,model_mjz_sha256:hash,policy_sha256:policy,evaluator_version:'fixture',...(trained?{navigation_policy_sha256:navigationSha}:{})};
 const replay=JSON.stringify({schema_version:'1.0',authority:'python-mujoco',hashes,robot:{id:'unitree_g1_12dof',joint_names:[],body_names:['robot/pelvis'],nq:19},frame:{units:'m',up_axis:'Z',quaternion_order:'wxyz'},route:{points_m:[[0,0,0],[1,0,0]],target_id:'fixture'},samples:[{time_s:0,qpos:Array(19).fill(0),body_poses:{'robot/pelvis':[0,0,.8,1,0,0,0]}},{time_s:1,qpos:Array(19).fill(0),body_poses:{'robot/pelvis':[1,0,.8,1,0,0,0]}}],events:[]});
 const report=JSON.stringify({execution:'completed',readiness:'PREVIEW_ONLY',hardware_readiness:'UNVERIFIED',hashes,...(trained?{navigation_policy:{trained:true}}:{navigation_policy:{trained:false}})});
 const manifest=JSON.stringify({schema_version:'1.0',hardware_readiness:'UNVERIFIED',model_mjz_sha256:hash,replay_sha256:await digest(replay),report_sha256:await digest(report)});
 return {replay,report,manifest};
}
test('native adapter binds byte integrity to scene and controller identity',async()=>{
 const f=await fixture();await validateNativeArtifacts(f.replay,f.manifest,f.report,hash);
 await assert.rejects(validateNativeArtifacts(f.replay+' ',f.manifest,f.report,hash),/integrity/);
 await assert.rejects(validateNativeArtifacts(f.replay,f.manifest,f.report,'b'.repeat(64)),/scene/);
 await assert.rejects(validateNativeArtifacts(f.replay,f.manifest,f.report+' ',hash),/integrity/);
});
test('diagnostic native view takes exact earlier sample without invented interpolation',async()=>{
 const f=await fixture(),bundle=await validateNativeArtifacts(f.replay,f.manifest,f.report,hash),view=new NativeReplayView(bundle);
 assert.deepEqual(view.update(.5).toArray(),[0,0,.8]);assert.deepEqual(view.update(1).toArray(),[1,0,.8]);view.dispose();
});
test('learned native evidence requires the loaded high-level PPO digest',async()=>{
 const f=await fixture({trained:true});
 await assert.rejects(validateNativeArtifacts(f.replay,f.manifest,f.report,hash),/matching loaded PPO/);
 await validateNativeArtifacts(f.replay,f.manifest,f.report,hash,'d'.repeat(64));
 await assert.rejects(validateNativeArtifacts(f.replay,f.manifest,f.report,hash,'e'.repeat(64)),/matching loaded PPO/);
});
