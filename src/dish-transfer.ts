import * as T from 'three';
import {sampleReplay, type Replay, type Transfer} from './replay';
import {Cleaning, trainingBasin} from './cleaning';
import type {ArticulatedRobot} from './robot-model';
import type {SavedObjectLayer} from './saved-objects';

const vec=(x=0,y=0,z=0)=>new T.Vector3(x,y,z);
const smooth=(x:number)=>T.MathUtils.smoothstep(x,0,1);

function enamel(geometry:T.BufferGeometry,color:number){
  return new T.Mesh(geometry,new T.MeshStandardMaterial({color,roughness:.36,side:T.DoubleSide}));
}

export class DishTransfer {
  readonly root=new T.Group();
  readonly vessel=new T.Group();
  readonly grip=new T.Group();
  readonly water=enamel(new T.CylinderGeometry(.046,.043,1,32),0x3aaee0);
  readonly stream=enamel(new T.CylinderGeometry(.004,.005,1,12),0x88dbff);
  private readonly dish:T.Mesh<T.BufferGeometry,T.MeshStandardMaterial>;
  private readonly offset:number;
  private readonly gripHeight:number;

  constructor(private replay:Replay,readonly track:Transfer){
    const glass=track.kind==='glass';
    const profile=glass?[[.043,-.07],[.053,.07],[.049,.07],[.039,-.064]]:
      [[.025,-.0554],[.058,-.04],[.095,.005],[.113,.045],[.107,.048],[.089,.013],[.053,-.034],[.025,-.044]];
    this.dish=enamel(new T.LatheGeometry(profile.map(([x,y])=>new T.Vector2(x,y)),48),glass?0xa9e1ef:0x487f93);
    this.dish.rotation.x=Math.PI/2;
    if(glass){this.dish.material.transparent=true;this.dish.material.opacity=.31;this.dish.material.depthWrite=false;}
    this.vessel.add(this.dish,this.grip,this.water);
    const radius=glass?.052:.108;
    // Removable rim handle. The robot grasps its outer vertical bar.
    this.offset=replay.robot==='g1'?.205:radius+.027;
    this.gripHeight=replay.robot==='g1'?.155:.038;
    const handle=new T.CatmullRomCurve3([vec(radius,0,.012),vec(this.offset,0,.012),vec(this.offset,0,this.gripHeight+.02),vec(radius,0,.049)]);
    this.grip.add(enamel(new T.TubeGeometry(handle,24,.007,10,false),0x3e5057));
    this.water.rotation.x=this.stream.rotation.x=Math.PI/2;
    this.water.visible=this.stream.visible=false;
    this.root.add(this.vessel,this.stream);
    const location=glass?track.pick:track.place;
    const base=sampleReplay(replay,glass?track.reach:track.lower).position;
    const floor=vec(...location);floor.z+=.011-(glass?.07:.0554);
    const towards=vec(location[0]-base[0],location[1]-base[1],0).normalize();
    this.root.add(trainingBasin(floor.toArray(),towards,glass));
  }

  update(time:number,robot:T.Group,model:ArticulatedRobot|null,rig:Cleaning,layer:SavedObjectLayer|null){
    const t=this.track;
    const pick=vec(...t.pick),place=vec(...t.place);
    const stretch=this.replay.robot==='stretch';
    const carry=robot.localToWorld(stretch?vec(.54,-.02,.94):vec(.13,.38,.93));
    const hover=place.clone();hover.z+=.15;
    const position=this.vessel.position.copy(pick);
    if(time>=t.release)position.copy(place);
    else if(time>=t.lower){
      const phase=2*(time-t.lower)/(t.release-t.lower);
      if(phase<1)position.copy(carry).lerp(hover,smooth(phase));
      else position.copy(hover).lerp(place,smooth(phase-1));
    }else if(time>=t.lift){
      const clear=pick.clone();
      if(t.kind==='glass'){
        const away=robot.position.clone().sub(pick);away.z=0;
        clear.addScaledVector(away.normalize(),.085);
      }else clear.z+=.12;
      if(time<t.lift+1)position.lerp(clear,smooth(time-t.lift));
      else position.copy(clear).lerp(carry,smooth(time-t.lift-1));
    }
    const bearing=(anchor:T.Vector3,base:T.Vector3)=>Math.atan2(base.y-anchor.y,base.x-anchor.x);
    const restBase=vec(...sampleReplay(this.replay,time<t.grasp?t.reach:t.release).position);
    this.grip.rotation.z=bearing(position,time<t.grasp||time>=t.release?restBase:robot.position);
    const hand=this.grip.localToWorld(vec(this.offset,0,this.gripHeight));
    if(time>=t.reach&&time<=t.end){
      const home=robot.localToWorld(stretch?vec(.48,-.02,.88):vec(.17,.27,.92));
      if(time<t.grasp)hand.lerp(home,1-smooth((time-t.reach)/(t.grasp-t.reach)));
      if(time>t.release)hand.lerp(home,smooth((time-t.release)/(t.end-t.release)));
      rig.reach(robot,model,hand,true);
    }
    if(t.objectId)this.dish.visible=!layer?.setPose(t.objectId,position);
    if(t.kind==='glass'){
      const fill=T.MathUtils.clamp((time-t.fillStart!)/(t.fillEnd!-t.fillStart!),0,1);
      const height=.112*fill;
      this.water.visible=fill>0;this.water.scale.y=height;this.water.position.z=-.064+height/2;
      this.stream.visible=time>=t.fillStart!&&time<t.fillEnd!;
      const outlet=pick.z-.07+.011+.19;
      const level=pick.z-.064+height;
      this.stream.position.set(pick.x,pick.y,(outlet+level)/2);
      this.stream.scale.y=outlet-level;
    }
    return hand;
  }
}
