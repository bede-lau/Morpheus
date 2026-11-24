import * as T from 'three';
import type {Cleaning} from './cleaning';
import type {ContactTrial, RobotId} from './replay';
import type {ArticulatedRobot} from './robot-model';

const smooth=(n:number)=>T.MathUtils.smoothstep(n,0,1);

export class ContactScene {
  readonly root=new T.Group();
  readonly fixture=new T.Group();
  readonly handle=new T.Vector3();
  private readonly base=new T.Vector3();
  private readonly travel=new T.Vector3();
  private indicator?:T.MeshStandardMaterial;

  constructor(readonly trial:ContactTrial){
    this.base.fromArray(trial.point);
    this.travel.fromArray(trial.travel);
    this.root.position.copy(this.base);
    this.root.add(this.fixture);
    const box=(width:number,depth:number,height:number,color:number,x:number,y:number,z:number)=>{
      const mesh=new T.Mesh(new T.BoxGeometry(width,depth,height),new T.MeshStandardMaterial({color,roughness:.72}));
      mesh.position.set(x,y,z);this.fixture.add(mesh);return mesh.material;
    };
    if(trial.kind==='drawer'){
      // Build the drawer from its near-side handle. Rails constrain it to a straight pull.
      box(.025,.29,.082,0x815737,-.049,0,-.226);
      box(.24,.29,.012,0xb89269,-.178,0,-.270);
      box(.24,.011,.052,0xbd9b77,-.178,-.14,-.243);
      box(.24,.011,.052,0xbd9b77,-.178,.14,-.243);
      box(.011,.29,.052,0xbd9b77,-.298,0,-.243);
      box(.012,.122,.018,0x253436,-.003,0,0);
      box(.016,.015,.215,0x253436,-.003,-.055,-.102);
      box(.016,.015,.215,0x253436,-.003,.055,-.102);
      box(.04,.13,.018,0x253436,-.025,0,-.204);
    }else{
      box(.145,.18,.018,0x537178,-.105,0,-.231);
      box(.055,.076,.21,0x698b91,-.105,0,-.117);
      box(.11,.057,.025,0x698b91,-.056,0,-.015);
      const [width,depth,height]=trial.padSize??[.028,.095,.042];
      this.indicator=box(width,depth,height,0xdcb54a,-width/2,0,0);
    }
  }

  update(time:number,robot:T.Group,model:ArticulatedRobot|null,rig:Cleaning,force?:number){
    const {start,contact,release,end}=this.trial;
    this.fixture.position.copy(this.travel).multiplyScalar(smooth((time-contact)/(release-contact)));
    this.handle.copy(this.base).add(this.fixture.position);
    if(this.indicator){
      const touched=force===undefined?time>=contact:force>1;
      this.indicator.color.setHex(touched?0x3ecf80:0xdcb54a);
      this.indicator.emissive.setHex(touched?0x097447:0x000000);
    }
    if(force!==undefined||time<start||time>end)return;
    const id=rig.robotId;
    const rest=robot.localToWorld(new T.Vector3(...HAND_REST[id]));
    const blend=smooth((time-start)/(contact-start))*smooth((end-time)/(end-release));
    rig.reach(robot,model,rest.lerp(this.handle,blend),true);
  }
}

const HAND_REST:Record<RobotId,[number,number,number]>={
  g1:[.17,.27,.92], go2:[.15,.25,.65], stretch:[.48,-.02,.88],
};
