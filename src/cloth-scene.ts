import * as T from 'three';
import type {RecordedCloth} from './replay';

export class ClothScene {
  readonly root=new T.Group();
  private readonly fabric:T.Mesh<T.BufferGeometry,T.MeshStandardMaterial>;
  private readonly tab=new T.Mesh(new T.BoxGeometry(.028,.056,.020),new T.MeshStandardMaterial({color:0xf0b377,roughness:.86}));
  private readonly marks:T.Mesh[];
  private readonly rotation=new T.Quaternion();
  private readonly nextTabPosition=new T.Vector3();

  constructor(private readonly cloth:RecordedCloth){
    const geometry=new T.BufferGeometry();
    geometry.setIndex(cloth.faces.flat());
    geometry.setAttribute('position',new T.Float32BufferAttribute(cloth.frames[0].points,3).setUsage(T.DynamicDrawUsage));
    geometry.computeVertexNormals();
    this.fabric=new T.Mesh(geometry,new T.MeshStandardMaterial({color:0xdf8751,side:T.DoubleSide,roughness:1}));
    this.fabric.frustumCulled=false;
    this.root.add(this.fabric,this.tab);
    this.marks=cloth.marks.map(position=>{
      const mark=new T.Mesh(new T.CircleGeometry(.009,10),new T.MeshStandardMaterial({color:0x795538}));
      mark.position.fromArray(position);
      this.root.add(mark);
      return mark;
    });
    this.update(0);
  }

  update(time:number){
    const frames=this.cloth.frames;
    const right=frames.findIndex(f=>f.time>time);
    const a=frames[right<0?frames.length-1:Math.max(0,right-1)];
    const b=right<0?a:frames[right];
    const mix=a===b?0:T.MathUtils.clamp((time-a.time)/(b.time-a.time),0,1);
    const positions=this.fabric.geometry.getAttribute('position');
    for(let i=0;i<positions.array.length;i++)positions.array[i]=T.MathUtils.lerp(a.points[i],b.points[i],mix);
    positions.needsUpdate=true;
    this.fabric.geometry.computeVertexNormals();
    this.tab.position.set(...a.tabPosition).lerp(this.nextTabPosition.set(...b.tabPosition),mix);
    this.tab.quaternion.set(a.tabQuaternion[1],a.tabQuaternion[2],a.tabQuaternion[3],a.tabQuaternion[0]);
    this.rotation.set(b.tabQuaternion[1],b.tabQuaternion[2],b.tabQuaternion[3],b.tabQuaternion[0]);
    this.tab.quaternion.slerp(this.rotation,mix).normalize();
    this.marks.forEach((mark,i)=>{mark.visible=!a.wiped.includes(i);});
  }
}
