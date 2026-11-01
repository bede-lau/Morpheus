import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { TransformControls } from 'three/addons/controls/TransformControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { SplatMesh, SparkRenderer, SplatFileType } from '@sparkjsdev/spark';
import { createSplatCutaway, suggestedCutawayHeight } from './cutaway';
import { computeBounds, estimateFloorZ, transformPoints } from './geometry';
import type { Collider, SceneInfo, Vec3 } from './types';

type Hooks = {onSelect:(id:string|null)=>void;onChange:(id:string,p:Vec3,s:Vec3,q:[number,number,number,number])=>void;onMeasure:(distance:number)=>void;onStatus:(text:string)=>void};
export class RoomViewer {
  scene=new THREE.Scene(); camera=new THREE.PerspectiveCamera(48,1,.02,250); renderer:THREE.WebGLRenderer;
  orbit:OrbitControls; transform:TransformControls; sourceRoot=new THREE.Group(); colliderRoot=new THREE.Group();
  splat:SplatMesh|null=null; mesh:THREE.Group|null=null; points:Float32Array=new Float32Array(); basePoints:Float32Array=new Float32Array();
  boxes=new Map<string,THREE.Mesh>(); selected:string|null=null; measuring=false; measurePoints:THREE.Vector3[]=[];
  scale=1; private offset=new THREE.Vector3(); private baseRotation=new THREE.Quaternion(); private resize:ResizeObserver;
  private helper:THREE.Object3D; private measureRoot=new THREE.Group(); private disposed=false; private generation=0;
  private info:SceneInfo|null=null; private down={x:0,y:0}; private showColliders=true;
  private cutaway:ReturnType<typeof createSplatCutaway>|null=null;
  constructor(private host:HTMLElement,private hooks:Hooks){
    THREE.Object3D.DEFAULT_UP.set(0,0,1); this.camera.up.set(0,0,1);this.camera.position.set(8,-10,8);
    this.scene.background=new THREE.Color('#e4e9e6');
    this.renderer=new THREE.WebGLRenderer({antialias:false});this.renderer.setPixelRatio(Math.min(devicePixelRatio,1.7));
    this.renderer.outputColorSpace=THREE.SRGBColorSpace;this.renderer.localClippingEnabled=true;this.host.append(this.renderer.domElement);
    const spark=new SparkRenderer({renderer:this.renderer});this.scene.add(spark,this.sourceRoot,this.colliderRoot,this.measureRoot);
    this.orbit=new OrbitControls(this.camera,this.renderer.domElement);this.orbit.enableDamping=true;this.orbit.maxDistance=60;this.orbit.minDistance=.3;
    this.orbit.maxPolarAngle=Math.PI*.93;
    this.transform=new TransformControls(this.camera,this.renderer.domElement);this.transform.setSize(.75);this.transform.setSpace('world');
    this.helper=this.transform.getHelper();this.scene.add(this.helper);
    this.transform.addEventListener('dragging-changed',e=>{this.orbit.enabled=!e.value});
    this.transform.addEventListener('mouseUp',()=>{const o=this.transform.object;if(o&&this.selected)this.hooks.onChange(this.selected,o.position.toArray() as Vec3,o.scale.toArray() as Vec3,[o.quaternion.w,o.quaternion.x,o.quaternion.y,o.quaternion.z]);});
    const grid=new THREE.GridHelper(30,60,0xaebfb8,0xcbd5cf);grid.rotation.x=Math.PI/2;grid.position.z=-.035;this.scene.add(grid);
    this.scene.add(new THREE.HemisphereLight(0xffffff,0x778a7c,2));const sun=new THREE.DirectionalLight(0xffffff,2);sun.position.set(3,-5,10);this.scene.add(sun);
    this.renderer.domElement.addEventListener('pointerdown',e=>{this.down={x:e.clientX,y:e.clientY}});
    this.renderer.domElement.addEventListener('pointerup',this.onClick);
    this.resize=new ResizeObserver(()=>{const w=this.host.clientWidth,h=this.host.clientHeight;this.camera.aspect=w/h;this.camera.updateProjectionMatrix();this.renderer.setSize(w,h)});this.resize.observe(host);
    this.renderer.setAnimationLoop(()=>{if(!this.disposed){this.orbit.update();this.renderer.render(this.scene,this.camera)}});
  }
  async load(splatUrl:string,meshUrl:string|undefined,name:string,kind:'sample'|'upload',fileName:string,up:'y-down'|'y-up'|'z-up'='y-down'){
    const token=++this.generation;this.hooks.onStatus('Loading splat and reference geometry…');this.clearSource();
    const extension=fileName.split('.').pop()?.toLowerCase();
    const fileType=extension==='spz'?SplatFileType.SPZ:extension==='ply'?SplatFileType.PLY:extension==='splat'?SplatFileType.SPLAT:undefined;
    const room=new SplatMesh({url:splatUrl,fileName,fileType});this.sourceRoot.add(room);this.splat=room;
    let geometry:THREE.Group|null;
    try { const [,loaded]=await Promise.all([room.initialized,meshUrl?new GLTFLoader().loadAsync(meshUrl):Promise.resolve(null)]);geometry=loaded?.scene||null; }catch(e){if(token===this.generation)this.clearSource();throw e}
    if(token!==this.generation){room.dispose();return null}
    this.mesh=geometry;
    if(this.mesh){this.sourceRoot.add(this.mesh);this.mesh.traverse(o=>{if(o instanceof THREE.Mesh)o.material=new THREE.MeshStandardMaterial({color:0x648c80,transparent:true,opacity:.24,side:THREE.DoubleSide,wireframe:false,depthWrite:false})});this.mesh.visible=false;}
    this.baseRotation.setFromAxisAngle(new THREE.Vector3(1,0,0),up==='y-down'?-Math.PI/2:up==='y-up'?Math.PI/2:0);
    this.sourceRoot.quaternion.copy(this.baseRotation);this.sourceRoot.updateMatrixWorld(true);
    this.hooks.onStatus('Sampling the room surfaces…');
    await new Promise(r=>setTimeout(r,25));
    const raw=this.mesh?this.sampleMesh(this.mesh):this.sampleSplat(room);
    const bounds=computeBounds(raw,.008),floor=estimateFloorZ(raw);
    this.offset.set(-bounds.center[0],-bounds.center[1],-floor);this.sourceRoot.position.copy(this.offset);
    this.basePoints=transformPoints(raw,1,this.offset.toArray() as Vec3);this.points=this.basePoints;this.scale=1;
    const aligned=computeBounds(this.points,.008);this.info={name,splats:room.numSplats,points:this.points.length/3,hasMesh:!!this.mesh,bounds:aligned.size,floorZ:0,sourceKind:kind,fileName};
    this.cutaway=createSplatCutaway(room,suggestedCutawayHeight(0,Math.max(.2,aligned.max[2])));this.setCutaway(true);
    this.home();this.hooks.onStatus('Room loaded · cutaway view. Check scale before export.');return this.info;
  }
  private sampleSplat(room:SplatMesh){const out:number[]=[];const stride=Math.max(1,Math.ceil(room.numSplats/90000));room.updateMatrixWorld(true);room.forEachSplat((i,p,s,_q,opacity)=>{if(i%stride||opacity<.3||Math.max(s.x,s.y,s.z)>.3)return;const v=p.clone().applyMatrix4(room.matrixWorld);if(Number.isFinite(v.lengthSq()))out.push(v.x,v.y,v.z)});return new Float32Array(out)}
  private sampleMesh(root:THREE.Group){
    const triangles:{a:THREE.Vector3;b:THREE.Vector3;c:THREE.Vector3;area:number}[]=[];let total=0;
    root.updateMatrixWorld(true);root.traverse(o=>{if(!(o instanceof THREE.Mesh))return;const g=o.geometry,p=g.getAttribute('position'),ix=g.getIndex();const count=ix?ix.count:p.count;
      for(let i=0;i<count;i+=3){const a=new THREE.Vector3().fromBufferAttribute(p,ix?ix.getX(i):i).applyMatrix4(o.matrixWorld),b=new THREE.Vector3().fromBufferAttribute(p,ix?ix.getX(i+1):i+1).applyMatrix4(o.matrixWorld),c=new THREE.Vector3().fromBufferAttribute(p,ix?ix.getX(i+2):i+2).applyMatrix4(o.matrixWorld);const area=new THREE.Vector3().subVectors(b,a).cross(new THREE.Vector3().subVectors(c,a)).length()*.5;if(area>0&&Number.isFinite(area)){triangles.push({a,b,c,area});total+=area}}});
    if(!triangles.length)throw new Error('The companion mesh contains no usable triangles.');
    const out:number[]=[];const target=90000;let budget=0;let index=0;
    for(const t of triangles){budget+=t.area/total*target;const n=Math.floor(budget);budget-=n;for(let j=0;j<n;j++){index++;const u=Math.sqrt((index*.618033988749895)%1),v=(index*.414213562373095)%1;const p=t.a.clone().multiplyScalar(1-u).addScaledVector(t.b,u*(1-v)).addScaledVector(t.c,u*v);out.push(p.x,p.y,p.z)}}return new Float32Array(out);
  }
  setScale(value:number){if(!Number.isFinite(value)||value<=0)throw new Error('Scale must be positive.');const ratio=value/this.scale;this.scale=value;this.sourceRoot.scale.setScalar(value);this.sourceRoot.position.copy(this.offset).multiplyScalar(value);this.points=transformPoints(this.basePoints,value);if(this.cutaway)this.cutaway.setHeight(this.cutaway.height*ratio);this.clearMeasure();this.home()}
  setCutaway(enabled:boolean){this.cutaway?.setEnabled(enabled);this.mesh?.traverse(o=>{if(o instanceof THREE.Mesh){const materials=Array.isArray(o.material)?o.material:[o.material];for(const material of materials){material.clippingPlanes=enabled&&this.cutaway?this.cutaway.planes:[];material.needsUpdate=true;}}})}
  getTransform(){this.sourceRoot.updateMatrixWorld(true);return {matrix:this.sourceRoot.matrixWorld.toArray(),matrixOrder:'column-major',unitsPerSourceUnit:this.scale}}
  getGeometryPoints(useMesh:boolean){return useMesh&&this.mesh?this.points:this.splat?this.sampleSplat(this.splat):this.points}
  interior(){if(!this.info||!this.splat)return;this.sourceRoot.updateMatrixWorld(true);const position=new THREE.Vector3(0,0,0).applyMatrix4(this.sourceRoot.matrixWorld),target=new THREE.Vector3(-1,0,1.5).applyMatrix4(this.sourceRoot.matrixWorld);this.camera.position.copy(position);this.orbit.target.copy(target);this.orbit.update()}
  home(){let center:Vec3,size:Vec3;if(this.points.length){const b=computeBounds(this.points,.008);center=b.center;size=b.size}else if(this.boxes.size){this.colliderRoot.updateMatrixWorld(true);const b=new THREE.Box3().setFromObject(this.colliderRoot);center=b.getCenter(new THREE.Vector3()).toArray() as Vec3;size=b.getSize(new THREE.Vector3()).toArray() as Vec3}else return;const r=Math.max(...size,1);this.orbit.target.set(center[0],center[1],size[2]*.35);this.camera.position.copy(this.orbit.target).add(new THREE.Vector3(r*.78,-r*.98,r*.78));this.camera.near=.02;this.camera.far=Math.max(100,r*12);this.camera.updateProjectionMatrix();this.orbit.update()}
  top(){if(!this.points.length)return;const b=computeBounds(this.points);this.orbit.target.set(b.center[0],b.center[1],0);this.camera.position.set(b.center[0],b.center[1]-.01,Math.max(...b.size)*1.6);this.orbit.update()}
  setLayers(scan:boolean,mesh:boolean,colliders:boolean){if(this.splat)this.splat.visible=scan;if(this.mesh)this.mesh.visible=mesh;this.colliderRoot.visible=colliders;this.showColliders=colliders;if(!colliders)this.transform.detach();else if(this.selected)this.select(this.selected)}
  setBoxes(colliders:Collider[]){
    this.transform.detach();this.colliderRoot.clear();for(const mesh of this.boxes.values()){mesh.geometry.dispose();(mesh.material as THREE.Material).dispose();for(const child of mesh.children){if(child instanceof THREE.LineSegments){child.geometry.dispose();(child.material as THREE.Material).dispose()}}}this.boxes.clear();
    for(const c of colliders){const color=c.role==='floor'?0x859f93:0x36a38a;const box=new THREE.Mesh(new THREE.BoxGeometry(1,1,1),new THREE.MeshBasicMaterial({color,transparent:true,opacity:c.role==='floor'?.09:.16,depthWrite:false}));box.position.fromArray(c.position);box.scale.fromArray(c.size);if(c.quaternion){const[w,x,y,z]=c.quaternion;box.quaternion.set(x,y,z,w).normalize()}box.userData.id=c.id;
      const lines=new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(1,1,1)),new THREE.LineBasicMaterial({color,transparent:true,opacity:.65}));box.add(lines);this.colliderRoot.add(box);this.boxes.set(c.id,box)}this.select(this.selected);
  }
  select(id:string|null){this.selected=id;this.transform.detach();for(const [key,box]of this.boxes){const mat=box.material as THREE.MeshBasicMaterial;mat.color.set(key===id?0xe6a443:key==='floor'?0x859f93:0x36a38a);mat.opacity=key===id?.4:.14;}const selected=id?this.boxes.get(id):undefined;if(selected&&this.showColliders)this.transform.attach(selected)}
  setTool(tool:'translate'|'scale'|'rotate'){this.transform.setMode(tool)}
  setMeasuring(active:boolean){this.measuring=active;this.clearMeasure();if(active)this.transform.detach();this.renderer.domElement.style.cursor=active?'crosshair':''}
  private clearMeasure(){this.measurePoints=[];this.measureRoot.traverse(o=>{if(o instanceof THREE.Mesh||o instanceof THREE.Line){o.geometry.dispose();(o.material as THREE.Material).dispose()}});this.measureRoot.clear()}
  private onClick=(e:PointerEvent)=>{
    if(Math.hypot(e.clientX-this.down.x,e.clientY-this.down.y)>5||this.transform.dragging||this.transform.axis)return;
    const rect=this.renderer.domElement.getBoundingClientRect(),ray=new THREE.Raycaster();ray.setFromCamera(new THREE.Vector2((e.clientX-rect.left)/rect.width*2-1,-(e.clientY-rect.top)/rect.height*2+1),this.camera);
    if(this.measuring){let hits=this.mesh?ray.intersectObject(this.mesh,true):this.splat?ray.intersectObject(this.splat,false):[];const hit=hits.find(h=>!this.cutaway?.enabled||h.point.z<=this.cutaway.height+.03);if(!hit)return;if(this.measurePoints.length===2)this.clearMeasure();this.measurePoints.push(hit.point.clone());const dot=new THREE.Mesh(new THREE.SphereGeometry(.045),new THREE.MeshBasicMaterial({color:0xe5a33c,depthTest:false}));dot.position.copy(hit.point);dot.renderOrder=100;this.measureRoot.add(dot);if(this.measurePoints.length===2){const line=new THREE.Line(new THREE.BufferGeometry().setFromPoints(this.measurePoints),new THREE.LineBasicMaterial({color:0xe5a33c,depthTest:false}));line.renderOrder=100;this.measureRoot.add(line);this.hooks.onMeasure(this.measurePoints[0].distanceTo(this.measurePoints[1]));}return;}
    if(!this.showColliders)return;const hit=ray.intersectObjects([...this.boxes.values()],false)[0];this.hooks.onSelect(hit?hit.object.userData.id:null);
  }
  private clearSource(){this.transform.detach();this.cutaway?.dispose();this.cutaway=null;if(this.splat){this.splat.dispose();this.splat=null}if(this.mesh)this.mesh.traverse(o=>{if(o instanceof THREE.Mesh){o.geometry.dispose();if(Array.isArray(o.material))o.material.forEach(m=>m.dispose());else o.material.dispose()}});this.sourceRoot.clear();this.sourceRoot.position.set(0,0,0);this.sourceRoot.scale.setScalar(1);this.sourceRoot.quaternion.identity();this.mesh=null;this.points=new Float32Array();this.basePoints=new Float32Array();this.clearMeasure()}
  unload(){this.generation++;this.clearSource()}
  dispose(){this.disposed=true;this.generation++;this.resize.disconnect();this.renderer.setAnimationLoop(null);this.clearSource();this.setBoxes([]);this.transform.dispose();this.orbit.dispose();this.renderer.dispose();this.renderer.domElement.remove()}
}
