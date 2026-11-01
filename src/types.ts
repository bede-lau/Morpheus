export type Vec3 = [number,number,number];
export type Collider = {id:string; name:string; type:'box'; position:Vec3; size:Vec3; quaternion?:[number,number,number,number]; role?:'floor'|'wall'|'obstacle'|'surface'; source?:string};
export type SceneManifest = {version:1;name:string;units:'m';upAxis:'Z';source:Record<string,unknown>;objects:Collider[]};
export type SceneInfo = {name:string;splats:number;points:number;hasMesh:boolean;bounds:Vec3;floorZ:number;sourceKind:'sample'|'upload';fileName:string};
