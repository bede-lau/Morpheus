"""Local SimReady catalogue, Decisions matching and mesh bundle API (port 8001)."""
from __future__ import annotations
import base64
import io
import json
import math
import os
from pathlib import Path
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import urllib.error
import urllib.request
from urllib.parse import urlparse
import xml.etree.ElementTree as ET
import zipfile
import threading

from .simready import CACHE, ROOT, catalogue, prepared_assets, asset_entry
from .export import validate_manifest, export_files

SCENE_LOCK = threading.Lock()

def normalize_object_scene(body):
    if not isinstance(body,dict) or body.get('version')!=1 or body.get('kind')!='simready-replacements':
        raise ValueError('Expected version 1 simready-replacements scene')
    if body.get('units')!='m' or body.get('upAxis')!='Z': raise ValueError('Expected metres, Z up')
    source=body.get('source',{})
    if not isinstance(source,dict) or not isinstance(source.get('fileName'),str): raise ValueError('Source filename is required')
    transform=source.get('transform',{})
    vector(transform.get('matrix'),16,'source transform')
    if transform.get('matrixOrder')!='column-major': raise ValueError('Expected column-major source transform')
    replacements=body.get('replacements')
    if not isinstance(replacements,list) or len(replacements)>30: raise ValueError('At most 30 object replacements')
    prepared={a['id']:a for a in prepared_assets()}
    ids=set()
    for item in replacements:
        if not isinstance(item,dict) or item.get('assetId') not in prepared: raise ValueError('Unprepared asset')
        ident=item.get('id')
        if not isinstance(ident,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,96}',ident) or ident in ids: raise ValueError('Invalid or duplicate object ID')
        ids.add(ident)
        vector(item.get('position'),3,'object position')
        q=vector(item.get('quaternion'),4,'object quaternion')
        length=math.hypot(*q)
        if length<1e-10: raise ValueError('Zero quaternion')
        item['quaternion']=[v/length for v in q]
        scale=item.get('scale');vector([scale],1,'object scale')
        if not .01<=scale<=100: raise ValueError('Invalid object scale')
        selection=item.get('selection',{})
        vector(selection.get('position'),3,'selection position')
        size=vector(selection.get('size'),3,'selection size')
        if any(v<=0 or v>5 for v in size): raise ValueError('Selection dimensions must be 0–5 metres')
        if item.get('chosenBy') not in ('manual','decisions'): raise ValueError('Record how the asset was chosen')
        item['asset']=prepared[item['assetId']]
    try: json.dumps(body,allow_nan=False)
    except (ValueError,TypeError): raise ValueError('Scene must contain finite JSON') from None
    return {'version':1,'kind':'simready-replacements','units':'m','upAxis':'Z',
            'source':source,'replacements':replacements,'interactionValidated':False}

def saved_scene():
    path=CACHE/'scene.json'
    return json.loads(path.read_text()) if path.exists() else {'version':1,'kind':'simready-replacements','units':'m','upAxis':'Z','revision':0,'source':None,'replacements':[],'interactionValidated':False}

def save_scene(body):
    normalized=normalize_object_scene(body)
    with SCENE_LOCK:
        current=saved_scene()
        if body.get('baseRevision')!=current['revision']: raise ValueError('Scene changed in another editor. Reload before saving.')
        normalized['revision']=current['revision']+1
        CACHE.mkdir(parents=True,exist_ok=True)
        temporary=CACHE/'scene.json.tmp'
        temporary.write_text(json.dumps(normalized,indent=2,allow_nan=False))
        temporary.replace(CACHE/'scene.json')
    return normalized

def load_local_key():
    key = os.environ.get('OPENAI_API_KEY', '').strip()
    if key: return key
    path = ROOT / '.env'
    if path.exists():
        for line in path.read_text().splitlines():
            match = re.match(r'^\s*(?:export\s+)?OPENAI_API_KEY\s*=\s*(.*?)\s*$', line)
            if match: return match[1].strip('"\'')
    return ''

def call_decisions(payload):
    key = load_local_key()
    if not key: raise ValueError('Set OPENAI_API_KEY in environment-pipeline/.env, then try again.')
    request = urllib.request.Request('https://api.openai.com/v1/decisions', data=json.dumps(payload).encode(),
        headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=90) as response: return json.load(response)
    except urllib.error.HTTPError as exc:
        # Never forward credentials, raw request data, or unexpected upstream bodies.
        raise ValueError(f'OpenAI Decisions returned HTTP {exc.code}. Check API access, credits and model availability.') from None

def match_objects(body, decide=call_decisions, assets=None):
    images = body.get('images', []) if isinstance(body, dict) else []
    if not isinstance(images, list) or not 1 <= len(images) <= 4: raise ValueError('Provide 1–4 object crops.')
    for image in images:
        if not isinstance(image, str) or not re.fullmatch(r'data:image/(jpeg|png);base64,[A-Za-z0-9+/=]+', image) or len(image) > 3_000_000:
            raise ValueError('Object crops must be inline JPEG/PNG images smaller than 3 MB.')
    assets = prepared_assets() if assets is None else assets
    if not assets: raise ValueError('Prepare SimReady assets before matching.')
    categories = sorted({a['category'] for a in assets} - {'other'})
    evidence = [{'role':'user', 'content':[
        {'type':'input_text','text':'These are views of ONE selected object in a Gaussian-splat room capture. Classify the central object, ignoring the surrounding counter, walls and background. Blurry or ambiguous evidence should remain unknown.'},
        *[{'type':'input_image','image_url':image} for image in images]]}]
    result = decide({'model':'gpt-6-luna','input':evidence,'questions':[
        {'name':'category','type':'choice','instructions':'Which available object category fits the selected object? Use unknown for ambiguous, multiple, or unsupported objects.',
         'choices':[{'value':c,'description':f'A {c}.'} for c in categories] + [{'value':'unknown','description':'Ambiguous object or category absent from this prepared catalogue.'}]},
        {'name':'handle','type':'predicate','instructions':'Does the selected object have a visibly attached handle?'},
        {'name':'shallow','type':'predicate','instructions':'Is the object a shallow dish with a broad, nearly flat interior?'},
    ]})
    answers = result.get('answers', [])
    answer = next((a for a in answers if a.get('name')=='category'), {})
    chosen = answer.get('choice') if answer.get('type')=='choice' else 'unknown'
    if chosen not in categories: chosen = 'unknown'
    shortlist = [a for a in assets if a['category']==chosen][:16]
    trace = [{'stage':'category','answers':answers,'requestId':result.get('id')}]
    if not shortlist:
        return {'category':'unknown','candidates':[],'trace':trace,'reviewRequired':True,
                'message':'No supported match. Select a clearer object or prepare more catalogue assets.'}
    questions = []
    for asset in shortlist:
        preview = CACHE/'prepared'/asset['id']/'preview.jpg'
        if not preview.exists(): continue
        candidate_image = 'data:image/jpeg;base64,' + base64.b64encode(preview.read_bytes()).decode()
        questions.append({'name':asset['id'],'type':'score','instructions':[
            {'type':'input_text','text':f'Compare this NVIDIA candidate ({asset["name"]}) with the selected object in the shared scan images. Rate visible shape, proportions, handle/rim structure and appearance. Ignore viewpoint/background differences. Do not claim exact identity or infer real dimensions from the image.'},
            {'type':'input_image','image_url':candidate_image}],
            'levels':[{'label':'Mismatch','description':'Different object or incompatible shape.'},
                      {'label':'Weak','description':'Same broad category; substantial shape mismatch or insufficient evidence.'},
                      {'label':'Plausible','description':'Similar overall shape and proportions; details differ or are obscured.'},
                      {'label':'Close','description':'Strong visible agreement in shape, proportions and distinguishing features.'}]})
    if not questions: raise ValueError('Candidate preview images are missing. Prepare assets again.')
    ranked = decide({'model':'gpt-6-luna','input':evidence,'questions':questions})
    trace.append({'stage':'candidate_comparison','answers':ranked.get('answers',[]),'requestId':ranked.get('id')})
    candidates = []
    allowed = {a['id']:a for a in shortlist}
    for answer in ranked.get('answers', []):
        score = answer.get('score')
        if answer.get('type') != 'score' or answer.get('name') not in allowed: continue
        if not isinstance(score,(int,float)) or not math.isfinite(score) or not 0<=score<=3: continue
        candidates.append({'asset':allowed[answer['name']], 'score':score,
                           'confidence':answer.get('confidence'), 'probabilities':answer.get('probabilities',[])})
    candidates.sort(key=lambda c:c['score'], reverse=True)
    return {'category':chosen,'candidates':candidates,'trace':trace,'reviewRequired':True,
            'message':'Scores rank this prepared catalogue only. Review the crop and fit before replacement.'}

def vector(value, n, name):
    if not isinstance(value,list) or len(value)!=n or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in value):
        raise ValueError(f'Invalid {name}')
    return value

def replacement_bundle(body):
    if not isinstance(body,dict): raise ValueError('Expected a scene object')
    replacements = body.get('replacements')
    if not isinstance(replacements,list) or not 1 <= len(replacements) <= 30: raise ValueError('Export 1–30 replacements.')
    prepared = {a['id']:a for a in prepared_assets()}
    root = ET.Element('mujoco', model='morpheus_objects')
    ET.SubElement(root,'compiler',angle='radian')
    ET.SubElement(root,'option', timestep='0.002', gravity='0 0 -9.81')
    asset_xml = ET.SubElement(root,'asset')
    world = ET.SubElement(root,'worldbody')
    floor_z = body.get('floorZ',0)
    vector([floor_z],1,'floorZ')
    ET.SubElement(world,'geom',name='env_floor',type='plane',size='10 10 .1',pos=f'0 0 {floor_z}')
    # Optional calibrated collision environment, preserving its own world frame.
    environment = body.get('environment')
    if environment is not None:
        normalized = validate_manifest(environment)
        world.remove(world[0])
        inc = ET.fromstring(export_files(normalized)['environment.xml'])
        for geom in inc.find('worldbody'): world.append(geom)
    files, manifest = {}, []
    for i, item in enumerate(replacements):
        if not isinstance(item,dict) or item.get('assetId') not in prepared: raise ValueError('Unprepared asset in export')
        entry = prepared[item['assetId']]
        pos = vector(item.get('position'),3,'position')
        q = vector(item.get('quaternion'),4,'quaternion WXYZ')
        scale = item.get('scale')
        vector([scale],1,'scale')
        if not .01 <= scale <= 100: raise ValueError('Asset scale must be 0.01–100')
        length = math.hypot(*q)
        if length < 1e-10: raise ValueError('Zero quaternion')
        q = [v/length for v in q]
        ident = f'object_{i}'
        folder = CACHE/'prepared'/entry['id']
        mesh_path = f'assets/{ident}.obj'
        files[mesh_path] = (folder/'visual.obj').read_bytes()
        files[f'assets/{ident}.glb'] = (folder/'model.glb').read_bytes()
        ET.SubElement(asset_xml,'mesh',name=ident,file=mesh_path,scale=' '.join([str(scale)]*3))
        node = ET.SubElement(world,'body',name=ident,pos=' '.join(map(str,pos)),quat=' '.join(map(str,q)))
        # Static visual and convex-hull collision approximation. No fabricated mass
        # or articulation: real robot interaction requires engine-specific assets.
        ET.SubElement(node,'geom',type='mesh',mesh=ident,rgba='.82 .82 .78 1')
        source = CACHE/'source'/entry['id']
        for license_file in (source/'.metadata').glob('*LICENSE*'):
            files[f'credits/{entry["id"]}/{license_file.name}'] = license_file.read_bytes()
        manifest.append({**item,'quaternion':q,'asset':entry,'collision':'MuJoCo convex hull; static',
                         'interactionValidated':False})
    ET.indent(root)
    xml = ET.tostring(root,encoding='unicode')
    files['scene.xml'] = xml.encode()
    report = {'compile':'not_run','interaction':'not_validated','warnings':[
        'Mesh collisions use convex hulls: bowl and mug cavities are filled.',
        'Objects are static. No grasping, mass, articulation, policy or robot comparison has been validated.',
        'Source splat selections and fitted scale require review.']}
    try:
        import mujoco
        model = mujoco.MjModel.from_xml_string(xml, {k:v for k,v in files.items() if k.endswith('.obj')})
        report.update(compile='passed', engineVersion=mujoco.__version__, geomCount=model.ngeom)
    except ImportError: pass
    except Exception as exc: raise ValueError(f'MuJoCo could not compile this object scene: {exc}') from None
    files['scene.json'] = json.dumps({'version':1,'kind':'simready-replacements','units':'m','upAxis':'Z',
        'source':body.get('source',{}),'replacements':manifest,'environment':environment},indent=2).encode()
    files['validation.json'] = json.dumps(report,indent=2).encode()
    files['CREDITS.txt'] = ('NVIDIA SimReady Assets — CC-BY-4.0\nhttps://huggingface.co/datasets/nvidia/simready-assets\n'
        'Modified: USD visual geometry converted to centered metre/Z-up GLB and OBJ; user-fitted pose/scale.\n'
        'Original USD physics metadata is not preserved in this MJCF approximation. Source URLs/revisions in scene.json.\n').encode()
    output = io.BytesIO()
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as archive:
        for name,data in files.items(): archive.writestr(name,data)
    return output.getvalue()

class Handler(BaseHTTPRequestHandler):
    def send(self,status,data,content_type='application/json'):
        if not isinstance(data,bytes): data=json.dumps(data,allow_nan=False).encode()
        self.send_response(status); self.send_header('Content-Type',content_type)
        self.send_header('Content-Length',str(len(data))); self.send_header('Cache-Control','no-store')
        self.end_headers(); self.wfile.write(data)
    def allowed(self):
        origin=self.headers.get('Origin')
        return not origin or urlparse(origin).hostname in ('127.0.0.1','localhost','::1')
    def do_GET(self):
        if not self.allowed(): return self.send(403,{'error':'Local origins only'})
        path=urlparse(self.path).path
        if path=='/api/objects/status':
            return self.send(200,{'decisionsConfigured':bool(load_local_key()),'preparedCount':len(prepared_assets())})
        if path=='/api/objects/scene': return self.send(200,saved_scene())
        if path=='/api/objects/catalogue':
            try: return self.send(200,{'dataset':'nvidia/simready-assets','catalogueCount':len(catalogue()['assets']),'assets':prepared_assets()})
            except ValueError as exc: return self.send(503,{'error':str(exc)})
        match=re.fullmatch(r'/api/objects/assets/([A-Za-z0-9_]+)/(?P<file>model\.glb|preview\.jpg)',path)
        if match:
            try: asset_entry(match[1])
            except ValueError: return self.send(404,{'error':'Unknown asset'})
            target=CACHE/'prepared'/match[1]/match['file']
            if target.is_file(): return self.send(200,target.read_bytes(),'model/gltf-binary' if match['file'].endswith('.glb') else 'image/jpeg')
        self.send(404,{'error':'Not found'})
    def do_POST(self):
        if not self.allowed(): return self.send(403,{'error':'Local origins only'})
        if self.path not in ('/api/objects/match','/api/objects/export','/api/objects/scene'): return self.send(404,{'error':'Not found'})
        try:
            size=int(self.headers.get('Content-Length','-1'))
            if not 0<size<=12_000_000: raise ValueError('Request size must be below 12 MB')
            if self.headers.get_content_type()!='application/json': raise ValueError('Expected application/json')
            body=json.loads(self.rfile.read(size))
            if self.path.endswith('/match'): self.send(200,match_objects(body))
            elif self.path.endswith('/scene'): self.send(200,save_scene(body))
            else: self.send(200,replacement_bundle(body),'application/zip')
        except (ValueError,KeyError,TypeError) as exc: self.send(400,{'error':str(exc)})
        except Exception:
            import traceback
            traceback.print_exc()
            self.send(502,{'error':'Object service failed. See the local server log; no replacement was applied.'})

if __name__=='__main__':
    print('Morpheus objects: http://127.0.0.1:8001 (key loaded from environment or .env)',flush=True)
    ThreadingHTTPServer(('127.0.0.1',8001),Handler).serve_forever()
