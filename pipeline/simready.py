"""Pinned NVIDIA SimReady catalogue and selective USD -> GLB conversion.

Downloads only allowlisted catalogue IDs. Original USD packages and licenses
remain alongside derived meshes; physics metadata is not claimed to survive GLB.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'output' / 'simready'
REPO = 'nvidia/simready-assets'
BASE = f'https://huggingface.co/datasets/{REPO}'
API = f'https://huggingface.co/api/datasets/{REPO}'
DEFAULT_IDS = ['dishware_plastic_bigPlate_a01', 'dishware_plastic_smallPlate_a01',
               'kitchenware_bowl_b01', 'kitchenware_bowl_c01', 'kitchenware_mug_a01',
               'kitchenware_mug_b01', 'kitchenware_cup_a01', 'kitchenware_spoon_h01']
# Study/office set for the office scan: indexed alongside the kitchenware so Decisions can match them.
OFFICE_IDS = ['book_open_a01', 'magazines_stack_a01', 'misc_notebook_a01', 'bin_waste_a01', 'misc_basket_plastic_a01',
              'storage_basket_plastic_a01', 'decor_candle_a01', 'misc_glass_vase_a01', 'measurement_clock_alarm_a01',
              'sofa_soft_b01', 'chair_office_fabric_manager_a01', 'computer_monitor_office_a01', 'computer_keyboard_a01',
              'misc_notebook_a02', 'trashCan_domed_blue_b01', 'chair_office_amf_black_a01']
MAX_BYTES = 100 * 1024 * 1024

def fetch_json(url):
    with urllib.request.urlopen(url, timeout=45) as response:
        return json.load(response)

def tree(revision, path=''):
    url = f'{API}/tree/{revision}{"/" + path if path else ""}?recursive={str(bool(path)).lower()}&limit=1000'
    rows = []
    while url:
        with urllib.request.urlopen(url, timeout=45) as response:
            rows.extend(json.load(response))
            match = re.search(r'<([^>]+)>; rel="next"', response.headers.get('Link', ''))
            url = match.group(1) if match else None
    return rows

def category(asset_id):
    words = asset_id.lower()
    for token, label in [('plate', 'plate'), ('bowl', 'bowl'), ('mug', 'mug'), ('cup', 'cup'),
                         ('spoon', 'spoon'), ('fork', 'fork'), ('knife', 'knife'), ('kettle', 'kettle'),
                         ('cuttingboard', 'cutting board'), ('jug', 'jug'), ('pan', 'pan'), ('tray', 'tray'),
                         ('book', 'book'), ('notebook', 'book'), ('magazine', 'book'), ('sofa', 'armchair'), ('chair', 'chair'),
                         ('bin_waste', 'bin'), ('trashcan', 'bin'), ('basket', 'basket'), ('candle', 'candle'), ('vase', 'vase'), ('clock', 'clock'),
                         ('monitor', 'monitor'), ('keyboard', 'keyboard')]:
        if token in words: return label
    return 'other'

def index_catalogue():
    revision = fetch_json(API)['sha']
    entries = []
    for row in tree(revision):
        ident = row['path']
        if row['type'] != 'directory' or not (ident.startswith(('kitchenware_', 'dishware_')) or ident in OFFICE_IDS): continue
        entries.append({'id': ident, 'category': category(ident),
                        'name': re.sub(r'([a-z])([A-Z])', r'\1 \2', ident).replace('_', ' '),
                        'sourceUrl': f'{BASE}/tree/{revision}/{ident}', 'license': 'CC-BY-4.0',
                        'attribution': 'NVIDIA SimReady Assets', 'revision': revision})
    CACHE.mkdir(parents=True, exist_ok=True)
    result = {'dataset': REPO, 'revision': revision, 'assets': entries}
    (CACHE / 'catalogue.json').write_text(json.dumps(result, indent=2))
    return result

def catalogue():
    path = CACHE / 'catalogue.json'
    if not path.exists(): raise ValueError('Run python -m pipeline.simready --prepare first.')
    return json.loads(path.read_text())

def asset_entry(ident):
    entry = next((a for a in catalogue()['assets'] if a['id'] == ident), None)
    if entry is None: raise ValueError('Unknown SimReady asset ID')
    return entry

def download_asset(entry):
    ident, revision = entry['id'], entry['revision']
    folder = CACHE / 'source' / ident
    if (folder / 'download.json').exists():
        data = json.loads((folder / 'download.json').read_text())
        if data['revision'] == revision: return data
    rows = [r for r in tree(revision, ident) if r['type'] == 'file']
    # Source assets include duplicate DCC textures. Keep USD composition, runtime
    # textures, license/metadata, and a thumbnail. Never fetch an entire dataset.
    rows = [r for r in rows if '/source_assets/' not in r['path'] and
            ('/.thumbs/' not in r['path'] or '/.thumbs/256x256/sm_' in r['path'])]
    total = sum(r.get('size', 0) for r in rows)
    if total > MAX_BYTES: raise ValueError(f'Asset exceeds the {MAX_BYTES // 2**20} MB per-object download limit.')
    def download(row):
        rel = Path(row['path']).relative_to(ident)
        if '..' in rel.parts: raise ValueError('Invalid catalogue path')
        dest = folder / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists() or dest.stat().st_size != row['size']:
            url = f'{BASE}/resolve/{revision}/{row["path"]}'
            with urllib.request.urlopen(url, timeout=90) as response:
                data = response.read(MAX_BYTES + 1)
            if len(data) != row['size']: raise ValueError(f'Incomplete asset file: {rel}')
            expected = row.get('lfs', {}).get('oid')
            if expected and hashlib.sha256(data).hexdigest() != expected: raise ValueError('Asset hash mismatch')
            dest.write_bytes(data)
        return {'path': str(rel), 'sha256': hashlib.sha256(dest.read_bytes()).hexdigest()}
    with ThreadPoolExecutor(max_workers=4) as pool: files = list(pool.map(download, rows))
    roots = sorted(folder.glob('simready_usd/*.usd')) + sorted(folder.glob('simready_usd/*.usda'))
    if not roots: raise ValueError('Asset has no supported root USD')
    result = {'revision': revision, 'root': str(roots[0].relative_to(folder)), 'files': files}
    (folder / 'download.json').write_text(json.dumps(result, indent=2))
    return result

def convert_asset(entry):
    import numpy as np
    import trimesh
    from PIL import Image
    from pxr import Usd, UsdGeom, UsdShade
    ident = entry['id']
    download = download_asset(entry)
    folder = CACHE / 'source' / ident
    target = CACHE / 'prepared' / ident
    target.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.Open(str(folder / download['root']))
    units = UsdGeom.GetStageMetersPerUnit(stage)
    axis = str(UsdGeom.GetStageUpAxis(stage))
    xf = UsdGeom.XformCache()
    meshes = []
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh): continue
        usd_mesh = UsdGeom.Mesh(prim)
        if UsdGeom.Imageable(prim).ComputeVisibility() == 'invisible': continue
        if UsdGeom.Imageable(prim).ComputePurpose() in ('guide', 'proxy'): continue
        points = np.asarray(usd_mesh.GetPointsAttr().Get(), dtype=float)
        counts = np.asarray(usd_mesh.GetFaceVertexCountsAttr().Get(), dtype=int)
        indices = np.asarray(usd_mesh.GetFaceVertexIndicesAttr().Get(), dtype=int)
        if not len(points) or not len(indices): continue
        matrix = np.asarray(xf.GetLocalToWorldTransform(prim))
        points = (np.column_stack((points, np.ones(len(points)))) @ matrix)[:, :3] * units
        if axis == 'Y': points = points[:, [0, 2, 1]] * [1, -1, 1]
        elif axis != 'Z': raise ValueError(f'Unsupported USD up axis: {axis}')
        # Expand corners to preserve face-varying UVs before triangulation.
        vertices = points[indices]
        faces = []
        offset = 0
        for count in counts:
            for i in range(1, int(count)-1): faces.append([offset, offset+i, offset+i+1])
            offset += int(count)
        faces = np.asarray(faces)
        if usd_mesh.GetOrientationAttr().Get() == 'leftHanded': faces = faces[:, ::-1]
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        uv = UsdGeom.PrimvarsAPI(prim).GetPrimvar('st')
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        texture = None
        if material:
            for child in Usd.PrimRange(material.GetPrim()):
                attr = child.GetAttribute('inputs:file')
                if attr:
                    path = attr.Get()
                    if path and any(tag in path.path.lower() for tag in ('_alb', 'basecolor', 'diffuse')):
                        resolved = Path(path.resolvedPath) if path.resolvedPath else None
                        if resolved and resolved.is_file(): texture = Image.open(resolved).convert('RGB'); break
        if uv and uv.HasValue():
            coords = np.asarray(uv.ComputeFlattened(), dtype=float)
            mode = uv.GetInterpolation()
            if mode in ('vertex', 'varying'): coords = coords[indices]
            elif mode == 'uniform': coords = np.repeat(coords, counts, axis=0)
            elif mode == 'constant': coords = np.repeat(coords[:1], len(indices), axis=0)
            if len(coords) == len(vertices):
                mesh.visual = trimesh.visual.TextureVisuals(uv=coords, material=trimesh.visual.material.PBRMaterial(
                    baseColorTexture=texture, baseColorFactor=[230,230,230,255], metallicFactor=0, roughnessFactor=.55))
        meshes.append(mesh)
    if not meshes: raise ValueError('No visible mesh geometry was found in the composed USD.')
    bounds = np.array([np.min([m.bounds[0] for m in meshes], axis=0), np.max([m.bounds[1] for m in meshes], axis=0)])
    center = bounds.mean(axis=0)
    for mesh in meshes: mesh.apply_translation(-center)
    scene = trimesh.Scene(meshes)
    scene.export(str(target / 'model.glb'))
    combined = trimesh.util.concatenate(meshes)
    combined.export(str(target / 'visual.obj'), include_texture=False)
    thumbnails = sorted((folder / 'simready_usd' / '.thumbs' / '256x256').glob('*_light.png'))
    if not thumbnails: thumbnails = sorted((folder / 'simready_usd' / '.thumbs' / '256x256').glob('*.png'))
    if thumbnails:
        Image.open(thumbnails[0]).convert('RGB').resize((384,384)).save(target / 'preview.jpg', quality=85)
    result = {**entry, 'dimensions': (bounds[1]-bounds[0]).tolist(), 'meshCount': len(meshes),
              'triangles': sum(len(m.faces) for m in meshes), 'units': 'm', 'upAxis': 'Z',
              'sourceMetersPerUnit': units, 'sourceUpAxis': axis, 'sourceCenterMeters': center.tolist(),
              'sourceRoot': download['root'], 'geometrySha256': hashlib.sha256((target/'model.glb').read_bytes()).hexdigest(),
              'conversion': 'Visible USD mesh geometry and base-color textures; centered, metres, Z-up. Physics/material extensions are not transferred to GLB.',
              'physicsStatus': 'not_validated'}
    (target / 'metadata.json').write_text(json.dumps(result, indent=2))
    return result

def prepared_assets():
    return [json.loads(p.read_text()) for p in sorted((CACHE/'prepared').glob('*/metadata.json'))]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--ids', nargs='+')
    parser.add_argument('--refresh', action='store_true')
    parser.add_argument('--office', action='store_true', help='prepare the study/office set')
    parser.add_argument('--max-mb', type=int, default=100, help='per-object download cap (default 100 MB)')
    args = parser.parse_args()
    global MAX_BYTES
    MAX_BYTES = args.max_mb * 1024 * 1024
    data = index_catalogue() if args.refresh or not (CACHE/'catalogue.json').exists() else catalogue()
    print(f'Indexed {len(data["assets"])} kitchen/dishware/office assets at {data["revision"]}', flush=True)
    if args.prepare or args.ids or args.office:
        for ident in args.ids or (OFFICE_IDS if args.office else DEFAULT_IDS):
            try: result = convert_asset(asset_entry(ident))
            except Exception as exc: print(f'{ident}: skipped ({exc})', flush=True); continue
            print(f'{ident}: {result["triangles"]} triangles, {result["dimensions"]} m', flush=True)

if __name__ == '__main__': main()
