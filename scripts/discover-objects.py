"""Automatic object discovery for the SimReady replacement stage.

Two phases, so that the "which objects are in this view" reasoning is done by
the Codex CLI agents (Astra and friends) rather than by an API model call. The
OpenAI API key is used only for the Decisions matching that the object service
already performs.

    python3 scripts/discover-objects.py render --views 16
        Renders a fan of camera views of the scan to output/discovery/view*.jpg
        and writes output/discovery/views.json (image size, view index).
        A Codex CLI agent (or several, one per view) then looks at the images
        and writes output/discovery/detections.json:
        {"detections":[{"view":3,"label":"bowl","x":512,"y":388,"confidence":0.8}, ...]}
        with labels from plate|bowl|mug|cup|glass|spoon and pixel centres.

    python3 scripts/discover-objects.py detect [--parallel 4] [--backend auto|api|codex]
        Detects objects per view. With OPENAI_TERRA_API_KEY in .env (backend
        "api") it calls the Responses API with gpt-6.1-terra directly, a few
        requests at a time, no subagents needed. Otherwise (backend "codex") it
        fans out one `codex exec` Astra subagent per view through the Codex
        CLI's ChatGPT login. The API path is pinned to terra-class models: it refuses
        any other model, so the key is never used for astra/sol/luna-class
        models. with the image attached, and merges their
        answers into output/discovery/detections.json. Equivalent to the manual
        step above.

    python3 scripts/discover-objects.py apply [output/discovery/detections.json]
        Re-renders the same views, raycasts every detection into the splat
        through the object workspace, de-duplicates the 3D hits, then drives the
        existing identify (Decisions) → fit → replace → save flow for each one.

Run with the system Python that has Playwright + Chrome, with the Vite dev
server (5174) and the object service (8001) running.

Honest limits: only the prepared catalogue can be matched (plates, bowls, mugs,
cups, spoons); raycasts on blurry splats can land on walls, so hits outside
0.35–1.6 m height or further than 0.6 m behind a counter edge are rejected.
"""
from __future__ import annotations
import argparse, base64, json, re, subprocess, sys, time, urllib.request
OBJECTS_API = 'http://127.0.0.1:8001/api/objects/scene'


def saved_scene():
    try:
        with urllib.request.urlopen(OBJECTS_API, timeout=10) as r: return json.load(r)
    except Exception: return None


def merge_saved(prior, current):
    """Keep prior replacements that the new save did not re-place (within 0.18 m), then POST the union."""
    if not prior or not prior.get('replacements'): return current
    kept = []
    for item in prior['replacements']:
        if not any(sum((a-b)**2 for a, b in zip(item['position'], c['position'])) ** .5 < 0.18 for c in current['replacements']): kept.append(item)
    if not kept: return current
    body = {k: v for k, v in current.items() if k not in ('revision',)}
    body['replacements'] = current['replacements'] + kept
    body['baseRevision'] = current['revision']
    for item in body['replacements']: item.pop('asset', None)
    req = urllib.request.Request(OBJECTS_API, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=30) as r: return json.load(r)
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output' / 'discovery'
URL = 'http://127.0.0.1:5174/?view=objects'
SIZES = {'plate': (0.30, 0.30, 0.06, -0.02), 'bowl': (0.24, 0.24, 0.12, -0.03), 'mug': (0.12, 0.12, 0.11, -0.03),
         'cup': (0.10, 0.10, 0.11, -0.03), 'glass': (0.09, 0.09, 0.12, -0.03), 'spoon': (0.18, 0.06, 0.03, -0.01)}
SCENE = 'kitchen'
OBJECTS = 'movable tabletop object a home robot could pick up'
IGNORE = 'plants, taps, appliances, furniture, wall decor'
Z_RANGE = (0.35, 1.6)
SAVE = True
VIEW_CROPS = False
CROP_PX = {}
CAMERA = 'Inside'
# --room office: the study/office scan (sparkjs.dev fireplace sample). Its scale is unconfirmed, so no height filter,
# and nothing is saved into the kitchen's simulation scene; Luna's decisions are recorded in the report instead.
ROOMS = {'office': {'out': ROOT / 'output' / 'discovery' / 'office', 'url': 'http://127.0.0.1:5174/?view=objects&scan=office',
                    'scene': 'study/office', 'objects': 'object or piece of furniture a home robot might move, fetch or avoid',
                    'ignore': 'walls, windows, the fireplace itself, floor, ceiling, rugs',
                    'sizes': {'book': (0.25, 0.20, 0.06, -0.02), 'armchair': (0.9, 0.9, 1.0, -0.45), 'chair': (0.6, 0.6, 0.9, -0.4),
                              'bin': (0.3, 0.3, 0.4, -0.18), 'basket': (0.4, 0.3, 0.3, -0.13), 'candle': (0.08, 0.08, 0.15, -0.06),
                              'vase': (0.15, 0.15, 0.3, -0.13), 'clock': (0.15, 0.08, 0.15, -0.06), 'mug': (0.12, 0.12, 0.11, -0.03),
                              'cup': (0.10, 0.10, 0.11, -0.03), 'bowl': (0.24, 0.24, 0.12, -0.03), 'plate': (0.30, 0.30, 0.06, -0.02)},
                    'z': None, 'save': False, 'camera': 'Room view', 'view_crops': True,
                    # Crop size in Terra-view pixels per label (views are ~850x874 overviews of the whole room).
                    'crop_px': {'armchair': 260, 'chair': 220, 'bin': 130, 'basket': 150, 'book': 110, 'candle': 90, 'vase': 110,
                                'clock': 100, 'mug': 90, 'cup': 90, 'bowl': 110, 'plate': 120}}}


def configure(room):
    global OUT, URL, SIZES, SCENE, OBJECTS, IGNORE, Z_RANGE, SAVE, CAMERA, VIEW_CROPS, CROP_PX
    if room == 'kitchen': return
    c = ROOMS[room]
    OUT, URL, SIZES, SCENE, OBJECTS, IGNORE, Z_RANGE, SAVE = c['out'], c['url'], c['sizes'], c['scene'], c['objects'], c['ignore'], c['z'], c['save']
    CAMERA = c.get('camera', CAMERA)
    VIEW_CROPS, CROP_PX = c.get('view_crops', False), c.get('crop_px', {})
def view_pose(page, box, v, yaw_steps):
    """Deterministic orbit drags so render and apply see identical views."""
    cx, cy = box['x'] + box['width'] / 2, box['y'] + box['height'] / 2
    if v == 0: return
    dx = -int(box['width'] * 1.0 / yaw_steps) if v % 2 == 0 else 0
    dy = 70 if v % 2 == 1 else -70
    page.mouse.move(cx, cy); page.mouse.down(); page.mouse.move(cx + dx, cy + dy, steps=10); page.mouse.up(); time.sleep(1.2)


def status(page):
    try: return page.inner_text('.object-status')
    except Exception: return '?'


def open_workspace(p):
    browser = p.chromium.launch(channel='chrome', args=['--ignore-gpu-blocklist', '--enable-gpu-rasterization', '--use-angle=metal'])
    page = browser.new_page(viewport={'width': 1440, 'height': 940})
    page.goto(URL, wait_until='load')
    for _ in range(90):
        time.sleep(1)
        if 'Select an object in the scan to begin' in status(page) and page.locator('.object-viewport').bounding_box(): break
    else: sys.exit('Scan did not load')
    box = page.locator('.object-viewport').bounding_box()
    page.get_by_text(CAMERA, exact=True).click(); time.sleep(2)
    return browser, page, box


def render(args):
    from playwright.sync_api import sync_playwright
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser, page, box = open_workspace(p)
        yaw_steps = max(1, args.views // 2)
        views = []
        for v in range(args.views):
            view_pose(page, box, v, yaw_steps)
            shot = OUT / f'view{v:02d}.jpg'
            page.locator('.object-viewport').screenshot(path=str(shot), type='jpeg', quality=85)
            views.append({'view': v, 'file': shot.name, 'width': int(box['width']), 'height': int(box['height'])})
            print('rendered', shot.name)
        browser.close()
    (OUT / 'views.json').write_text(json.dumps({'views': views, 'labels': list(SIZES),
        'instructions': f'For each view image list each {OBJECTS} ({", ".join(SIZES)}) with pixel centres; write detections.json as {{"detections":[{{"view","label","x","y","confidence"}}]}}.'}, indent=2))
    print(f'{len(views)} views in {OUT}. Have the Codex CLI agent write {OUT / "detections.json"}, then run apply.')


def detect_prompt(w, h):
    labels = list(SIZES)
    return (f'The attached image is a {w}x{h} render of a {SCENE} Gaussian splat. List every {OBJECTS}, using only these '
            f'labels: {", ".join(labels)}. Ignore {IGNORE}, and anything too blurry to identify. Reply with ONLY a JSON object, no '
            f'prose, no code fence: {{"detections":[{{"label":"{labels[0]}","x":<pixel x of centre>,"y":<pixel y of centre>,'
            '"confidence":<0-1>}]}')


def detect_view(view, model):
    """One Codex CLI subagent per view. Returns (view index, detections, note)."""
    out = OUT / f'detections.view{view["view"]:02d}.json'
    cmd = ['codex', 'exec', '--sandbox', 'read-only', '--skip-git-repo-check', '-m', model,
           '-i', str(OUT / view['file']), '--output-last-message', str(out),
           detect_prompt(view['width'], view['height'])]
    try:
        # stdin must be closed: codex exec otherwise waits on a piped stdin forever.
        subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=200, stdin=subprocess.DEVNULL)
        text = out.read_text().strip() if out.exists() else ''
        text = text[text.find('{'):text.rfind('}') + 1]
        items = json.loads(text)['detections']
        clean = [{'view': view['view'], 'label': d['label'], 'x': int(d['x']), 'y': int(d['y']), 'confidence': float(d.get('confidence', 1))}
                 for d in items if d.get('label') in SIZES and 0 <= int(d['x']) < view['width'] and 0 <= int(d['y']) < view['height']]
        return view['view'], clean, 'ok'
    except Exception as e:
        return view['view'], [], f'failed: {str(e).splitlines()[0] if str(e) else type(e).__name__}'


# Terra-class only: the key must never be used for sol/astra/luna-class models here.
API_MODEL = 'gpt-6.1-terra'
API_ALLOWED = {'gpt-6.1-terra', 'gpt-6-terra', 'gpt-5.6-terra'}
def api_schema():
    return {'type': 'object', 'additionalProperties': False, 'required': ['detections'], 'properties': {'detections': {
        'type': 'array', 'items': {'type': 'object', 'additionalProperties': False, 'required': ['label', 'x', 'y', 'confidence'],
        'properties': {'label': {'type': 'string', 'enum': list(SIZES)}, 'x': {'type': 'integer'}, 'y': {'type': 'integer'},
                       'confidence': {'type': 'number'}}}}}}


def terra_key():
    env = ROOT / '.env'
    if not env.exists(): return ''
    for line in env.read_text().splitlines():
        m = re.match(r'^\s*(?:export\s+)?OPENAI_TERRA_API_KEY\s*=\s*(.*?)\s*$', line)
        if m: return m[1].strip('"\'')
    return ''


def detect_view_api(view, key, model=API_MODEL):
    """One Responses API call per view (pinned to terra). Returns (view index, detections, note)."""
    if model not in API_ALLOWED: raise ValueError(f'API detection is restricted to {sorted(API_ALLOWED)}')
    out = OUT / f'detections.view{view["view"]:02d}.json'
    try:
        image = base64.b64encode((OUT / view['file']).read_bytes()).decode()
        payload = {'model': model, 'input': [{'role': 'user', 'content': [
            {'type': 'input_text', 'text': detect_prompt(view['width'], view['height'])},
            {'type': 'input_image', 'image_url': 'data:image/jpeg;base64,' + image}]}],
            'text': {'format': {'type': 'json_schema', 'name': 'detections', 'schema': api_schema(), 'strict': True}}}
        req = urllib.request.Request('https://api.openai.com/v1/responses', data=json.dumps(payload).encode(),
                                     headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=120) as r: resp = json.load(r)
        text = resp.get('output_text') or ''
        if not text:
            for item in resp.get('output', []):
                for part in item.get('content', []) or []:
                    if part.get('type') == 'output_text': text = part['text']
        out.write_text(text)
        items = json.loads(text)['detections']
        clean = [{'view': view['view'], 'label': d['label'], 'x': int(d['x']), 'y': int(d['y']), 'confidence': float(d.get('confidence', 1))}
                 for d in items if d.get('label') in SIZES and 0 <= int(d['x']) < view['width'] and 0 <= int(d['y']) < view['height']]
        return view['view'], clean, f'ok ({resp.get("model", model)})'
    except Exception as e:
        # Never echo the request (it carries the key header) or the raw body.
        return view['view'], [], f'failed: {type(e).__name__}: {str(e)[:120]}'


def detect(args):
    views = json.loads((OUT / 'views.json').read_text())['views']
    key = terra_key()
    backend = args.backend if args.backend != 'auto' else ('api' if key else 'codex')
    if backend == 'api' and not key: sys.exit('OPENAI_TERRA_API_KEY missing in .env')
    if backend == 'api':
        parallel = min(args.parallel, 4)
        print(f'{len(views)} views -> Responses API ({API_MODEL}), {parallel} requests at a time')
        worker = lambda view: detect_view_api(view, key)
    else:
        parallel = args.parallel
        print(f'fanning out {len(views)} Codex subagents ({args.model}), {parallel} at a time')
        worker = lambda view: detect_view(view, args.model)
    merged, notes = [], {}
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        for v, items, note in pool.map(worker, views):
            notes[v] = note; merged.extend(items)
            print(f'view {v}: {note}, {len(items)} detections', [(d["label"], d["x"], d["y"], round(d["confidence"], 2)) for d in items])
    source = f'Responses API ({API_MODEL}), one request per view' if backend == 'api' else f'codex exec ({args.model}), one subagent per view'
    (OUT / 'detections.json').write_text(json.dumps({'source': source, 'notes': notes, 'detections': merged}, indent=2))
    print(f'{len(merged)} detections written to {OUT / "detections.json"}')


def view_crop(view_file, x, y, label):
    """Square crop of a Terra view around a detection, as a data URL (Luna evidence when close-up splat captures are too blurry)."""
    import io
    from PIL import Image
    im = Image.open(OUT / view_file).convert('RGB')
    half = CROP_PX.get(label, 140) // 2
    box = (max(0, x - half), max(0, y - half), min(im.width, x + half), min(im.height, y + half))
    crop = im.crop(box).resize((384, 384))
    buf = io.BytesIO(); crop.save(buf, 'JPEG', quality=88)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()


def match_crops(images):
    req = urllib.request.Request(OBJECTS_API.replace('/scene', '/match'), data=json.dumps({'images': images[:4]}).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=240) as r: return json.load(r)


def apply(args):
    from playwright.sync_api import sync_playwright
    detections = json.loads(Path(args.detections).read_text())['detections']
    by_view = {}
    for d in detections: by_view.setdefault(int(d['view']), []).append(d)
    views = json.loads((OUT / 'views.json').read_text())['views']
    yaw_steps = max(1, len(views) // 2)
    prior = saved_scene() if SAVE else None
    found = [{'label': 'saved', 'position': r['position'], 'result': 'already-saved', 'asset': r['assetId']} for r in (prior or {}).get('replacements', [])]
    print(f'{len(found)} objects already saved; new hits within 0.18 m of them are skipped')
    report = {'objects': [], 'source': 'detections.json', 'priorRevision': (prior or {}).get('revision'), 'saved': SAVE}
    with sync_playwright() as p:
        browser, page, box = open_workspace(p)
        last_match = {}
        def on_response(resp):
            if resp.url.endswith('/api/objects/match') and resp.request.method == 'POST':
                try: last_match['body'] = resp.json()
                except Exception: pass
        page.on('response', on_response)
        for v in range(len(views)):
            view_pose(page, box, v, yaw_steps)
            for det in by_view.get(v, []):
                if det.get('confidence', 1) < args.min_confidence or det['label'] not in SIZES: continue
                px, py = box['x'] + det['x'], box['y'] + det['y']
                # After a click that misses the splat the button stays in picking mode with a different label.
                page.get_by_role('button', name=re.compile('Select object in scan|Click an object in the scan')).click(); time.sleep(0.4)
                page.mouse.click(px, py); time.sleep(0.8)
                if 'Adjust the selection box' not in status(page): print(f'view {v} {det["label"]}: no splat hit'); continue
                try: pos = [float(page.get_by_label(f'Selection {a}').input_value()) for a in 'XYZ']
                except Exception: continue
                if Z_RANGE and not Z_RANGE[0] <= pos[2] <= Z_RANGE[1]: print(f'view {v} {det["label"]} hit at z={pos[2]:.2f}: rejected (not a surface height)'); continue
                near = max(0.18, max(SIZES[det['label']][:2]) / 2)  # big furniture needs a bigger duplicate radius than cups
                same = next((f for f in found if ((pos[0]-f['position'][0])**2 + (pos[1]-f['position'][1])**2) ** .5 < near), None)
                if same:
                    if VIEW_CROPS and same.get('label') == det['label'] and len(same.get('crops', [])) < 3:
                        same.setdefault('crops', []).append(view_crop(views[v]['file'], det['x'], det['y'], det['label']))
                    print(f'view {v} {det["label"]} at {[round(x, 2) for x in pos]}: duplicate'); continue
                w, d, h, dz = SIZES[det['label']]
                page.get_by_label('Width').fill(str(w)); page.get_by_label('Depth').fill(str(d)); page.get_by_label('Height').fill(str(h))
                page.get_by_label('Selection Z').fill(f'{pos[2] + dz:.3f}'); time.sleep(0.4)
                entry = {'label': det['label'], 'view': v, 'pixel': [det['x'], det['y']], 'position': pos, 'confidence': det.get('confidence')}
                if args.dry_run: found.append(entry); print('   dry-run located', entry); continue
                if VIEW_CROPS:
                    # Luna is called after the sweep, once every view of this object has contributed a crop.
                    entry['crops'] = [view_crop(views[v]['file'], det['x'], det['y'], det['label'])]; entry['result'] = 'pending'; found.append(entry); continue
                # Two evidence views for Decisions: capture, nudge the orbit, capture, nudge back.
                page.get_by_text('Capture object view').click(); time.sleep(1.2)
                cx, cy = box['x'] + box['width'] / 2, box['y'] + box['height'] / 2
                page.mouse.move(cx, cy); page.mouse.down(); page.mouse.move(cx - 60, cy - 20, steps=8); page.mouse.up(); time.sleep(1)
                page.get_by_text('Capture object view').click(); time.sleep(1.2)
                page.mouse.move(cx - 60, cy - 20); page.mouse.down(); page.mouse.move(cx, cy, steps=8); page.mouse.up(); time.sleep(1)
                last_match.pop('body', None)
                page.get_by_text('Identify with Decisions').click()
                for _ in range(120):
                    time.sleep(1)
                    if 'Comparing' not in page.inner_text('body'): break
                cards = page.locator('.object-asset-grid .object-asset')
                entry['decisions'] = status(page)
                if last_match.get('body'): entry['decision'] = last_match['body']
                if cards.count() == 0 or 'No supported match' in status(page):
                    entry['result'] = 'unknown'; found.append(entry)
                    print(f'view {v} {det["label"]} at {[round(x, 2) for x in pos]}: Decisions found no catalogue match'); continue
                top = cards.nth(0).inner_text().replace('\n', ' | ')
                m = re.search(r'([0-9.]+)/3', top)
                score = float(m[1]) if m else 0.0
                if score < args.min_score:
                    entry['result'] = 'weak'; entry['asset'] = top; found.append(entry)
                    print(f'view {v} {det["label"]} at {[round(x, 2) for x in pos]}: best Decisions score {score:.2f}/3 below {args.min_score}; left as scan'); continue
                cards.nth(0).click()
                for _ in range(30):
                    time.sleep(1)
                    if 'Preview loaded' in status(page): break
                page.get_by_text('Replace selected splats').click(); time.sleep(1.2)
                entry['result'] = 'replaced'; entry['asset'] = top; found.append(entry)
                print(f'view {v} {det["label"]} at {[round(x, 2) for x in pos]}: replaced with {top}')
        for entry in [f for f in found if f.get('result') == 'pending']:
            try: result = match_crops(entry['crops'])
            except Exception as e: entry['result'] = 'error'; entry['decisions'] = str(e)[:160]; print(f'{entry["label"]}: Decisions failed {entry["decisions"]}'); continue
            entry['decision'], entry['decisions'], entry['evidenceViews'] = result, result.get('message', ''), len(entry['crops'])
            top = (result.get('candidates') or [None])[0]
            entry['result'] = 'unknown' if not top else ('matched' if top['score'] >= args.min_score else 'weak')
            if top: entry['asset'] = top['asset']['id']
            print(f'{entry["label"]} at {[round(x, 2) for x in entry["position"]]} ({len(entry["crops"])} views): luna {result.get("category")}'
                  + (f' -> {top["asset"]["id"]} {top["score"]:.2f}/3' if top else ' (no catalogue match)'))
        for entry in found: entry.pop('crops', None)
        replaced = [f for f in found if f.get('result') == 'replaced']
        if replaced and not args.dry_run and SAVE:
            page.get_by_text('Save to simulation workspace').click()
            for _ in range(30):
                time.sleep(1)
                if 'Saved' in status(page) or page.locator('.object-error').count(): break
            print('save ->', status(page))
            try: print('error:', page.inner_text('.object-error'))
            except Exception: pass
            try:
                merged = merge_saved(prior, saved_scene())
                print(f'saved scene revision {merged.get("revision")} holds {len(merged.get("replacements", []))} objects (earlier ones kept)')
            except Exception as e: print('merge with earlier saved objects failed:', str(e)[:160])
        page.get_by_text('Room view', exact=True).click(); time.sleep(2)
        page.screenshot(path=str(OUT / 'final.png'))
        browser.close()
    report['objects'] = found
    (OUT / 'report.json').write_text(json.dumps(report, indent=2))
    print(f'{len(replaced) if not args.dry_run else 0} replaced / {len([f for f in found if f.get("result") != "already-saved"])} located; report at {OUT / "report.json"}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--room', choices=['kitchen', *ROOMS], default='kitchen', help='which scan to run on (default kitchen)')
    sub = ap.add_subparsers(dest='phase', required=True)
    r = sub.add_parser('render'); r.add_argument('--views', type=int, default=16)
    d = sub.add_parser('detect'); d.add_argument('--parallel', type=int, default=4); d.add_argument('--model', default='gpt-6-astra', help='codex backend model')
    d.add_argument('--backend', choices=['auto', 'api', 'codex'], default='auto')
    a = sub.add_parser('apply'); a.add_argument('detections', nargs='?', default=None)
    a.add_argument('--min-confidence', type=float, default=0.5); a.add_argument('--min-score', type=float, default=0.5, help='minimum Decisions score (0-3) to accept a match')
    a.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    configure(args.room)
    if args.phase == 'apply' and not args.detections: args.detections = str(OUT / 'detections.json')
    {'render': render, 'detect': detect, 'apply': apply}[args.phase](args)


if __name__ == '__main__': main()
