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
import argparse, json, re, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output' / 'discovery'
URL = 'http://127.0.0.1:5174/?view=objects'
SIZES = {'plate': (0.30, 0.30, 0.06, -0.02), 'bowl': (0.24, 0.24, 0.12, -0.03), 'mug': (0.12, 0.12, 0.11, -0.03),
         'cup': (0.10, 0.10, 0.11, -0.03), 'glass': (0.09, 0.09, 0.12, -0.03), 'spoon': (0.18, 0.06, 0.03, -0.01)}
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
        if 'Select an object in the scan to begin' in status(page): break
    else: sys.exit('Scan did not load')
    box = page.locator('.object-viewport').bounding_box()
    page.get_by_text('Inside', exact=True).click(); time.sleep(2)
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
        'instructions': 'For each view image list movable tabletop objects (plate, bowl, mug, cup, glass, spoon) with pixel centres; write detections.json as {"detections":[{"view","label","x","y","confidence"}]}.'}, indent=2))
    print(f'{len(views)} views in {OUT}. Have the Codex CLI agent write {OUT / "detections.json"}, then run apply.')


def apply(args):
    from playwright.sync_api import sync_playwright
    detections = json.loads(Path(args.detections).read_text())['detections']
    by_view = {}
    for d in detections: by_view.setdefault(int(d['view']), []).append(d)
    views = json.loads((OUT / 'views.json').read_text())['views']
    yaw_steps = max(1, len(views) // 2)
    found, report = [], {'objects': [], 'source': 'codex-cli detections'}
    with sync_playwright() as p:
        browser, page, box = open_workspace(p)
        for v in range(len(views)):
            view_pose(page, box, v, yaw_steps)
            for det in by_view.get(v, []):
                if det.get('confidence', 1) < args.min_confidence or det['label'] not in SIZES: continue
                px, py = box['x'] + det['x'], box['y'] + det['y']
                page.get_by_text('Select object in scan').click(); time.sleep(0.4)
                page.mouse.click(px, py); time.sleep(0.8)
                if 'Adjust the selection box' not in status(page): print(f'view {v} {det["label"]}: no splat hit'); continue
                try: pos = [float(page.get_by_label(f'Selection {a}').input_value()) for a in 'XYZ']
                except Exception: continue
                if not 0.35 <= pos[2] <= 1.6: print(f'view {v} {det["label"]} hit at z={pos[2]:.2f}: rejected (not a surface height)'); continue
                if any(((pos[0]-f['position'][0])**2 + (pos[1]-f['position'][1])**2) ** .5 < 0.18 for f in found):
                    print(f'view {v} {det["label"]} at {[round(x, 2) for x in pos]}: duplicate'); continue
                w, d, h, dz = SIZES[det['label']]
                page.get_by_label('Width').fill(str(w)); page.get_by_label('Depth').fill(str(d)); page.get_by_label('Height').fill(str(h))
                page.get_by_label('Selection Z').fill(f'{pos[2] + dz:.3f}'); time.sleep(0.4)
                entry = {'label': det['label'], 'view': v, 'pixel': [det['x'], det['y']], 'position': pos, 'confidence': det.get('confidence')}
                if args.dry_run: found.append(entry); print('   dry-run located', entry); continue
                # Two evidence views for Decisions: capture, nudge the orbit, capture, nudge back.
                page.get_by_text('Capture object view').click(); time.sleep(1.2)
                cx, cy = box['x'] + box['width'] / 2, box['y'] + box['height'] / 2
                page.mouse.move(cx, cy); page.mouse.down(); page.mouse.move(cx - 60, cy - 20, steps=8); page.mouse.up(); time.sleep(1)
                page.get_by_text('Capture object view').click(); time.sleep(1.2)
                page.mouse.move(cx - 60, cy - 20); page.mouse.down(); page.mouse.move(cx, cy, steps=8); page.mouse.up(); time.sleep(1)
                page.get_by_text('Identify with Decisions').click()
                for _ in range(120):
                    time.sleep(1)
                    if 'Comparing' not in page.inner_text('body'): break
                cards = page.locator('.object-asset-grid .object-asset')
                entry['decisions'] = status(page)
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
        replaced = [f for f in found if f.get('result') == 'replaced']
        if replaced and not args.dry_run:
            page.get_by_text('Save to simulation workspace').click()
            for _ in range(30):
                time.sleep(1)
                if 'Saved' in status(page) or page.locator('.object-error').count(): break
            print('save ->', status(page))
            try: print('error:', page.inner_text('.object-error'))
            except Exception: pass
        page.get_by_text('Room view', exact=True).click(); time.sleep(2)
        page.screenshot(path=str(OUT / 'final.png'))
        browser.close()
    report['objects'] = found
    (OUT / 'report.json').write_text(json.dumps(report, indent=2))
    print(f'{len(replaced) if not args.dry_run else 0} replaced / {len(found)} located; report at {OUT / "report.json"}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='phase', required=True)
    r = sub.add_parser('render'); r.add_argument('--views', type=int, default=16)
    a = sub.add_parser('apply'); a.add_argument('detections', nargs='?', default=str(OUT / 'detections.json'))
    a.add_argument('--min-confidence', type=float, default=0.5); a.add_argument('--min-score', type=float, default=0.5, help='minimum Decisions score (0-3) to accept a match')
    a.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    render(args) if args.phase == 'render' else apply(args)


if __name__ == '__main__': main()
