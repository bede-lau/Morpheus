"""Generate the README's light and dark architecture SVGs.

Run: python3 docs/diagrams/render_architecture.py
Requires only Python's standard library. The diagram describes the local demo,
its two HTTP services, shared files, and the offline object-discovery path.
"""
from html import escape
from pathlib import Path

THEMES = {
    'light': dict(bg='#ffffff', ink='#1f2328', muted='#59636e', edge='#8c959f',
                  border='#d0d7de', group='#f6f8fa', panel='#ffffff', header='#24292f', onheader='#ffffff'),
    'dark': dict(bg='#0d1117', ink='#e6edf3', muted='#9da7b3', edge='#8b949e',
                 border='#30363d', group='#161b22', panel='#0d1117', header='#252c35', onheader='#f0f6fc'),
}


def render(t):
    out = [f'''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="1080" viewBox="0 0 1200 1080" role="img" aria-labelledby="title desc">
<title id="title">Morpheus system architecture</title>
<desc id="desc">A React, Three.js and Spark browser client calls two local Python HTTP services. The environment service validates geometry with MuJoCo and plans robot tasks. The object service matches objects through the OpenAI Decisions API and serves prepared NVIDIA SimReady assets. Both services read and write local scene, rollout and replacement files. Offline discovery renders room views, detects objects using Terra vision or Codex, matches assets, and saves its results. Solid arrows show calls; dashed arrows show file access. Robot rollouts are planned or scripted, not learned policies.</desc>
<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M1 1 L9 5 L1 9" fill="none" stroke="{t['edge']}" stroke-width="1.5"/></marker></defs>
<style>text{{font-family:Helvetica Neue,Helvetica,Arial,sans-serif;fill:{t['ink']};font-size:18px}}.heading{{font-size:23px;font-weight:600}}.title{{font-size:24px;font-weight:500}}.meta{{fill:{t['muted']};font-size:17px}}.code{{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:15px;fill:{t['muted']}}}.label{{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:14px;letter-spacing:1px;fill:{t['muted']}}}.inverse{{fill:{t['onheader']}}}</style>
<rect width="1200" height="1080" fill="{t['bg']}"/>''']

    def text(x, y, s, cls='', anchor='start'):
        out.append(f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{escape(s)}</text>')

    def rect(x, y, w, h, fill='panel', dash=False):
        attr = ' stroke-dasharray="5 5"' if dash else ''
        out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{t[fill]}" stroke="{t["border"]}" stroke-width="1.3"{attr}/>')

    def edge(path, dashed=False, both=False):
        attrs = ' stroke-dasharray="5 5"' if dashed else ''
        if both:
            attrs += ' marker-start="url(#arrow)"'
        out.append(f'<path d="{path}" fill="none" stroke="{t["edge"]}" stroke-width="1.5" stroke-linejoin="round" marker-end="url(#arrow)"{attrs}/>')

    def rule(x1, y, x2):
        out.append(f'<path d="M{x1} {y} H{x2}" stroke="{t["border"]}"/>')

    text(40, 40, 'Morpheus / system architecture', 'title')
    text(1160, 40, 'LOCAL DEMO', 'label', 'end')
    text(40, 84, 'INPUTS', 'label')
    text(128, 84, 'Room capture (SPZ / PLY + GLB) · household task · robot model', 'meta')

    rect(40, 110, 1120, 110, 'group')
    text(64, 147, 'Browser client', 'heading')
    text(1136, 147, 'React · Three.js · World Labs Spark', 'code', 'end')
    text(64, 181, 'Room viewer · object replacement · robot POV · model comparison')
    text(64, 205, 'Demo / workspace / object inspector / collider builder', 'meta')

    # Bidirectional request/response channels to the two independent servers.
    edge('M310 220 V316', both=True)
    edge('M890 220 V316', both=True)
    text(332, 264, '/api/*', 'code')
    text(332, 288, 'HTTP / JSON', 'meta')
    text(912, 264, '/api/objects/*', 'code')
    text(912, 288, 'HTTP / JSON', 'meta')

    # Service boundaries contain their in-process dependencies.
    rect(40, 316, 540, 350, 'group')
    rect(620, 316, 540, 350, 'group')
    rect(40, 316, 540, 100, 'header')
    rect(620, 316, 540, 100, 'header')
    text(64, 352, 'Environment service', 'heading inverse')
    text(556, 352, ':8000', 'code inverse', 'end')
    text(64, 385, 'pipeline.server', 'code inverse')
    text(644, 352, 'Object service', 'heading inverse')
    text(1136, 352, ':8001', 'code inverse', 'end')
    text(644, 385, 'pipeline.object_server', 'code inverse')

    for x in (178, 442, 758, 1022):
        edge(f'M{x} 416 V460')
    for x in (60, 324, 640, 904):
        rect(x, 460, 236, 150)

    text(78, 492, 'Geometry', 'heading')
    text(78, 522, 'MJCF + MuJoCo', 'code')
    text(78, 551, 'Compile · contact probe', 'meta')
    text(78, 579, 'Validate + export', 'meta')
    text(342, 492, 'Task planner', 'heading')
    text(342, 522, 'Landmarks + A*', 'code')
    text(342, 551, 'Scripted robot actions', 'meta')
    text(342, 579, 'Decisions feasibility', 'meta')
    text(658, 492, 'Object matching', 'heading')
    text(658, 522, 'OpenAI Decisions API', 'code')
    text(658, 551, 'Classify + rank assets', 'meta')
    text(658, 579, 'gpt-6-luna', 'code')
    text(922, 492, 'Asset catalogue', 'heading')
    text(922, 522, 'NVIDIA SimReady', 'code')
    text(922, 551, 'Prepared GLB meshes', 'meta')
    text(922, 579, 'Static collision export', 'meta')

    text(64, 643, 'Scenes · tasks · replays · validation bundles', 'meta')
    text(644, 643, 'Catalogue · matching · saved replacements', 'meta')

    edge('M310 666 V750', dashed=True, both=True)
    edge('M890 666 V750', dashed=True, both=True)
    text(332, 712, 'read / write', 'code')
    text(912, 712, 'read / write', 'code')

    rect(40, 750, 1120, 124, 'group')
    text(64, 779, 'LOCAL FILES', 'label')
    text(1136, 779, 'Committed demo data + generated outputs', 'meta', 'end')
    for x, title, code in [(64, 'Environments', 'output/<room>/'),
                           (340, 'Robot rollouts', 'rollouts/<room>/'),
                           (616, 'Replacements + meshes', 'output/simready/'),
                           (892, 'Vision detections', 'public/discovery/')]:
        text(x, 817, title)
        text(x, 846, code, 'code')

    edge('M600 944 V874', dashed=True)
    text(622, 915, 'save detections + matched scene', 'code')
    rect(40, 944, 1120, 96, dash=True)
    text(64, 978, 'Offline object discovery', 'heading')
    text(1136, 978, 'scripts/discover-objects.py', 'code', 'end')
    text(64, 1012, 'Orbit renders → Terra 6.1 vision or Codex → 3D deduplication → Decisions matching', 'meta')

    text(40, 1066, 'Solid arrows: calls     Dashed arrows: file access', 'code')
    text(1160, 1066, 'Robot rollouts: planned / scripted', 'code', 'end')
    out.append('</svg>\n')
    return '\n'.join(out)


if __name__ == '__main__':
    destination = Path(__file__).resolve().parents[1] / 'media'
    destination.mkdir(parents=True, exist_ok=True)
    for mode, theme in THEMES.items():
        name = 'architecture.svg' if mode == 'light' else 'architecture-dark.svg'
        (destination / name).write_text(render(theme))
