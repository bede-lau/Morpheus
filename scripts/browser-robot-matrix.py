"""Drive the running workspace (127.0.0.1:5174) through every robot and prerendered task, scrub to each key event and save contact sheets. Needs system Python playwright + Chrome; pass an output dir."""
import json, io, re
from playwright.sync_api import sync_playwright
from PIL import Image, ImageDraw
import sys; OUT=sys.argv[1]
ROBOTS=['g1','go2','stretch']; TASKS=['wipe the table','fill a glass of water','wipe the table and move the bowl to the sink']
KEY=['Wiping','Filling','Picked','Placed','cannot','out of reach','No collision','Task completed','failed']
SCRUB="""(t)=>{const el=document.querySelector('input[aria-label="Replay timeline"]');const set=Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set;set.call(el,String(t));el.dispatchEvent(new Event('input',{bubbles:true}));}"""
STATE="""()=>{const s=window.__robotScene;let meshes=0;s.robot.traverse(o=>{if(o.isMesh&&o.userData.shared)meshes++});const objs=[];s.viewer.scene.traverse(o=>{if(o.userData&&(o.userData.objectId||o.userData.assetId)&&o.parent&&!(o.parent.userData&&o.parent.userData.objectId)){const p=o.getWorldPosition(new o.position.constructor());objs.push([o.userData.assetId||o.userData.objectId,p.toArray().map(v=>+v.toFixed(2))])}});
 return {robot:s.replay.robot,src:s.replay.source,dur:s.replay.duration,verdict:s.replay.verdict?.status,events:s.replay.events,carried:s.replay.carried,meshes,pos:s.robot.position.toArray().map(v=>+v.toFixed(2)),yaw:+s.robot.rotation.z.toFixed(2),objs}}"""
with sync_playwright() as p:
    b=p.chromium.launch(channel='chrome',args=['--use-angle=metal','--ignore-gpu-blocklist'])
    pg=b.new_page(viewport={'width':1500,'height':950}); errs=[]
    pg.on('console',lambda m: errs.append(m.text[:200]) if m.type=='error' else None); pg.on('pageerror',lambda e: errs.append('PAGEERROR '+str(e)[:200]))
    pg.goto('http://127.0.0.1:5174/?view=workspace'); pg.wait_for_timeout(10000)
    report={}
    for r in ROBOTS:
        sheet=[]
        for task in TASKS:
            pg.locator('.task-list button').filter(has=pg.locator(f'span:text-is("{task}")')).first.click(); pg.wait_for_timeout(500)
            pg.click(f'input[name=robot][value={r}]'); pg.wait_for_timeout(4500)
            st=pg.evaluate(STATE)
            times=[(0,'start')]+[(e['time']+.6,e['label'][:24]) for e in st['events'] if any(k.lower() in e['label'].lower() for k in KEY)]
            if st['carried']: times.append((st['carried'][0]['pickTime']+2.5,'carrying'))
            times=sorted(set(times))[:6]; tiles=[]; checks=[]
            for t,label in times:
                pg.evaluate(SCRUB,t); pg.wait_for_timeout(400); pg.evaluate("()=>window.__robotScene.focus()"); pg.wait_for_timeout(700)
                cur=pg.evaluate(STATE); checks.append({'t':t,'pos':cur['pos'],'objs':cur['objs']})
                img=Image.open(io.BytesIO(pg.screenshot(type='jpeg',quality=80,clip={'x':236,'y':60,'width':970,'height':855}))).resize((485,428))
                d=ImageDraw.Draw(img); d.rectangle((0,0,485,16),fill=(0,0,0)); d.text((4,2),f"{r} · {task[:30]} · {t:.1f}s {label}",fill=(255,255,255)); tiles.append(img)
            report[f'{r} / {task}']={'robot':st['robot'],'src':st['src'],'verdict':st['verdict'],'meshes':st['meshes'],'dur':st['dur'],'checks':checks}
            row=Image.new('RGB',(485*6,428),(235,235,235)); [row.paste(im,(i*485,0)) for i,im in enumerate(tiles)]; sheet.append(row)
        full=Image.new('RGB',(485*6,428*3)); [full.paste(rw,(0,i*428)) for i,rw in enumerate(sheet)]; full.save(f'{OUT}/ui_{r}.jpg',quality=82)
    json.dump(report,open(f'{OUT}/ui_report.json','w'),indent=1)
    for k,v in report.items(): print(k,'|',v['robot'],v['src'],v['verdict'],'meshes',v['meshes'],'| positions',[c['pos'][:2] for c in v['checks']])
    print('console errors:',errs[:6])
    b.close()
