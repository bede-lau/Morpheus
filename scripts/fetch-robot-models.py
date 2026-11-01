"""Fetch a MuJoCo Menagerie robot (MJCF + referenced assets) at a pinned commit."""
import sys, re, urllib.request, pathlib, xml.etree.ElementTree as ET
COMMIT="b0655057a9df2abe39ea7fc989914ba6f248301c"
RAW=f"https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{COMMIT}/"
robot, xml, out = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
def get(rel):
    dest=out/rel
    if dest.exists(): return dest
    dest.parent.mkdir(parents=True,exist_ok=True)
    print("fetch",rel,flush=True)
    with urllib.request.urlopen(urllib.request.Request(RAW+robot+"/"+rel,headers={"User-Agent":"morpheus/1"}),timeout=60) as r: dest.write_bytes(r.read())
    return dest
def walk(rel):
    root=ET.parse(get(rel)).getroot()
    comp=root.find("compiler"); assetdir=(comp.get("assetdir","") if comp is not None else "").rstrip("/"); meshdir=(comp.get("meshdir",assetdir) if comp is not None else "").rstrip("/"); texdir=(comp.get("texturedir",assetdir) if comp is not None else meshdir).rstrip("/")
    for inc in root.iter("include"): walk(inc.get("file"))
    for m in root.iter("mesh"):
        f=m.get("file");
        if f:
            rel=(meshdir+"/" if meshdir and not f.startswith("/") else "")+f; get(rel)
            if rel.lower().endswith(".obj"):
                try: get(rel[:-4]+".mtl")
                except Exception: pass
    for t in root.iter("texture"):
        f=t.get("file")
        if f: get((texdir+"/" if texdir and not f.startswith("/") else "")+f)
    for h in root.iter("hfield"):
        f=h.get("file")
        if f: get(f)
walk(xml); get("LICENSE")
print("done",robot)
