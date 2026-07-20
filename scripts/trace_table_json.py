#!/usr/bin/env python
"""Run every table-building script under an open()/glob tracer and record exactly which
result JSONs each reads. Emits logs/table_json_manifest.txt (one relative path per line).
This is the authoritative 'what data does the paper's tables need' list."""
import builtins, glob, os, runpy, sys, io

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

opened = set()
_open = builtins.open
def traced_open(f, *a, **k):
    if isinstance(f, str) and f.endswith(".json"):
        opened.add(os.path.relpath(os.path.abspath(f), ROOT))
    return _open(f, *a, **k)
builtins.open = traced_open

# tables to build (each writes paper/*.tex, reading results/cv5/**.json)
SCRIPTS = ["build_cv5_tex.py", "build_injection_table.py",
           "build_posf1_table.py", "build_pm_depth_tables.py"]
for s in SCRIPTS:
    path = os.path.join(ROOT, "scripts", s)
    print(f"[trace] running {s}")
    try:
        runpy.run_path(path, run_name="__main__")
    except SystemExit:
        pass
    except Exception as e:
        print(f"  [warn] {s} raised {type(e).__name__}: {e}")

builtins.open = _open
manifest = sorted(p for p in opened if p.endswith(".json"))
os.makedirs("logs", exist_ok=True)
with _open("logs/table_json_manifest.txt", "w") as fh:
    fh.write("\n".join(manifest) + "\n")
total = 0
for p in manifest:
    try: total += os.path.getsize(os.path.join(ROOT, p))
    except OSError: pass
print(f"[trace] {len(manifest)} JSON files needed by tables, {total/1024:.0f} KB total")
