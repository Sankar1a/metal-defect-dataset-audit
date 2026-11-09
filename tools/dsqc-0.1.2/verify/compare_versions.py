"""Scan the same synthetic datasets with dsqc 0.1.1 and 0.1.2 and compare every output that 0.1.1 writes.

usage: python verify/compare_versions.py <dsqc-0.1.1 dir> <dsqc-0.1.2 dir> <work dir (new, empty)>
Each version runs in its own interpreter process (python -I ... dsqc_run.py), so the two packages never mix.
"""
import csv, json, os, subprocess, sys

OLD, NEW, WORK = map(os.path.abspath, sys.argv[1:4])
sys.path.insert(0, os.path.join(NEW, "tests"))
import make_synthetic as ms   # noqa: E402
import yaml                   # noqa: E402

os.makedirs(WORK, exist_ok=True)
cases = {}
for name, fn in (("yolo", ms.make_yolo), ("gc10like", ms.make_gc10_like), ("coco", ms.make_coco),
                 ("jitter", ms.make_jitter), ("voclists", ms.make_voc_lists)):
    root = os.path.join(WORK, "data", name)
    spec = fn(root)
    if isinstance(spec, dict) and "sources" in spec:
        sp = os.path.join(WORK, f"{name}.yaml")
        with open(sp, "w", encoding="utf-8") as f:
            yaml.safe_dump(spec, f, allow_unicode=True)
        cases[name] = ["--spec", sp, root]
    else:
        cases[name] = [root]

def run(tool, args, out):
    r = subprocess.run([sys.executable, "-I", os.path.join(tool, "dsqc_run.py"), "scan", *args, "-o", out,
                        "--workers", "1"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr

sys.path.insert(0, os.path.join(NEW, "verify"))
from compare_outputs import compare   # noqa: E402

total = 0
for name, args in cases.items():
    o_old, o_new = os.path.join(WORK, "out_old", name), os.path.join(WORK, "out_new", name)
    run(OLD, args, o_old)
    run(NEW, args, o_new)
    print(f"== {name}")
    total += compare(o_old, o_new, out=lambda m: print("   " + m.replace(chr(10), chr(10) + "   ")))
print("TOTAL differences:", total)
