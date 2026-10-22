"""Command line.

  python -m dsqc scan --spec specs/gc10_raw.yaml -o out/gc10            # explicit layout
  python -m dsqc scan /data/my_yolo_dataset -o out/mine [--format yolo]  # standard layout
  python -m dsqc diff out/a out/b                                        # compare two scans' manifests

The output directory must not exist or be empty; nothing is ever written inside the dataset.
"""
import argparse
import csv
import datetime
import json
import os
import platform
import sys
import time

from . import __version__


def _pkg_versions():
    out = {}
    try:
        from importlib import metadata
    except ImportError:
        return out
    for p in ("numpy", "scipy", "pillow", "PyYAML"):
        try:
            out[p] = metadata.version(p)
        except Exception:
            out[p] = "?"
    return out


def cmd_scan(a):
    from .checks import DEFAULTS, run_all
    from .index.hashing import hash_dataset, sha256_file
    from .io.loader import load_auto, load_spec
    from .report.writers import write_outputs

    out = os.path.abspath(a.out)
    if os.path.exists(out) and os.listdir(out):
        sys.exit(f"STOP: output directory exists and is not empty: {out}")
    cfg = dict(DEFAULTS)
    if a.config:
        import yaml
        with open(a.config, encoding="utf-8") as f:
            cfg.update(yaml.safe_load(f) or {})
    t0 = time.time()
    started = datetime.datetime.now().isoformat(timespec="seconds")
    if a.spec:
        ds = load_spec(a.spec, a.root)
    elif a.root:
        names = a.names.split(",") if a.names else None
        ds = load_auto(a.root, a.format, names)
    else:
        sys.exit("give a dataset root or --spec")
    if out.startswith(ds.root.rstrip("/") + "/"):
        sys.exit("STOP: output directory must not be inside the dataset")
    print(f"[dsqc] {ds.name}: {len(ds.images)} images, sources {ds.sources}, primary={ds.primary}, split={ds.split_mode}")
    hash_dataset(ds, workers=a.workers, want_canon=not a.no_transform, cache_dir=a.cache_dir,
                 progress=lambda m: print(f"[dsqc] {m}"))
    ds.finalize()
    print(f"[dsqc] hashing done in {time.time() - t0:.1f}s; running checks")
    results = run_all(ds, cfg, skip=set(a.skip.split(",")) if a.skip else ())
    os.makedirs(out, exist_ok=True)
    thumbs = {}
    if a.thumbs > 0:
        from .report.thumbs import panel
        os.makedirs(os.path.join(out, "thumbs"), exist_ok=True)
        src = ds.primary
        for r in results:
            if r.name not in ("c02_dup_label_types", "c03_source_consistency", "c04a_transform_dup", "c06_box_sanity"):
                continue
            k = 0
            for f in sorted(r.findings, key=lambda f: ({"high": 0, "medium": 1, "low": 2}.get(f.severity, 3), f.code)):
                if k >= a.thumbs or not f.images or f.severity == "info":
                    continue
                rel = f"thumbs/{f.id}.jpg"
                if panel(ds, f.images, src, os.path.join(out, rel)):
                    thumbs[f.id] = rel
                    k += 1
    run_info = {
        "dsqc_version": __version__, "command": " ".join(sys.argv), "started": started,
        "finished": datetime.datetime.now().isoformat(timespec="seconds"),
        "seconds": round(time.time() - t0, 1), "python": sys.version.split()[0], "executable": sys.executable,
        "platform": platform.platform(), "packages": _pkg_versions(), "workers": a.workers,
        "rotation_flip_hashes": not a.no_transform, "config": cfg,
        "spec": os.path.abspath(a.spec) if a.spec else None,
        "spec_sha256": sha256_file(a.spec) if a.spec else None,
        "thumbnails": len(thumbs),
    }
    summary = write_outputs(out, ds, results, run_info, thumbs)
    print(f"[dsqc] wrote {out}  ({run_info['seconds']}s)")
    for name, c in summary["checks"].items():
        keys = list(c["stats"].items())[:12]
        print(f"  {name}: " + ", ".join(f"{k}={v}" for k, v in keys))
    print(f"  findings: {summary['findings_by_severity']}")


def _read_manifest(d):
    p = os.path.join(d, "manifest.csv") if os.path.isdir(d) else d
    with open(p, newline="", encoding="utf-8") as f:
        return {r["path"]: r for r in csv.DictReader(f)}


def cmd_diff(a):
    A, B = _read_manifest(a.a), _read_manifest(a.b)
    added = sorted(set(B) - set(A))
    removed = sorted(set(A) - set(B))
    changed = sorted(p for p in set(A) & set(B) if A[p]["sha256"] != B[p]["sha256"])
    sha_a = {}
    for p in removed:
        sha_a.setdefault(A[p]["sha256"], []).append(p)
    moved = [(sha_a[B[p]["sha256"]][0], p) for p in added if B[p]["sha256"] in sha_a]
    print(f"A={a.a} ({len(A)} files)  B={a.b} ({len(B)} files)")
    print(f"added={len(added)} removed={len(removed)} changed={len(changed)} moved_or_renamed={len(moved)}")
    rows = [{"status": "added", "path": p, "other": ""} for p in added] + \
           [{"status": "removed", "path": p, "other": ""} for p in removed] + \
           [{"status": "changed", "path": p, "other": ""} for p in changed] + \
           [{"status": "moved", "path": q, "other": p} for p, q in moved]
    for r in rows[:50]:
        print(f"  {r['status']:8s} {r['path']} {r['other']}")
    if a.out:
        with open(a.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["status", "path", "other"])
            w.writeheader()
            w.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="dsqc", description="Dataset quality check for detection datasets")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan", help="scan a dataset and write a report")
    s.add_argument("root", nargs="?", help="dataset root (auto layout) or override of the spec root")
    s.add_argument("--spec", help="layout spec YAML")
    s.add_argument("--format", default="auto", choices=["auto", "yolo", "voc", "coco"])
    s.add_argument("--names", help="comma-separated class names (YOLO ids -> names)")
    s.add_argument("-o", "--out", required=True)
    s.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    s.add_argument("--no-transform", action="store_true", help="skip rotation/flip hashes (check 04a)")
    s.add_argument("--thumbs", type=int, default=0, help="thumbnails per check (0 = none; images then never leave the dataset)")
    s.add_argument("--cache-dir", default=None, help="hash cache directory (default: no cache)")
    s.add_argument("--config", help="YAML overriding thresholds in checks.DEFAULTS")
    s.add_argument("--skip", help="comma-separated check names to skip")
    s.set_defaults(fn=cmd_scan)
    d = sub.add_parser("diff", help="compare the manifests of two scans")
    d.add_argument("a")
    d.add_argument("b")
    d.add_argument("-o", "--out", help="write the differences to this CSV")
    d.set_defaults(fn=cmd_diff)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
