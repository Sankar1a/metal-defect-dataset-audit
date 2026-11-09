"""Compare two dsqc output directories, e.g. the same dataset scanned by 0.1.1 and by 0.1.2.

usage: python verify/compare_outputs.py <output A (older)> <output B (newer)>

Everything A writes is compared with B: every statistic, note, findings-by-code and by-severity of
every check, the totals, every column of every table in A, manifest.csv, and the set of finding ids
with their severities. Statistics, tables and columns that only B has are listed, not counted.
Exit code 0 when nothing that A writes differs in B.
"""
import csv
import json
import os
import sys


def _rows(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compare(a_dir, b_dir, out=print):
    a = json.load(open(os.path.join(a_dir, "summary.json"), encoding="utf-8"))
    b = json.load(open(os.path.join(b_dir, "summary.json"), encoding="utf-8"))
    diffs, added = [], []
    for chk, ca in a["checks"].items():
        cb = b["checks"].get(chk)
        if cb is None:
            diffs.append(f"check {chk} missing in B")
            continue
        for k, v in ca["stats"].items():
            if cb["stats"].get(k) != v:
                diffs.append(f"{chk}.stats.{k}: {v} -> {cb['stats'].get(k)}")
        added += [f"{chk}.stats.{k} = {cb['stats'][k]}" for k in cb["stats"] if k not in ca["stats"]]
        for fld in ("findings_by_code", "findings_by_severity"):
            if ca[fld] != cb[fld]:
                diffs.append(f"{chk}.{fld}: {ca[fld]} -> {cb[fld]}")
        if ca["notes"] != cb["notes"]:
            short = lambda n: (n[:2] + [f"... ({len(n)} notes)"]) if len(n) > 3 else n   # noqa: E731
            diffs.append(f"{chk}.notes: {short(ca['notes'])} -> {short(cb['notes'])}")
    for k in ("findings_total", "findings_by_severity", "split_mode", "layout", "primary_source", "sources"):
        if a.get(k) != b.get(k):
            diffs.append(f"{k}: {a.get(k)} -> {b.get(k)}")
    tdir_a, tdir_b = os.path.join(a_dir, "tables"), os.path.join(b_dir, "tables")
    for fn in sorted(os.listdir(tdir_a)):
        pb = os.path.join(tdir_b, fn)
        if not os.path.isfile(pb):
            diffs.append(f"table {fn} missing in B")
            continue
        ra, rb = _rows(os.path.join(tdir_a, fn)), _rows(pb)
        cols = list(ra[0].keys()) if ra else []
        if [{c: r[c] for c in cols} for r in ra] != [{c: r.get(c) for c in cols} for r in rb]:
            n = sum(1 for x, y in zip(ra, rb) if {c: x[c] for c in cols} != {c: y.get(c) for c in cols})
            diffs.append(f"table {fn}: {n} rows differ in A's columns (A {len(ra)} rows, B {len(rb)} rows)")
        new_cols = [c for c in (list(rb[0].keys()) if rb else []) if c not in cols]
        if new_cols:
            added.append(f"table {fn}: new columns {new_cols}")
    added += [f"new table {fn}" for fn in sorted(os.listdir(tdir_b)) if fn not in os.listdir(tdir_a)]
    if _rows(os.path.join(a_dir, "manifest.csv")) != _rows(os.path.join(b_dir, "manifest.csv")):
        diffs.append("manifest.csv differs")
    fa = {r["id"]: r for r in _rows(os.path.join(a_dir, "findings.csv"))}
    fb = {r["id"]: r for r in _rows(os.path.join(b_dir, "findings.csv"))}
    if set(fa) - set(fb):
        diffs.append(f"finding ids only in A: {sorted(set(fa) - set(fb))}")
    if set(fb) - set(fa):
        diffs.append(f"finding ids only in B: {sorted(set(fb) - set(fa))}")
    sev = [f"{i} {fa[i]['code']} {fa[i]['severity']} -> {fb[i]['severity']}"
           for i in sorted(set(fa) & set(fb)) if fa[i]["severity"] != fb[i]["severity"]]
    out(f"A = {a_dir}\nB = {b_dir}")
    out(f"differences in what A writes: {len(diffs)}")
    for x in diffs:
        out(f"  DIFF  {x}")
    for x in sev:
        out(f"  SEVERITY  {x}")
    for x in added:
        out(f"  ONLY-B  {x}")
    return len(diffs)


if __name__ == "__main__":
    sys.exit(1 if compare(sys.argv[1], sys.argv[2]) else 0)
