"""Check 07 - split composition and dataset version fingerprint.

* per split and class: images containing the class and number of boxes (primary source)
* class absent from a split where other splits have it -> class_missing_in_split
* manifest of every image and annotation file (size, sha256) and a fingerprint of the whole
  dataset (sha256 over the sorted "path<TAB>sha256" lines). Two copies with the same fingerprint are
  byte-identical; `dsqc diff` lists what changed between two scans.
"""
import hashlib
import os
from collections import Counter, defaultdict

from ..index.hashing import sha256_file
from ..model import Finding
from . import CheckResult

NAME = "c07_split_version"
TITLE = "Split composition and version fingerprint"


def run(ds, cfg, ctx):
    res = CheckResult(NAME, TITLE)
    # --- manifest + fingerprint
    man = []
    for im in ds.images:
        man.append({"kind": "image", "path": im.rel, "size": im.size_bytes, "sha256": im.byte_sha or ""})
    seen = set()
    for src, files in ds.annotation_files.items():
        for f in files:
            if f in seen:
                continue
            seen.add(f)
            p = os.path.join(ds.root, f)
            man.append({"kind": "annotation", "path": f, "size": os.path.getsize(p), "sha256": sha256_file(p)})
    man.sort(key=lambda r: (r["kind"], r["path"]))
    h = hashlib.sha256()
    for r in sorted(man, key=lambda r: r["path"]):
        h.update(f"{r['path']}\t{r['sha256']}\n".encode("utf-8"))
    res.tables["manifest"] = (["kind", "path", "size", "sha256"], man)
    res.stats["fingerprint_sha256"] = h.hexdigest()
    res.stats["manifest_files"] = len(man)

    # --- split composition
    if ds.split_mode == "none" or not ds.primary:
        res.notes.append("no split defined" if ds.split_mode == "none" else "no box source")
        return res
    img_cnt, box_cnt, tot = defaultdict(Counter), defaultdict(Counter), Counter()
    for im in ds.images:
        sp = str(im.split)
        tot[sp] += 1
        ls = im.labels.get(ds.primary)
        if ls is None:
            continue
        for c in ls.classes():
            img_cnt[sp][c] += 1
        for b in ls.boxes:
            box_cnt[sp][b.cls] += 1
    classes = sorted({c for sp in img_cnt for c in img_cnt[sp]})
    rows = []
    for sp in sorted(tot):
        for c in classes:
            rows.append({"split": sp, "class": c, "images_with_class": img_cnt[sp][c], "boxes": box_cnt[sp][c],
                         "pct_of_split_images": round(100.0 * img_cnt[sp][c] / tot[sp], 2) if tot[sp] else 0.0})
    res.tables["split_class_counts"] = (["split", "class", "images_with_class", "boxes", "pct_of_split_images"], rows)
    for c in classes:
        have = [sp for sp in tot if img_cnt[sp][c] > 0]
        lack = [sp for sp in tot if img_cnt[sp][c] == 0 and sp != "None"]
        if have and lack:
            res.findings.append(Finding(NAME, "class_missing_in_split", "medium", [],
                                        f"class '{c}' has no image in split(s) {', '.join(lack)}",
                                        {"class": c, "missing_in": ";".join(lack)}))
    for sp in sorted(tot):
        res.stats[f"split.{sp}.images"] = tot[sp]
    return res
