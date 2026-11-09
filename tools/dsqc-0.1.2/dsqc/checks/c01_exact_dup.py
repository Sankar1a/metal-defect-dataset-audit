"""Check 01 - pixel-identical images and leakage across splits.

Two image files are duplicates when their decoded RGB pixels are identical (pixel_sha), even if
the files differ in bytes (re-encoding). Groups spanning several splits are leakage. When the
dataset has no split, a group is reported as a risk: a random split can put copies on both sides.
"""
from collections import defaultdict

from ..model import MISSING, Finding
from . import CheckResult

NAME = "c01_exact_dup"
TITLE = "Pixel-identical duplicates and cross-split leakage"


def pixel_groups(ds, ctx):
    if "pixel_groups" not in ctx:
        g = defaultdict(list)
        for im in ds.images:
            if im.pixel_sha:
                g[im.pixel_sha].append(im)
        ctx["pixel_groups"] = {k: sorted(v, key=lambda im: im.rel) for k, v in g.items() if len(v) > 1}
    return ctx["pixel_groups"]


def run(ds, cfg, ctx):
    res = CheckResult(NAME, TITLE)
    groups = pixel_groups(ds, ctx)
    rows, n_files, n_cross = [], 0, 0
    for k, (sha, members) in enumerate(sorted(groups.items(), key=lambda x: x[1][0].rel), 1):
        gid = f"P{k:04d}"
        splits = sorted({str(im.split) for im in members})
        cross = ds.split_mode != "none" and len(splits) > 1
        n_files += len(members)
        n_cross += cross
        label_status = "n/a"
        if ds.primary:
            lsets = [im.labels.get(ds.primary) for im in members]
            if any(ls is None or ls.status == MISSING for ls in lsets):
                label_status = "missing"
            elif lsets[0].file and all(ls.file == lsets[0].file for ls in lsets):
                label_status = "shared_file"
            else:
                keys = [sorted(b.key() for b in ls.boxes) for ls in lsets]
                label_status = "identical" if all(x == keys[0] for x in keys[1:]) else "different"
        data = {"group_id": gid, "pixel_sha256": sha, "n_files": len(members),
                "byte_distinct": len({im.byte_sha for im in members}), "splits": ";".join(splits),
                "cross_split": cross, "label_status": label_status}
        for im in members:
            rows.append({**data, "file": im.rel, "split": im.split, "byte_sha256": im.byte_sha,
                         "width": im.width, "height": im.height})
        if ds.split_mode == "none":
            code, sev, msg = "pixel_duplicate", "medium", "pixel-identical copies (no split defined: a random split may separate them)"
        elif cross:
            code, sev, msg = "cross_split_leak", "high", f"pixel-identical copies in splits {', '.join(splits)}"
        else:
            code, sev, msg = "pixel_duplicate_same_split", "medium", "pixel-identical copies within one split"
        res.findings.append(Finding(NAME, code, sev, [im.rel for im in members],
                                    f"{msg}; labels {label_status}", data))
    res.tables["pixel_duplicate_groups"] = (
        ["group_id", "pixel_sha256", "n_files", "byte_distinct", "splits", "cross_split", "label_status",
         "file", "split", "byte_sha256", "width", "height"], rows)
    n = len(ds.images)
    res.stats.update({"groups": len(groups), "files_in_groups": n_files,
                      "pct_files_in_groups": round(100.0 * n_files / n, 2) if n else 0.0,
                      "cross_split_groups": n_cross,
                      "groups_with_reencoding": sum(1 for v in groups.values() if len({im.byte_sha for im in v}) > 1)})
    return res
