"""Check 04a - images that become pixel-identical after a rotation (90/180/270) or a flip.

No threshold is involved. Groups that are already pixel-identical (check 01) are not repeated:
a group is reported only if it contains at least two different pixel hashes.
Approximate duplicates (crop, resize, re-compression) are version 2 (thresholds from injection tests).
"""
from collections import defaultdict

from ..model import Finding
from . import CheckResult

NAME = "c04a_transform_dup"
TITLE = "Duplicates after rotation / flip"


def run(ds, cfg, ctx):
    res = CheckResult(NAME, TITLE)
    if not any(im.canon_sha for im in ds.images):
        res.notes.append("rotation/flip hashes not computed (--no-transform): skipped")
        return res
    g = defaultdict(list)
    for im in ds.images:
        if im.canon_sha:
            g[im.canon_sha].append(im)
    rows, n_groups, n_cross = [], 0, 0
    for sha, members in sorted(g.items(), key=lambda x: min(im.rel for im in x[1])):
        if len({im.pixel_sha for im in members}) < 2:
            continue
        members = sorted(members, key=lambda im: im.rel)
        n_groups += 1
        gid = f"T{n_groups:04d}"
        splits = sorted({str(im.split) for im in members})
        cross = ds.split_mode != "none" and len(splits) > 1
        n_cross += cross
        for im in members:
            rows.append({"group_id": gid, "canon_sha256": sha, "n_files": len(members),
                         "n_distinct_pixels": len({m.pixel_sha for m in members}), "splits": ";".join(splits),
                         "cross_split": cross, "file": im.rel, "pixel_sha256": im.pixel_sha})
        res.findings.append(Finding(NAME, "rot_flip_cross_split" if cross else "rot_flip_duplicate",
                                    "high" if cross else "medium", [im.rel for im in members],
                                    "same pixels after rotation/flip" + (f" across splits {', '.join(splits)}" if cross else ""),
                                    {"group_id": gid, "splits": ";".join(splits)}))
    res.tables["rot_flip_groups"] = (["group_id", "canon_sha256", "n_files", "n_distinct_pixels", "splits",
                                      "cross_split", "file", "pixel_sha256"], rows)
    res.stats.update({"groups": n_groups, "cross_split_groups": n_cross,
                      "files_in_groups": len(rows)})
    return res
