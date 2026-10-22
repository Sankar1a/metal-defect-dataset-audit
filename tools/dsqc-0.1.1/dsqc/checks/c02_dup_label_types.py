"""Check 02 - how the annotations of pixel-identical copies differ (types A/B/C1-C3, as defined in the accompanying manuscript).

The wording is deliberately neutral: a difference says the copies disagree, not which copy is right.
"""
from collections import Counter

from ..geometry import SEVERITY_ORDER, group_type
from ..model import MISSING, Finding
from . import CheckResult
from .c01_exact_dup import pixel_groups

NAME = "c02_dup_label_types"
TITLE = "Annotation differences inside duplicate groups (A / B / C1-C3)"

SEV = {"A": "high", "B": "high", "C1": "medium", "C2": "low", "C3": "low", "identical": "info", "shared_file": "info"}
MEANING = {"A": "class sets differ", "B": "same classes, box counts differ",
           "C1": "same classes and counts, min matched IoU < low threshold",
           "C2": "same classes and counts, min matched IoU between thresholds",
           "C3": "same classes and counts, min matched IoU >= high threshold",
           "identical": "annotations identical",
           "shared_file": "copies share one annotation file (cannot differ; not evidence of agreement)"}


def run(ds, cfg, ctx):
    res = CheckResult(NAME, TITLE)
    if not ds.primary:
        res.notes.append("no box source: skipped")
        return res
    groups = pixel_groups(ds, ctx)
    rows, types = [], Counter()
    for k, (sha, members) in enumerate(sorted(groups.items(), key=lambda x: x[1][0].rel), 1):
        gid = f"P{k:04d}"
        lsets = [im.labels.get(ds.primary) for im in members]
        if any(ls is None or ls.status == MISSING for ls in lsets):
            t, v = "missing", None
        elif lsets[0].file and all(ls.file == lsets[0].file for ls in lsets):
            t, v = "shared_file", None   # copies read the same annotation file (e.g. one XML for two folders)
        else:
            anns = [[(b.cls, b.xyxy()) for b in ls.boxes] for ls in lsets]
            t, v = group_type(anns, cfg["c_iou_low"], cfg["c_iou_high"])
        types[t] += 1
        row = {"group_id": gid, "n_copies": len(members), "files": "|".join(im.rel for im in members),
               "type": t, "min_iou": "" if v is None else f"{v:.4f}", "pixel_sha256": sha}
        rows.append(row)
        if t != "identical":
            res.findings.append(Finding(NAME, f"type_{t}", SEV.get(t, "medium"), [im.rel for im in members],
                                        MEANING.get(t, "a copy has no annotation"), row))
    res.tables["duplicate_group_types"] = (["group_id", "n_copies", "files", "type", "min_iou", "pixel_sha256"], rows)
    res.stats.update({f"type_{t}": types.get(t, 0) for t in SEVERITY_ORDER + ["shared_file", "missing"]})
    res.stats["groups"] = len(rows)
    return res
