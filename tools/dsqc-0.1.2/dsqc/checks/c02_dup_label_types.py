"""Check 02 - how the annotations of pixel-identical copies differ (types A/B/C1-C3, as defined in the accompanying manuscript).

The wording is deliberately neutral: a difference says the copies disagree, not which copy is right.

0.1.2: for groups of type C1-C3 the table also gives the pixel distances between matched boxes and a
small_box_jitter flag (see geometry.group_geometry). The flag never changes the type; a flagged C1/C2
group is reported with severity low (config jitter_downgrade) because a few pixels of offset on a
small box lower the IoU without the copies marking different objects.
"""
from collections import Counter

from ..geometry import SEVERITY_ORDER, group_geometry, group_type
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
    rows, types, jitter = [], Counter(), Counter()
    small = float(cfg.get("small_box_area_px2", 1024.0))
    for k, (sha, members) in enumerate(sorted(groups.items(), key=lambda x: x[1][0].rel), 1):
        gid = f"P{k:04d}"
        lsets = [im.labels.get(ds.primary) for im in members]
        geo = None
        if any(ls is None or ls.status == MISSING for ls in lsets):
            t, v = "missing", None
        elif lsets[0].file and all(ls.file == lsets[0].file for ls in lsets):
            t, v = "shared_file", None   # copies read the same annotation file (e.g. one XML for two folders)
        else:
            anns = [[(b.cls, b.xyxy()) for b in ls.boxes] for ls in lsets]
            t, v = group_type(anns, cfg["c_iou_low"], cfg["c_iou_high"])
            if t in ("C1", "C2", "C3") and not any(b.normalized for ls in lsets for b in ls.boxes):
                geo = group_geometry(anns, cfg["c_iou_high"], small)
        types[t] += 1
        row = {"group_id": gid, "n_copies": len(members), "files": "|".join(im.rel for im in members),
               "type": t, "min_iou": "" if v is None else f"{v:.4f}", "pixel_sha256": sha,
               "max_coord_diff_px": "", "max_center_offset_px": "", "low_iou_max_area_px2": "", "small_box_jitter": ""}
        sev, msg = SEV.get(t, "medium"), MEANING.get(t, "a copy has no annotation")
        if geo is not None:
            row.update({"max_coord_diff_px": f"{geo['max_coord_diff_px']:.1f}",
                        "max_center_offset_px": f"{geo['max_center_offset_px']:.1f}",
                        "low_iou_max_area_px2": "" if geo["low_iou_max_area_px2"] is None
                        else f"{geo['low_iou_max_area_px2']:.0f}",
                        "small_box_jitter": "yes" if geo["small_box_jitter"] else "no"})
            msg += f"; matched boxes differ by at most {geo['max_coord_diff_px']:.1f} px per coordinate"
            if geo["small_box_jitter"] and t in ("C1", "C2"):
                jitter[t] += 1
                msg += (f"; every low-IoU pair is two small boxes (area < {small:.0f} px2) whose centres lie inside "
                        "each other: probably the same object marked with a few pixels of offset")
                if cfg.get("jitter_downgrade", True):
                    sev = "low"
        rows.append(row)
        if t != "identical":
            res.findings.append(Finding(NAME, f"type_{t}", sev, [im.rel for im in members], msg, row))
    res.tables["duplicate_group_types"] = (["group_id", "n_copies", "files", "type", "min_iou", "pixel_sha256",
                                            "max_coord_diff_px", "max_center_offset_px", "low_iou_max_area_px2",
                                            "small_box_jitter"], rows)
    res.stats.update({f"type_{t}": types.get(t, 0) for t in SEVERITY_ORDER + ["shared_file", "missing"]})
    res.stats["groups"] = len(rows)
    res.stats["type_C1_small_box_jitter"] = jitter.get("C1", 0)
    res.stats["type_C2_small_box_jitter"] = jitter.get("C2", 0)
    return res
