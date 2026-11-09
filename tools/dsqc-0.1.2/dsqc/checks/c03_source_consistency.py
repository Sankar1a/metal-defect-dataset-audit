"""Check 03 - agreement between label sources of the same image.

image-class source (folder, XML <folder> tag) vs box source: is the image-level class among the
box classes? (GC10-DET: 81 of 2306 annotated folder entries; an empty annotation counts
as 'not present', as in the accompanying manuscript).
image-class source vs image-class source (e.g. folder vs XML <folder> tag): equal or not.
box source vs box source: the A/B/C comparator of check 02.
The check reports disagreement only; it does not decide which source is correct.
"""
from collections import Counter, defaultdict

from ..geometry import pair_type
from ..model import EMPTY, ERROR, MISSING, OK, UNMAPPED, Finding
from . import CheckResult

NAME = "c03_source_consistency"
TITLE = "Agreement between label sources"


def run(ds, cfg, ctx):
    res = CheckResult(NAME, TITLE)
    stem_count = Counter(im.stem for im in ds.images)
    for cs in ds.class_sources():
        for bs in ds.box_sources():
            rows, contingency, n_both, n_mis = [], Counter(), 0, 0
            for im in ds.images:
                c, b = im.labels.get(cs), im.labels.get(bs)
                c_ok = c is not None and c.status == OK
                b_has = b is not None and b.status in (OK, EMPTY)
                box_classes = sorted(b.classes()) if b_has else []
                present = "n/a"
                if c_ok and b_has:
                    n_both += 1
                    present = "yes" if c.image_class in box_classes else "no"
                    for bc in (box_classes or ["(none)"]):
                        contingency[(c.image_class, bc)] += 1
                    if present == "no":
                        n_mis += 1
                        if cs not in ds.contingency_only:
                            code = "image_class_vs_empty_annotation" if not box_classes else "image_class_absent_from_boxes"
                            res.findings.append(Finding(
                                NAME, code, "medium", [im.rel],
                                f"{cs}='{c.image_class}' not among {bs} classes {box_classes or '(empty)'}",
                                {"class_source": cs, "box_source": bs, "image_class": c.image_class,
                                 "raw_image_class": c.raw_image_class, "box_classes": ";".join(box_classes)}))
                rows.append({"relative_path": im.rel, "stem": im.stem,
                             "image_class": c.image_class if c_ok else "",
                             "raw_image_class": (c.raw_image_class or "") if c is not None else "",
                             "box_label_status": b.status if b is not None else MISSING,
                             "n_objects": len(b.boxes) if b_has else "",
                             "box_classes": ";".join(box_classes),
                             "class_in_boxes": present if (c_ok and b_has) else (
                                 "image_class_" + c.status if (c is not None and not c_ok) else
                                 "box_label_" + (b.status if b else MISSING)),
                             "stem_file_count": stem_count[im.stem]})
            key = f"{cs}_vs_{bs}"
            res.tables[f"per_image_{key}"] = (["relative_path", "stem", "image_class", "raw_image_class",
                                              "box_label_status", "n_objects", "box_classes",
                                              "class_in_boxes", "stem_file_count"], rows)
            per_cls = Counter()
            per_cls_mis = Counter()
            for r in rows:
                if r["class_in_boxes"] in ("yes", "no"):
                    per_cls[r["image_class"]] += 1
                    per_cls_mis[r["image_class"]] += r["class_in_boxes"] == "no"
            res.tables[f"by_class_{key}"] = (["image_class", "images", "class_absent_from_boxes", "pct_absent"],
                                             [{"image_class": k, "images": n, "class_absent_from_boxes": per_cls_mis[k],
                                               "pct_absent": round(100.0 * per_cls_mis[k] / n, 2)}
                                              for k, n in sorted(per_cls.items())])
            res.tables[f"contingency_{key}"] = (["image_class", "box_class", "n_images"],
                                                [{"image_class": a, "box_class": b, "n_images": n}
                                                 for (a, b), n in sorted(contingency.items())])
            res.stats[f"{key}.images_with_both"] = n_both
            n_unm = sum(1 for im in ds.images if im.labels.get(cs) is not None and im.labels[cs].status == UNMAPPED)
            if n_unm:
                res.stats[f"{key}.image_class_unmapped"] = n_unm
            if cs in ds.contingency_only:
                res.notes.append(f"{cs}: no class mapping given; see contingency_{key} (no findings raised)")
                continue
            res.stats[f"{key}.disagree"] = n_mis
            res.stats[f"{key}.pct_disagree"] = round(100.0 * n_mis / n_both, 2) if n_both else 0.0

    csrc = ds.class_sources()
    for i in range(len(csrc)):
        for j in range(i + 1, len(csrc)):
            a, b = csrc[i], csrc[j]
            if a in ds.contingency_only or b in ds.contingency_only:
                continue
            key, n_both, n_mis = f"{a}_vs_{b}", 0, 0
            for im in ds.images:
                la, lb = im.labels.get(a), im.labels.get(b)
                if not (la and lb and la.status == OK and lb.status == OK):
                    continue
                n_both += 1
                if la.image_class != lb.image_class:
                    n_mis += 1
                    res.findings.append(Finding(NAME, "image_class_sources_differ", "medium", [im.rel],
                                                f"{a}='{la.image_class}' vs {b}='{lb.image_class}'",
                                                {"a": la.image_class, "b": lb.image_class}))
            res.stats[f"{key}.images_with_both"] = n_both
            res.stats[f"{key}.disagree"] = n_mis

    bsrc = ds.box_sources()
    for i in range(len(bsrc)):
        for j in range(i + 1, len(bsrc)):
            a, b = bsrc[i], bsrc[j]
            key, types = f"{a}_vs_{b}", Counter()
            for im in ds.images:
                la, lb = im.labels.get(a), im.labels.get(b)
                if not (la and lb) or MISSING in (la.status, lb.status) or ERROR in (la.status, lb.status):
                    continue
                t, v = pair_type([(x.cls, x.xyxy()) for x in la.boxes], [(x.cls, x.xyxy()) for x in lb.boxes],
                                 cfg["c_iou_low"], cfg["c_iou_high"])
                types[t] += 1
                if t != "identical":
                    res.findings.append(Finding(NAME, f"box_sources_type_{t}", "high" if t in ("A", "B") else "low",
                                                [im.rel], f"{a} vs {b}: type {t}",
                                                {"min_iou": "" if v is None else f"{v:.4f}"}))
            for t, n in types.items():
                res.stats[f"{key}.type_{t}"] = n
    if not ds.class_sources() and len(bsrc) < 2:
        res.notes.append("only one label source: nothing to compare")
    return res
