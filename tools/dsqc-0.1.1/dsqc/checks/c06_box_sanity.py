"""Check 06 - annotation-file and box-level sanity, for every box source.

codes (severity):
  missing_annotation (medium)   image without an annotation file
  empty_annotation (low)        annotation file without any object (may be a valid defect-free image)
  unreadable_annotation (high)  annotation file could not be parsed
  parse_issue (high)            malformed line / object without bndbox / bad bbox / several files per image
  unknown_class (high)          class not in class_names (only when class_names is given)
  alias_class (info)            class written in a variant spelling that was mapped by class_aliases
  size_mismatch (medium)        image size written in the annotation differs from the real image
  degenerate_box (high)         zero or negative width/height
  out_of_bounds (medium)        box extends outside the image
  tiny_box (low)                a side shorter than tiny_min_side_px
  full_image_box (low)          box covers >= full_image_ratio of the image
  duplicate_box (medium)        two same-class boxes in one image with IoU >= dup_box_iou
"""
from collections import Counter

from ..geometry import iou
from ..model import EMPTY, ERROR, MISSING, Finding
from . import CheckResult

NAME = "c06_box_sanity"
TITLE = "Annotation files and boxes"

SEV = {"missing_annotation": "medium", "empty_annotation": "low", "unreadable_annotation": "high",
       "parse_issue": "high", "unknown_class": "high", "alias_class": "info", "size_mismatch": "medium",
       "degenerate_box": "high", "out_of_bounds": "medium", "tiny_box": "low", "full_image_box": "low",
       "duplicate_box": "medium"}


def run(ds, cfg, ctx):
    res = CheckResult(NAME, TITLE)
    known = set(ds.class_names) if ds.class_names else None
    tol = cfg["oob_tol_px"]
    rows, counts = [], Counter()

    def flag(src, im, code, detail, n=1, extra=None):
        counts[(src, code)] += n
        rows.append({"source": src, "file": im.rel, "code": code, "severity": SEV[code], "n": n, "detail": detail})
        res.findings.append(Finding(NAME, code, SEV[code], [im.rel], f"{src}: {detail}",
                                    {"source": src, "n": n, **(extra or {})}))

    for src in ds.box_sources():
        box_counts = Counter()
        for im in ds.images:
            ls = im.labels.get(src)
            if ls is None or ls.status == MISSING:
                flag(src, im, "missing_annotation", "no annotation file")
                continue
            if ls.status == ERROR:
                flag(src, im, "unreadable_annotation", "; ".join(d for _, d in ls.issues) or "parse error")
                continue
            for code, d in ls.issues:
                if code != "parse_error":
                    flag(src, im, "parse_issue", f"{code}: {d}")
            if ls.status == EMPTY:
                flag(src, im, "empty_annotation", "annotation has no object")
            W, H = im.width, im.height
            if ls.declared_size and W and H and tuple(ls.declared_size) != (W, H):
                flag(src, im, "size_mismatch", f"annotation says {ls.declared_size[0]}x{ls.declared_size[1]}, image is {W}x{H}")
            per = Counter()
            for b in ls.boxes:
                box_counts[b.cls] += 1
                if known is not None and b.cls not in known:
                    per[("unknown_class", b.cls)] += 1
                if b.raw_cls != b.cls and not b.raw_cls.isdigit():
                    per[("alias_class", f"{b.raw_cls}->{b.cls}")] += 1
                if b.normalized or W is None:
                    continue
                w, h = b.x2 - b.x1, b.y2 - b.y1
                if w <= 0 or h <= 0:
                    per[("degenerate_box", b.cls)] += 1
                    continue
                if b.x1 < -tol or b.y1 < -tol or b.x2 > W + tol or b.y2 > H + tol:
                    per[("out_of_bounds", b.cls)] += 1
                if min(w, h) < cfg["tiny_min_side_px"]:
                    per[("tiny_box", b.cls)] += 1
                if w * h >= cfg["full_image_ratio"] * W * H:
                    per[("full_image_box", b.cls)] += 1
            for i in range(len(ls.boxes)):
                for j in range(i + 1, len(ls.boxes)):
                    a, c = ls.boxes[i], ls.boxes[j]
                    if a.cls == c.cls and not a.normalized and iou(a.xyxy(), c.xyxy()) >= cfg["dup_box_iou"]:
                        per[("duplicate_box", a.cls)] += 1
            by_code = {}
            for (code, what), n in per.items():
                by_code.setdefault(code, []).append((what, n))
            for code, items in by_code.items():
                flag(src, im, code, ", ".join(f"{w} x{n}" for w, n in items), sum(n for _, n in items))
        res.tables[f"class_counts_{src}"] = (["class", "boxes"], [{"class": c, "boxes": n}
                                                                 for c, n in sorted(box_counts.items())])
    res.tables["box_issues"] = (["source", "file", "code", "severity", "n", "detail"], rows)
    for (src, code), n in sorted(counts.items()):
        res.stats[f"{src}.{code}"] = n
    res.stats["images_flagged"] = len({r["file"] for r in rows if r["severity"] != "info"})
    return res
