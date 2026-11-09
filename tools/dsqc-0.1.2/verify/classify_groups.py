"""Classify APSPC pixel-identical duplicate groups by how their annotations differ.

Input : apspc_duplicate_groups.csv, apspc_duplicate_boxes.csv (in audit/)
Output: audit/apspc_group_types.csv

Rules (manuscript Section 2.2.2, Table 2-2), applied to every pair of copies in a group;
the group takes the most severe pairwise type (A > B > C1 > C2 > C3):
  A  : the copies' sets of defect classes differ
  B  : same classes, but the number of boxes of some class differs
  C1 : same classes and counts; after optimal one-to-one matching of same-class boxes,
       the minimum IoU over matched pairs is < 0.5
  C2 : 0.5 <= minimum IoU < 0.9
  C3 : minimum IoU >= 0.9
Usage: python classify_groups.py [audit_folder]   (requires numpy, scipy)
"""
import csv, itertools, os, sys
from collections import defaultdict, Counter
from scipy.optimize import linear_sum_assignment
import numpy as np

SEVERITY = ["C3", "C2", "C1", "B", "A"]


def iou(a, b):
    ax0, ay0, ax1, ay1 = min(a[0], a[2]), min(a[1], a[3]), max(a[0], a[2]), max(a[1], a[3])
    bx0, by0, bx1, by1 = min(b[0], b[2]), min(b[1], b[3]), max(b[0], b[2]), max(b[1], b[3])
    iw, ih = max(0.0, min(ax1, bx1) - max(ax0, bx0)), max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0 else 0.0


def pair_type(ba, bb):
    ca, cb = Counter(c for c, _ in ba), Counter(c for c, _ in bb)
    if set(ca) != set(cb):
        return "A", None
    if ca != cb:
        return "B", None
    worst = 1.0
    for cls in ca:
        A = [box for c, box in ba if c == cls]
        B = [box for c, box in bb if c == cls]
        m = np.array([[iou(x, y) for y in B] for x in A])
        r, c = linear_sum_assignment(-m)            # maximise total IoU
        worst = min(worst, float(m[r, c].min()))
    return ("C1" if worst < 0.5 else "C2" if worst < 0.9 else "C3"), worst


def main(folder):
    groups = defaultdict(list)
    with open(os.path.join(folder, "apspc_duplicate_groups.csv"), newline="") as f:
        for r in csv.DictReader(f):
            groups[r["group_id"]].append(r["filename"])
    boxes = defaultdict(list)
    with open(os.path.join(folder, "apspc_duplicate_boxes.csv"), newline="") as f:
        for r in csv.DictReader(f):
            boxes[r["filename"]].append((r["class_name"], tuple(float(r[k]) for k in ("xmin", "ymin", "xmax", "ymax"))))
    rows = []
    for gid in sorted(groups):
        files = sorted(groups[gid])
        worst_type, min_iou = "C3", None
        for fa, fb in itertools.combinations(files, 2):
            t, v = pair_type(boxes[fa], boxes[fb])
            if SEVERITY.index(t) > SEVERITY.index(worst_type):
                worst_type = t
            if v is not None:
                min_iou = v if min_iou is None else min(min_iou, v)
        rows.append({"group_id": gid, "n_copies": len(files), "files": "|".join(files),
                     "type": worst_type,
                     "min_iou": "" if worst_type in ("A", "B") or min_iou is None else f"{min_iou:.4f}"})
    out = os.path.join(folder, "apspc_group_types.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(Counter(r["type"] for r in rows))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
