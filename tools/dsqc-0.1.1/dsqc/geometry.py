"""Box geometry and the annotation comparator used by checks 02 and 03.

The comparator reproduces the one described in the accompanying manuscript
(audit/scripts/classify_groups.py) exactly, with one addition:
two annotations whose (class, box) multisets are equal are typed 'identical' instead of C3.
Rules for a pair of annotations a, b:
  identical : same multiset of (class, x1, y1, x2, y2)
  A  : the sets of classes differ
  B  : same classes, but the number of boxes of some class differs
  C1 : same classes and counts; after optimal one-to-one matching of same-class boxes
       (Hungarian, maximising IoU) the minimum IoU over matched pairs is < 0.5
  C2 : 0.5 <= minimum IoU < 0.9
  C3 : minimum IoU >= 0.9
A group takes the most severe pairwise type.
"""
from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment

SEVERITY_ORDER = ["identical", "C3", "C2", "C1", "B", "A"]   # mild -> severe


def iou(a, b):
    ax0, ay0, ax1, ay1 = min(a[0], a[2]), min(a[1], a[3]), max(a[0], a[2]), max(a[1], a[3])
    bx0, by0, bx1, by1 = min(b[0], b[2]), min(b[1], b[3]), max(b[0], b[2]), max(b[1], b[3])
    iw, ih = max(0.0, min(ax1, bx1) - max(ax0, bx0)), max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0 else 0.0


def pair_type(ba, bb, c1=0.5, c2=0.9):
    """ba, bb: lists of (cls, (x1, y1, x2, y2)). Returns (type, min_iou or None)."""
    if Counter((c, tuple(box)) for c, box in ba) == Counter((c, tuple(box)) for c, box in bb):
        return "identical", (1.0 if ba else None)
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
        r, c = linear_sum_assignment(-m)
        worst = min(worst, float(m[r, c].min()))
    return ("C1" if worst < c1 else "C2" if worst < c2 else "C3"), worst


def group_type(annotations, c1=0.5, c2=0.9):
    """annotations: list of box lists, one per copy. Returns (worst type, min_iou or None).

    min_iou is the minimum over all pairs that produced a value (manuscript convention) and is
    reported only when the worst type is not A or B.
    """
    worst, min_iou = "identical", None
    for i in range(len(annotations)):
        for j in range(i + 1, len(annotations)):
            t, v = pair_type(annotations[i], annotations[j], c1, c2)
            if SEVERITY_ORDER.index(t) > SEVERITY_ORDER.index(worst):
                worst = t
            if v is not None:
                min_iou = v if min_iou is None else min(min_iou, v)
    if worst in ("A", "B"):
        min_iou = None
    return worst, min_iou
