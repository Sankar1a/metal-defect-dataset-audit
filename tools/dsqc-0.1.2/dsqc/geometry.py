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

group_geometry (0.1.2) does not change any type; it describes, in pixels, how far apart the matched
boxes of a C-type group are, and flags groups whose low IoU comes from small boxes marked with a few
pixels of offset (IoU is a poor measure of agreement for small objects).
"""
import math
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


def matched_pairs(ba, bb):
    """The same-class one-to-one matching used by pair_type (Hungarian, maximising IoU).

    Returns [(cls, box_a, box_b, iou)], or None when the class sets or the per-class counts differ.
    """
    ca, cb = Counter(c for c, _ in ba), Counter(c for c, _ in bb)
    if ca != cb:
        return None
    out = []
    for cls in ca:
        A = [box for c, box in ba if c == cls]
        B = [box for c, box in bb if c == cls]
        m = np.array([[iou(x, y) for y in B] for x in A])
        r, c = linear_sum_assignment(-m)
        out += [(cls, A[i], B[j], float(m[i, j])) for i, j in zip(r, c)]
    return out


def _norm(b):
    return min(b[0], b[2]), min(b[1], b[3]), max(b[0], b[2]), max(b[1], b[3])


def _area(r):
    return (r[2] - r[0]) * (r[3] - r[1])


def _inside(p, r):
    return r[0] <= p[0] <= r[2] and r[1] <= p[1] <= r[3]


def group_geometry(annotations, c_high=0.9, small_area=32.0 * 32.0):
    """Pixel distances between the matched boxes of a group whose copies share classes and counts.

    max_coord_diff_px     largest |difference| of one box coordinate, over all matched pairs
    max_center_offset_px  largest distance between the centres of two matched boxes
    low_iou_max_area_px2  largest box area among the matched pairs with IoU < c_high (None if none)
    small_box_jitter      True when every matched pair with IoU < c_high consists of two small boxes
                          (area < small_area; 32 x 32 px is the COCO limit for small objects) whose
                          centres each lie inside the other box, i.e. the same object marked with a
                          few pixels of offset
    Returns None when some pair of copies differs in classes or counts (types A / B).
    """
    max_d = max_c = 0.0
    low = []
    for i in range(len(annotations)):
        for j in range(i + 1, len(annotations)):
            mp = matched_pairs(annotations[i], annotations[j])
            if mp is None:
                return None
            for _, a, b, v in mp:
                a, b = _norm(a), _norm(b)
                max_d = max(max_d, max(abs(x - y) for x, y in zip(a, b)))
                ca = ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2)
                cb = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
                max_c = max(max_c, math.hypot(ca[0] - cb[0], ca[1] - cb[1]))
                if v < c_high:
                    low.append((a, b, ca, cb))
    geo = {"max_coord_diff_px": max_d, "max_center_offset_px": max_c,
           "low_iou_max_area_px2": None, "small_box_jitter": False}
    if low:
        geo["low_iou_max_area_px2"] = max(max(_area(a), _area(b)) for a, b, _, _ in low)
        geo["small_box_jitter"] = all(max(_area(a), _area(b)) < small_area and _inside(ca, b) and _inside(cb, a)
                                      for a, b, ca, cb in low)
    return geo
