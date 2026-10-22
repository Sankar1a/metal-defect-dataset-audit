"""Check registry. Each check: run(ds, cfg, ctx) -> CheckResult.

ctx is a dict shared between checks (e.g. the pixel-duplicate groups computed once).
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

DEFAULTS = {
    "c_iou_low": 0.5,            # C1 / C2 boundary
    "c_iou_high": 0.9,           # C2 / C3 boundary
    "tiny_min_side_px": 4.0,     # box with a side shorter than this -> tiny_box
    "full_image_ratio": 0.95,    # box area / image area at or above this -> full_image_box
    "dup_box_iou": 0.95,         # same-class boxes in one image overlapping at least this -> duplicate_box
    "oob_tol_px": 0.0,           # tolerance for out-of-bounds coordinates
    "max_rows_html": 200,        # findings shown per check in report.html
}


@dataclass
class CheckResult:
    name: str
    title: str
    findings: list = field(default_factory=list)
    tables: Dict[str, Tuple[List[str], List[dict]]] = field(default_factory=dict)
    stats: dict = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


def all_checks():
    from . import (c00_inventory, c01_exact_dup, c02_dup_label_types, c03_source_consistency,
                   c04a_transform_dup, c06_box_sanity, c07_split_version)
    return [c00_inventory, c01_exact_dup, c02_dup_label_types, c03_source_consistency,
            c04a_transform_dup, c06_box_sanity, c07_split_version]


def run_all(ds, cfg, skip=()):
    ctx, results = {}, []
    for mod in all_checks():
        if mod.NAME in skip:
            continue
        results.append(mod.run(ds, cfg, ctx))
    return results
