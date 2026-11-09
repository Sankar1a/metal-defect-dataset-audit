"""Compare dsqc scans of the three reference datasets with the audit lists published with the manuscript.

Usage:
  python -I tests/golden/check_manuscript.py --apspc OUT_APSPC --gc10 OUT_GC10 --neu OUT_NEU

PASS/FAIL lines are the acceptance test of dsqc v1 (every line must PASS).
INFO lines are numbers the manuscript did not report (new; to be reviewed before any use).
"""
import argparse
import csv
import json
import os
from collections import Counter, defaultdict

G = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manuscript")
RESULTS = []


def rows(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summary(out):
    with open(os.path.join(out, "summary.json"), encoding="utf-8") as f:
        return json.load(f)["checks"]


def check(name, ok, detail=""):
    RESULTS.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


def info(name, value):
    print(f"INFO  {name} = {value}")


def apspc(out):
    s = summary(out)
    inv, c1, c2 = s["c00_inventory"]["stats"], s["c01_exact_dup"]["stats"], s["c02_dup_label_types"]["stats"]
    check("APSPC image files = 1885", inv["image_files"] == 1885, inv["image_files"])
    check("APSPC duplicate groups = 82", c1["groups"] == 82, c1["groups"])
    check("APSPC files in groups = 174", c1["files_in_groups"] == 174, c1["files_in_groups"])
    check("APSPC share = 9.23%", c1["pct_files_in_groups"] == 9.23, c1["pct_files_in_groups"])
    gold = defaultdict(set)
    gold_sha = {}
    for r in rows(os.path.join(G, "apspc_duplicate_groups.csv")):
        gold[r["group_id"]].add(r["filename"])
        gold_sha[r["group_id"]] = r["pixel_sha256"]
    gold_sets = {frozenset(v): k for k, v in gold.items()}
    ours = defaultdict(set)
    our_sha = {}
    for r in rows(os.path.join(out, "tables", "c01_exact_dup__pixel_duplicate_groups.csv")):
        ours[r["group_id"]].add(os.path.basename(r["file"]))
        our_sha[r["group_id"]] = r["pixel_sha256"]
    our_sets = {frozenset(v): k for k, v in ours.items()}
    check("APSPC group membership identical to the audit", set(gold_sets) == set(our_sets),
          f"only audit={len(set(gold_sets) - set(our_sets))}, only dsqc={len(set(our_sets) - set(gold_sets))}")
    same_sha = all(gold_sha[gold_sets[k]] == our_sha[our_sets[k]] for k in set(gold_sets) & set(our_sets))
    check("APSPC pixel sha256 per group identical", same_sha)
    gt = {frozenset(r["files"].split("|")): r for r in rows(os.path.join(G, "apspc_group_types.csv"))}
    mism = 0
    for r in rows(os.path.join(out, "tables", "c02_dup_label_types__duplicate_group_types.csv")):
        key = frozenset(os.path.basename(x) for x in r["files"].split("|"))
        g = gt.get(key)
        if g is None or g["type"] != r["type"] or g["min_iou"] != r["min_iou"]:
            mism += 1
    check("APSPC type and min IoU per group identical", mism == 0, f"mismatches={mism}")
    want = {"A": 30, "B": 5, "C1": 11, "C2": 27, "C3": 9}
    got = {t: c2.get(f"type_{t}") for t in want}
    check("APSPC types A30 B5 C1 11 C2 27 C3 9", got == want, got)
    info("APSPC identical-annotation groups", c2.get("type_identical"))
    info("APSPC groups with re-encoded bytes", c1.get("groups_with_reencoding"))
    info("APSPC rotation/flip groups (beyond pixel-identical)", s["c04a_transform_dup"]["stats"].get("groups"))
    info("APSPC distinct images after rotation/flip", inv.get("distinct_after_rot_flip"))
    c3 = s["c03_source_consistency"]["stats"]
    info("APSPC XML <folder> values not mapped to a class", inv.get("xml_folder.images_label_unmapped"))
    info("APSPC XML <folder> class compared with boxes (images)", c3.get("xml_folder_vs_xml.images_with_both"))
    info("APSPC XML <folder> class absent from boxes", c3.get("xml_folder_vs_xml.disagree"))
    p = os.path.join(out, "tables", "c03_source_consistency__by_class_xml_folder_vs_xml.csv")
    if os.path.isfile(p):
        for r in rows(p):
            info(f"APSPC   <folder>={r['image_class']}", f"{r['class_absent_from_boxes']}/{r['images']} absent")


def gc10(out):
    s = summary(out)
    inv = s["c00_inventory"]["stats"]
    for k, v in (("image_files", 2312), ("distinct_stems", 2300), ("xml.annotation_files", 2294),
                 ("xml.images_with_label", 2306)):
        check(f"GC10 {k} = {v}", inv.get(k) == v, inv.get(k))
    per = rows(os.path.join(out, "tables", "c03_source_consistency__per_image_folder_vs_xml.csv"))
    no = {r["relative_path"] for r in per if r["class_in_boxes"] == "no"}
    gold_no = {r["relative_path"] for r in rows(os.path.join(G, "gc10_folder_xml_mismatch.csv"))}
    check("GC10 folder-XML disagreements = 81", len(no) == 81, len(no))
    check("GC10 disagreement list identical to the audit", no == gold_no,
          f"only audit={len(gold_no - no)}, only dsqc={len(no - gold_no)}")
    gold_all = {r["relative_path"]: r for r in rows(os.path.join(G, "gc10_folder_xml_consistency.csv"))}
    conv = {"yes": "yes", "no": "no", "box_label_missing": "xml_missing"}
    bad = [r["relative_path"] for r in per
           if gold_all.get(r["relative_path"], {}).get("folder_in_xml") != conv.get(r["class_in_boxes"])]
    check("GC10 per-image folder_in_xml identical for all 2312 entries", not bad and len(per) == len(gold_all),
          f"differ={len(bad)}, rows dsqc={len(per)} audit={len(gold_all)}")
    stems = {r["stem"] for r in per if int(r["stem_file_count"]) > 1}
    gold_stems = {r["stem"] for r in rows(os.path.join(G, "gc10_cross_folder_duplicates.csv"))}
    check("GC10 stems in two folders identical (12)", stems == gold_stems and len(stems) == 12, len(stems))
    c1, c3 = s["c01_exact_dup"]["stats"], s["c03_source_consistency"]["stats"]
    info("GC10 distinct pixel images", inv.get("distinct_pixel_sha256"))
    info("GC10 pixel-identical groups", c1.get("groups"))
    info("GC10 rotation/flip groups (beyond pixel-identical)", s["c04a_transform_dup"]["stats"].get("groups"))
    info("GC10 XML <folder> values not mapped to a class", inv.get("xml_folder.images_label_unmapped"))
    info("GC10 duplicate groups sharing one XML", s["c02_dup_label_types"]["stats"].get("type_shared_file"))
    info("GC10 XML <folder> tag vs class folder disagreements", c3.get("xml_folder_vs_folder.disagree"))
    info("GC10 XML <folder> tag vs XML boxes disagreements", c3.get("xml_folder_vs_xml.disagree"))
    info("GC10 box issues", {k: v for k, v in s["c06_box_sanity"]["stats"].items()})


def neu(out):
    s = summary(out)
    c1 = s["c01_exact_dup"]["stats"]
    check("NEU image files = 1800", s["c00_inventory"]["stats"]["image_files"] == 1800)
    check("NEU pixel-identical groups = 1", c1["groups"] == 1, c1["groups"])
    g = rows(os.path.join(out, "tables", "c01_exact_dup__pixel_duplicate_groups.csv"))
    files = sorted(r["file"] for r in g)
    check("NEU pair = train/patches_101 + val/patches_105",
          files == ["images/train/patches_101.jpg", "images/val/patches_105.jpg"], files)
    check("NEU pair crosses train/val", bool(g) and g[0]["cross_split"] == "True")
    check("NEU pixel sha256 = cfba9246...", bool(g) and g[0]["pixel_sha256"].startswith("cfba9246320418e2"))
    info("NEU pair annotation type", rows(os.path.join(out, "tables", "c02_dup_label_types__duplicate_group_types.csv"))[0]["type"] if g else None)
    info("NEU rotation/flip groups (beyond pixel-identical)", s["c04a_transform_dup"]["stats"].get("groups"))
    info("NEU box issues", s["c06_box_sanity"]["stats"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apspc")
    ap.add_argument("--gc10")
    ap.add_argument("--neu")
    a = ap.parse_args()
    for fn, out in ((apspc, a.apspc), (gc10, a.gc10), (neu, a.neu)):
        if out:
            print(f"--- {fn.__name__}: {out}")
            try:
                fn(out)
            except Exception as e:  # report and continue with the next dataset
                check(f"{fn.__name__} checker ran", False, f"{type(e).__name__}: {e}")
    n_fail = RESULTS.count(False)
    print(f"=== GOLDEN {'PASS' if n_fail == 0 else 'FAIL'}: {len(RESULTS) - n_fail}/{len(RESULTS)} checks passed ===")


if __name__ == "__main__":
    main()
