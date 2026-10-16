#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Round 17 · B3 · per-class instance counts for the 9 APSPC dose val sets.

Reads label files only. No model, no GPU.
Output: B_valset_class_counts.csv  (valset, class_name, n_boxes, n_images)
"""
import csv
import os
import sys

import yaml

ROOT = "/home/DSJ44/paper2_yolov11_steel"
OUT = os.path.join(ROOT, "round17_eval_20261005")
CFG = os.path.join(ROOT, "exp_apspc_dose_20260817", "configs")

VAL_SETS = ["val_d0",
            "val_d20_same", "val_d28_same", "val_d41_same", "val_d82_same",
            "val_d20_conf", "val_d28_conf", "val_d41_conf", "val_d82_conf"]


def main():
    out_csv = os.path.join(OUT, "B_valset_class_counts.csv")
    f = open(out_csv, "w", newline="", encoding="utf-8")
    w = csv.writer(f)
    w.writerow(["valset", "class_name", "n_boxes", "n_images"])
    f.flush()

    total_rows = 0
    for v in VAL_SETS:
        yp = os.path.join(CFG, "apspc_%s.yaml" % v[len("val_"):])
        with open(yp, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)

        root = cfg["path"]
        val_rel = cfg["val"]                      # e.g. "val_d0/images"
        names = cfg["names"]                      # {0: 'aoxian', ...}
        if isinstance(names, list):
            names = {i: n for i, n in enumerate(names)}

        # labels dir sits next to the images dir
        img_dir = os.path.join(root, val_rel)
        lab_dir = os.path.join(os.path.dirname(img_dir), "labels")

        box_counts = {i: 0 for i in names}
        img_counts = {i: 0 for i in names}

        lab_files = sorted(x for x in os.listdir(lab_dir) if x.endswith(".txt"))
        for lf in lab_files:
            seen = set()
            with open(os.path.join(lab_dir, lf), "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    ci = int(float(line.split()[0]))
                    box_counts[ci] = box_counts.get(ci, 0) + 1
                    seen.add(ci)
            for ci in seen:
                img_counts[ci] = img_counts.get(ci, 0) + 1

        for ci in sorted(names):
            w.writerow([v, names[ci], box_counts.get(ci, 0), img_counts.get(ci, 0)])
            total_rows += 1
        f.flush()
        print("B3 %-13s labels=%d images_dir=%d boxes=%d"
              % (v, len(lab_files), len(os.listdir(img_dir)), sum(box_counts.values())),
              flush=True)

    f.close()
    print("B3 wrote %s rows=%d" % (out_csv, total_rows), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
