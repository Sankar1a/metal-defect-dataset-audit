#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Round 16 line 2/3/4/5: evaluation + CSV deliverables.

A arm: each best.pt -> all 9 val sets (27 evaluations)
B arm: each best.pt -> own val + val_d0 (6 evaluations)

All reported values are best.pt re-evaluated on the target val set
(r.box.map50 / r.box.map). Last-row and best-row log values are NOT used.
"""
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (AUG, ARM_RUNS, EXP, RUNS, VAL_SETS, VAL_YAML, YAML_A, YAML_B)

CLEAN_VAL_YAML = YAML_B           # datasets/apspc_clean1711_seed0 val
VAL_ORDER = VAL_SETS              # val_d0 first, then same*, then conf*
SEEDS = [0, 1, 42]


def read_results(run_dir):
    p = os.path.join(run_dir, "results.csv")
    with open(p, "r", encoding="utf-8") as f:
        rd = list(csv.DictReader(f))

    def g(row, k):
        for kk in row:
            if kk.strip() == k:
                return row[kk]
        return None

    best_i = max(range(len(rd)), key=lambda i: float(g(rd[i], "metrics/mAP50(B)")))
    return {
        "epochs_actual": len(rd),
        "best_epoch": int(float(g(rd[best_i], "epoch"))),
        "best_row_map50": float(g(rd[best_i], "metrics/mAP50(B)")),
        "train_seconds": float(g(rd[-1], "time")),
        "last_map50": float(g(rd[-1], "metrics/mAP50(B)")),
        "last_map5095": float(g(rd[-1], "metrics/mAP50-95(B)")),
    }


def august_meta():
    """val_set -> (dose_groups, dose_images, val_total_images, label_variant)
    taken verbatim from the August dose_results.csv so the columns match."""
    meta = {}
    with open(os.path.join(AUG, "dose_results.csv"), "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            meta[r["val_set"]] = (r["dose_groups"], r["dose_images"],
                                  r["val_total_images"], r["label_variant"])
    return meta


def main():
    from ultralytics import YOLO

    print("T16 eval start=%s" % time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    meta = august_meta()
    EVALDIR = os.path.join(EXP, "runs_eval")
    os.makedirs(EVALDIR, exist_ok=True)

    dose_rows, clean_rows, run_meta = [], [], []

    for arm, seed, data_yaml, run_name in ARM_RUNS:
        run_dir = os.path.join(RUNS, run_name)
        w = os.path.join(run_dir, "weights", "best.pt")
        info = read_results(run_dir)
        print("\n%s  epochs_actual=%d best_epoch=%d train_seconds=%.1f"
              % (run_name, info["epochs_actual"], info["best_epoch"], info["train_seconds"]),
              flush=True)
        run_meta.append({
            "run_name": run_name, "seed": seed, "arm": arm,
            "data_yaml": data_yaml,
            "epochs_actual": info["epochs_actual"],
            "best_epoch": info["best_epoch"],
            "train_seconds": "%.1f" % info["train_seconds"],
        })

        if arm == "A_leaktrain":
            targets = [(v, VAL_YAML[v]) for v in VAL_ORDER]
        else:
            targets = [("clean_val", CLEAN_VAL_YAML), ("val_d0", VAL_YAML["val_d0"])]

        for vname, vyaml in targets:
            r = YOLO(w).val(data=vyaml, imgsz=640, split="val", batch=16,
                            device=0, verbose=False,
                            project=EVALDIR, name="%s__on_%s" % (run_name, vname))
            m50, m5095 = float(r.box.map50), float(r.box.map)
            print("  %-14s map50=%.6f map5095=%.6f" % (vname, m50, m5095), flush=True)

            if arm == "A_leaktrain":
                dg, di, vti, lv = meta[vname]
                dose_rows.append({
                    "seed": seed, "run_name": run_name, "val_set": vname,
                    "dose_groups": dg, "dose_images": di,
                    "val_total_images": vti, "label_variant": lv,
                    "map50": "%.6f" % m50, "map5095": "%.6f" % m5095,
                })
            else:
                clean_rows.append({
                    "seed": seed, "run_name": run_name, "val_set": vname,
                    "map50": "%.6f" % m50, "map5095": "%.6f" % m5095,
                    "epochs_actual": info["epochs_actual"],
                    "best_epoch": info["best_epoch"],
                })

    # 5.1 dose_results_unified.csv
    p = os.path.join(EXP, "dose_results_unified.csv")
    cols = ["seed", "run_name", "val_set", "dose_groups", "dose_images",
            "val_total_images", "label_variant", "map50", "map5095"]
    with open(p, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for r in dose_rows:
            wr.writerow(r)
    print("\nwrote %s rows=%d" % (p, len(dose_rows)), flush=True)

    # 5.2 clean_repro_results.csv
    p = os.path.join(EXP, "clean_repro_results.csv")
    cols = ["seed", "run_name", "val_set", "map50", "map5095", "epochs_actual", "best_epoch"]
    with open(p, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for r in clean_rows:
            wr.writerow(r)
    print("wrote %s rows=%d" % (p, len(clean_rows)), flush=True)

    # 5.3 runs_meta.csv
    p = os.path.join(EXP, "runs_meta.csv")
    cols = ["run_name", "seed", "arm", "data_yaml", "epochs_actual", "best_epoch", "train_seconds"]
    with open(p, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for r in run_meta:
            wr.writerow(r)
    print("wrote %s rows=%d" % (p, len(run_meta)), flush=True)

    # terminal dump
    for p in (os.path.join(EXP, "dose_results_unified.csv"),
              os.path.join(EXP, "clean_repro_results.csv"),
              os.path.join(EXP, "runs_meta.csv")):
        print("\n" + "=" * 74, flush=True)
        print(p, flush=True)
        print("=" * 74, flush=True)
        with open(p, "r", encoding="utf-8") as f:
            sys.stdout.write(f.read())

    print("\nT16 eval end=%s" % time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
