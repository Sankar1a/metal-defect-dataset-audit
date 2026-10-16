#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Line ED: evaluation + aggregation for exp_gc10_label_20260914.

ED1 self-eval : best.pt on its own condition's val
ED2 cross-eval: best.pt on the OTHER condition's val
ED3 per-class AP50
ED4 gc10_label_results.csv
ED5 terminal print (all fields, no diffs/means/conclusions)

Run only AFTER line EC finishes.
"""
import csv
import os
import sys
import time

ROOT = "/home/DSJ44/paper2_yolov11_steel"
EXP = os.path.join(ROOT, "exp_gc10_label_20260914")
RUNS = os.path.join(EXP, "runs")
EVALDIR = os.path.join(EXP, "runs_eval")

YAML = {
    "xml": os.path.join(EXP, "configs", "gc10_xml.yaml"),
    "dir": os.path.join(EXP, "configs", "gc10_dir.yaml"),
}
CLASS_NAMES = ["chongkong", "hanfeng", "yueyawan", "shuiban", "youban",
               "siban", "yiwu", "yahen", "zhehen", "yaozhe"]

JOBS = [("xml", 0), ("xml", 1), ("xml", 42),
        ("dir", 0), ("dir", 1), ("dir", 42)]

# 统一基准 bestpt_val_* 口径 (verify3_20260805/metric_reconciliation.csv)
UNIFIED = {
    0:  {"map50": "67.12", "map5095": "34.80", "epochs_actual": 200},
    1:  {"map50": "65.89", "map5095": "34.46", "epochs_actual": 185},
    42: {"map50": "67.01", "map5095": "34.19", "epochs_actual": 200},
}


def read_results(run_dir):
    p = os.path.join(run_dir, "results.csv")
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        rd = list(csv.DictReader(f))
    if not rd:
        return None
    def g(row, k):
        for kk in row:
            if kk.strip() == k:
                return row[kk]
        return None
    last = rd[-1]
    best_i = max(range(len(rd)), key=lambda i: float(g(rd[i], "metrics/mAP50(B)")))
    best = rd[best_i]
    return {
        "epochs_total": len(rd),
        "last_map50": float(g(last, "metrics/mAP50(B)")),
        "last_map5095": float(g(last, "metrics/mAP50-95(B)")),
        "best_row_epoch": int(float(g(best, "epoch"))),
        "best_row_map50": float(g(best, "metrics/mAP50(B)")),
        "best_row_map5095": float(g(best, "metrics/mAP50-95(B)")),
    }


def main():
    from ultralytics import YOLO

    print("ED0 python=%s" % sys.version.split()[0], flush=True)
    print("ED0 start=%s" % time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    os.makedirs(EVALDIR, exist_ok=True)

    rows = []
    per_class_rows = []

    for cond, seed in JOBS:
        name = "gc10_yolo11n_%s_s%d" % (cond, seed)
        run_dir = os.path.join(RUNS, name)
        w = os.path.join(run_dir, "weights", "best.pt")
        print("\n" + "=" * 74, flush=True)
        print("ED %s" % name, flush=True)
        print("ED weights=%s exists=%s" % (w, os.path.exists(w)), flush=True)

        res = read_results(run_dir)
        if res is None:
            print("ED SKIP: results.csv 缺失或为空", flush=True)
            continue

        if not os.path.exists(w):
            print("ED SKIP: best.pt 缺失", flush=True)
            continue

        ev = {}
        for target in ("xml", "dir"):
            m = YOLO(w)
            r = m.val(data=YAML[target], imgsz=640, split="val",
                      batch=16, device=0, verbose=False,
                      project=EVALDIR, name="%s__on_%s" % (name, target))
            b = r.box
            ev["eval_on_%s_map50" % target] = float(b.map50)
            ev["eval_on_%s_map5095" % target] = float(b.map)
            print("ED %s eval_on_%s: map50=%.6f map5095=%.6f"
                  % (name, target, float(b.map50), float(b.map)), flush=True)

            # ED3 per-class AP50 (self-eval only)
            if target == cond:
                ap50 = b.ap50
                idx = b.ap_class_index
                got = {}
                for i, c in enumerate(list(idx)):
                    got[int(c)] = float(ap50[i])
                print("ED %s per-class 接口: r.box.ap50 (len=%d), r.box.ap_class_index=%s"
                      % (name, len(list(ap50)), list(idx)), flush=True)
                for ci in range(len(CLASS_NAMES)):
                    v = got.get(ci)
                    per_class_rows.append({
                        "run_name": name, "cond": cond, "seed": seed,
                        "class_index": ci, "class_name": CLASS_NAMES[ci],
                        "ap50": ("%.6f" % v) if v is not None else "",
                    })

        row = {"run_name": name, "cond": cond, "seed": seed}
        row.update(res)
        row.update(ev)
        rows.append(row)

    # ---------------- ED4 ----------------
    cols = ["run_name", "cond", "seed", "epochs_total", "last_map50", "last_map5095",
            "best_row_epoch", "best_row_map50", "best_row_map5095",
            "eval_on_xml_map50", "eval_on_xml_map5095",
            "eval_on_dir_map50", "eval_on_dir_map5095"]
    out = os.path.join(EXP, "gc10_label_results.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    print("\nED4 wrote %s rows=%d" % (out, len(rows)), flush=True)

    pc = os.path.join(EXP, "per_class_ap.csv")
    with open(pc, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["run_name", "cond", "seed",
                                          "class_index", "class_name", "ap50"])
        w.writeheader()
        for r in per_class_rows:
            w.writerow(r)
    print("ED4 wrote %s rows=%d" % (pc, len(per_class_rows)), flush=True)

    # ---------------- xml_vs_unified.csv ----------------
    xu = os.path.join(EXP, "xml_vs_unified.csv")
    with open(xu, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seed", "xml_run_name",
                    "xml_map50", "xml_map5095",
                    "unified_map50", "unified_map5095",
                    "xml_epochs_actual", "unified_epochs_actual",
                    "unified_source"])
        for cond, seed in JOBS:
            if cond != "xml":
                continue
            name = "gc10_yolo11n_xml_s%d" % seed
            r = next((x for x in rows if x["run_name"] == name), None)
            u = UNIFIED[seed]
            w.writerow([
                seed, name,
                ("%.6f" % r["eval_on_xml_map50"]) if r else "",
                ("%.6f" % r["eval_on_xml_map5095"]) if r else "",
                u["map50"], u["map5095"],
                r["epochs_total"] if r else "",
                u["epochs_actual"],
                "verify3_20260805/metric_reconciliation.csv bestpt_val_* 列",
            ])
    print("ED4 wrote %s" % xu, flush=True)

    # ---------------- ED5 ----------------
    print("\n" + "=" * 74, flush=True)
    print("ED5 gc10_label_results.csv 全部内容", flush=True)
    print("=" * 74, flush=True)
    with open(out, "r", encoding="utf-8") as f:
        sys.stdout.write(f.read())
    print("\n" + "=" * 74, flush=True)
    print("ED5 per_class_ap.csv 全部内容", flush=True)
    print("=" * 74, flush=True)
    with open(pc, "r", encoding="utf-8") as f:
        sys.stdout.write(f.read())
    print("\n" + "=" * 74, flush=True)
    print("ED5 xml_vs_unified.csv 全部内容", flush=True)
    print("=" * 74, flush=True)
    with open(xu, "r", encoding="utf-8") as f:
        sys.stdout.write(f.read())

    print("\nED end=%s" % time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
