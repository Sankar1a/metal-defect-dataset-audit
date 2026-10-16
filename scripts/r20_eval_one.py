#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Round 20 · 第3d步: 对某个 run 的 weights/best.pt 做实测。

model.val(...) 参数抄自 exp_apspc_dose_unified_20260915/scripts/eval_dose.py
(第十六轮评估脚本, 任务书指定第 88-90 行;原文见 eval_kwargs_source.txt),
只改 data 与输出位置:
  data         : 该 run 模板原来的 data yaml(只在本 run 自己的 val 上评估)
  project/name : round20_yolov8_20261008/val_runs/<name>__selfval

输出 JSON: round20_yolov8_20261008/_eval/<name>.json

Usage: r20_eval_one.py <run_name>
"""
import argparse
import hashlib
import json
import os
import sys
import time

import yaml

ROOT = "/home/DSJ44/paper2_yolov11_steel"
EXP = os.path.join(ROOT, "round20_yolov8_20261008")
TEMPLATE_ROOT = os.path.join(ROOT, "runs_unified_20260803")
VALRUNS = os.path.join(EXP, "val_runs")
OUTDIR = os.path.join(EXP, "_eval")


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def count_data_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    return max(0, len(lines) - 1)   # minus header


def template_of(run_name):
    tname = run_name.replace("yolov8n", "yolo11n").replace("yolov8s", "yolo11s")
    return os.path.join(TEMPLATE_ROOT, tname, "args.yaml")


def val_with_info(weight, data_yaml, project, name):
    """按第十六轮参数做 model.val,并取 model.info() 的参数量/GFLOPs。"""
    from ultralytics import YOLO

    model = YOLO(weight)

    import logging
    from ultralytics.utils import LOGGER

    printed = []
    real_print = print

    class _Cap(logging.Handler):
        def emit(self, record):
            printed.append(record.getMessage())

    _cap = _Cap(level=logging.INFO)
    LOGGER.addHandler(_cap)
    try:
        ret = model.info()
    finally:
        LOGGER.removeHandler(_cap)
    info_line = ""
    for ln in "\n".join(printed).splitlines():
        if "parameters" in ln and "GFLOPs" in ln:
            info_line = ln.strip()
    params, gflops, layers = "", "", None
    if isinstance(ret, (tuple, list)) and len(ret) >= 4:
        params = int(ret[1])
        gflops = round(float(ret[3]), 4)
        layers = int(ret[0])

    r = model.val(data=data_yaml, imgsz=640, split="val",
                  batch=16, device=0, verbose=False,
                  project=project, name=name)
    return {"map50": float(r.box.map50), "map5095": float(r.box.map),
            "params": params, "gflops": gflops, "layers": layers,
            "info_line": info_line}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_name")
    a = ap.parse_args()
    run_name = a.run_name

    run_dir = os.path.join(EXP, run_name)
    weight = os.path.join(run_dir, "weights", "best.pt")
    results_csv = os.path.join(run_dir, "results.csv")
    tpath = template_of(run_name)

    os.makedirs(VALRUNS, exist_ok=True)
    os.makedirs(OUTDIR, exist_ok=True)

    with open(tpath, "r", encoding="utf-8") as f:
        tmpl = yaml.safe_load(f)
    data_yaml = tmpl["data"]
    seed = tmpl["seed"]

    print("[E] run_name=%s" % run_name, flush=True)
    print("[E] best.pt=%s exists=%s" % (weight, os.path.exists(weight)), flush=True)
    print("[E] data=%s" % data_yaml, flush=True)

    rec = {
        "run_name": run_name, "seed": seed, "data": data_yaml, "best_pt": weight,
        "epochs_run": count_data_rows(results_csv) if os.path.exists(results_csv) else 0,
        "best_pt_sha256": sha256_file(weight) if os.path.exists(weight) else "",
        "map50": None, "map5095": None, "params": "", "gflops": "", "info_line": "",
    }

    if not os.path.exists(weight):
        print("[E] ERROR best.pt 缺失", flush=True)
        with open(os.path.join(OUTDIR, run_name + ".json"), "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
        return 1

    out = val_with_info(weight, data_yaml, VALRUNS, "%s__selfval" % run_name)
    rec.update(out)
    print("[E] model.info: %s" % out["info_line"], flush=True)
    print("[E] best.pt 实测 map50=%.6f (%.4f%%)  map5095=%.6f (%.4f%%)"
          % (rec["map50"], rec["map50"] * 100, rec["map5095"], rec["map5095"] * 100), flush=True)
    print("[E] params=%s gflops=%s" % (rec["params"], rec["gflops"]), flush=True)

    rec["eval_end"] = time.strftime("%Y-%m-%d %H:%M:%S")
    outp = os.path.join(OUTDIR, run_name + ".json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    print("[E] wrote %s" % outp, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
