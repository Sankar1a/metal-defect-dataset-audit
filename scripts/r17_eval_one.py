#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Round 17 · 第3d步: 对某个 run 的 weights/best.pt 做实测。

model.val(...) 参数抄自 exp_apspc_dose_unified_20260915/scripts/eval_dose.py
(第十六轮评估脚本, 原文见 eval_kwargs_source.txt),只改 data 与输出位置:
  data   : 该 run 模板原来的 data yaml(不跨数据集,只在本 run 自己的 val 上评估)
  project/name : round17_yolo11m_20261005/runs_eval/<name>__selfval

输出 JSON: round17_yolo11m_20261005/_eval/<name>.json
  {run_name, seed, data, epochs_run, best_pt_sha256, map50, map5095,
   params, gflops, info_line}

Usage: r17_eval_one.py <run_name>
"""
import argparse
import hashlib
import json
import os
import sys
import time

import yaml

ROOT = "/home/DSJ44/paper2_yolov11_steel"
EXP = os.environ.get("R17_EXP", os.path.join(ROOT, "round17_yolo11m_20261005"))
TEMPLATE_ROOT = os.path.join(ROOT, "runs_unified_20260803")
EVALDIR = os.path.join(EXP, "runs_eval")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_name")
    a = ap.parse_args()
    run_name = a.run_name

    run_dir = os.path.join(EXP, run_name)
    weight = os.path.join(run_dir, "weights", "best.pt")
    results_csv = os.path.join(run_dir, "results.csv")
    tpath = os.path.join(TEMPLATE_ROOT, run_name.replace("11m", "11s"), "args.yaml")

    os.makedirs(EVALDIR, exist_ok=True)
    os.makedirs(OUTDIR, exist_ok=True)

    with open(tpath, "r", encoding="utf-8") as f:
        tmpl = yaml.safe_load(f)
    data_yaml = tmpl["data"]
    seed = tmpl["seed"]

    print("[E] run_name=%s" % run_name, flush=True)
    print("[E] best.pt=%s exists=%s" % (weight, os.path.exists(weight)), flush=True)
    print("[E] data=%s" % data_yaml, flush=True)

    rec = {
        "run_name": run_name,
        "seed": seed,
        "data": data_yaml,
        "best_pt": weight,
        "epochs_run": count_data_rows(results_csv) if os.path.exists(results_csv) else 0,
        "best_pt_sha256": sha256_file(weight) if os.path.exists(weight) else "",
        "map50": None, "map5095": None,
        "params": "", "gflops": "", "info_line": "",
    }

    if not os.path.exists(weight):
        print("[E] ERROR best.pt 缺失", flush=True)
        with open(os.path.join(OUTDIR, run_name + ".json"), "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
        return 1

    from ultralytics import YOLO

    model = YOLO(weight)

    # ---- 参数量与 GFLOPs: model.info() 的输出 ----
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
    info_txt = "\n".join(printed)
    for ln in info_txt.splitlines():
        real_print("[E] model.info: %s" % ln, flush=True)
    info_line = ""
    for ln in info_txt.splitlines():
        if "parameters" in ln and "GFLOPs" in ln:
            info_line = ln.strip()
    if isinstance(ret, (tuple, list)) and len(ret) >= 4:
        n_layers, n_params, n_grads, n_flops = ret[0], ret[1], ret[2], ret[3]
        rec["params"] = int(n_params)
        rec["gflops"] = round(float(n_flops), 4)
        real_print("[E] model.info() 返回 n_layers=%s n_params=%s n_gradients=%s GFLOPs=%s"
                   % (n_layers, n_params, n_grads, n_flops), flush=True)
    rec["info_line"] = info_line
    rec["layers"] = int(ret[0]) if isinstance(ret, (tuple, list)) and len(ret) >= 4 else None

    # ---- 第十六轮评估脚本的 model.val(...) 参数,只改 data 与输出位置 ----
    r = model.val(data=data_yaml, imgsz=640, split="val",
                  batch=16, device=0, verbose=False,
                  project=EVALDIR, name="%s__selfval" % run_name)
    rec["map50"] = float(r.box.map50)
    rec["map5095"] = float(r.box.map)
    real_print("[E] best.pt 实测 map50=%.6f (%.4f%%)  map5095=%.6f (%.4f%%)"
               % (rec["map50"], rec["map50"] * 100, rec["map5095"], rec["map5095"] * 100),
               flush=True)

    rec["eval_end"] = time.strftime("%Y-%m-%d %H:%M:%S")
    outp = os.path.join(OUTDIR, run_name + ".json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    real_print("[E] wrote %s" % outp, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
