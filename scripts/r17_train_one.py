#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Round 17: train ONE YOLO11m run from its YOLO11s template args.yaml.

做法沿用 exp_gc10_label_20260914/scripts/ec_train_one.py(该批 6 次训练参数闸门全部 PASS):
读模板 args.yaml -> 修改字段 -> 调用 YOLO(...).train(...)。
本脚本只改三个字段: model -> yolo11m.pt, project -> round17 目录, name -> 11s 改 11m。
data 与 seed 等其余全部字段原样取自模板,不做任何改动。

Usage: r17_train_one.py <run_name>
"""
import argparse
import os
import sys
import time

import yaml

ROOT = "/home/DSJ44/paper2_yolov11_steel"
EXP = os.path.join(ROOT, "round17_yolo11m_20261005")
TEMPLATE_ROOT = os.path.join(ROOT, "runs_unified_20260803")
WEIGHT = os.path.join(ROOT, "yolo11m.pt")

# 这三个键不作为 train() 的 kwargs 传入(由程序自行管理/派生)
EXCLUDE = {"task", "mode", "save_dir", "pretrained"}


def template_path(run_name):
    """模板 run 名 = 本次 run 名中的 11m 换回 11s。"""
    tname = run_name.replace("11m", "11s")
    return os.path.join(TEMPLATE_ROOT, tname, "args.yaml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_name")
    ap.add_argument("--project", default=EXP)
    ap.add_argument("--log", default=None)
    a = ap.parse_args()

    run_name = a.run_name
    tpath = template_path(run_name)
    logf = a.log or os.path.join(EXP, "logs", run_name + ".log")
    os.makedirs(os.path.dirname(logf), exist_ok=True)

    with open(tpath, "r", encoding="utf-8") as f:
        tmpl = yaml.safe_load(f)

    # ---- 只允许修改的三个字段 ----
    tmpl["model"] = WEIGHT
    tmpl["project"] = a.project
    tmpl["name"] = run_name

    kwargs = {k: v for k, v in tmpl.items() if k not in EXCLUDE}

    from ultralytics import YOLO

    print("[R17] ================================================", flush=True)
    print("[R17] start %s" % time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    print("[R17] run_name=%s" % run_name, flush=True)
    print("[R17] template=%s" % tpath, flush=True)
    print("[R17] model=%s" % WEIGHT, flush=True)
    print("[R17] data=%s" % tmpl.get("data"), flush=True)
    print("[R17] seed=%s" % tmpl.get("seed"), flush=True)
    print("[R17] project=%s" % a.project, flush=True)
    print("[R17] log=%s" % logf, flush=True)
    print("[R17] 只改三个字段: model / project / name;其余字段原样继承模板 args.yaml", flush=True)

    t0 = time.time()
    status = "OK"
    try:
        m = YOLO(WEIGHT)
        m.train(**kwargs)
    except Exception as e:
        status = "FAILED: %r" % (e,)
        import traceback
        traceback.print_exc()

    dt = time.time() - t0
    print("[R17] end %s" % time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    print("[R17] exit_status=%s" % status, flush=True)
    print("[R17] elapsed_sec=%.1f" % dt, flush=True)
    print("[R17] save_dir=%s" % os.path.join(a.project, run_name), flush=True)
    return 0 if status == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
