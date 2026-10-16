#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Round 20: train ONE YOLOv8 run from its YOLO11 template args.yaml.

做法沿用第十七轮 round17_yolo11m_20261005/scripts/r17_train_one.py
(该批 9 次训练参数闸门全部 PASS):读模板 args.yaml -> 修改字段 -> 调用 YOLO(...).train(...)。
本脚本只改三个字段: model -> yolov8n/yolov8s, project -> round20 目录, name -> 11n/11s 改 v8n/v8s。
data 与 seed 等其余全部字段原样取自模板,不做任何改动。

Usage: r20_train_one.py <run_name>
"""
import argparse
import os
import sys
import time

import yaml

ROOT = "/home/DSJ44/paper2_yolov11_steel"
EXP = os.path.join(ROOT, "round20_yolov8_20261008")
TEMPLATE_ROOT = os.path.join(ROOT, "runs_unified_20260803")

# 这三个键不作为 train() 的 kwargs 传入(由程序自行管理/派生)
EXCLUDE = {"task", "mode", "save_dir", "pretrained"}

# run 名中的尺寸标记 -> 权重文件名(权重放在本轮输出目录内)
WEIGHT_OF = {"yolov8n": "yolov8n.pt", "yolov8s": "yolov8s.pt"}


def size_key(run_name):
    for k in ("yolov8n", "yolov8s"):
        if k in run_name:
            return k
    raise ValueError("run_name 中找不到 yolov8n/yolov8s: %s" % run_name)


def weight_path(run_name):
    return os.path.join(EXP, WEIGHT_OF[size_key(run_name)])


def template_path(run_name):
    """模板 run 名 = 本次 run 名中的 yolov8n->yolo11n、yolov8s->yolo11s。"""
    tname = run_name.replace("yolov8n", "yolo11n").replace("yolov8s", "yolo11s")
    return os.path.join(TEMPLATE_ROOT, tname, "args.yaml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_name")
    ap.add_argument("--project", default=EXP)
    ap.add_argument("--log", default=None)
    ap.add_argument("--name-override", default=None,
                    help="空跑时用于改 save_dir 名(加 dryrun_ 前缀)")
    a = ap.parse_args()

    run_name = a.run_name
    tpath = template_path(run_name)
    logf = a.log or os.path.join(EXP, "logs", run_name + ".log")
    os.makedirs(os.path.dirname(logf), exist_ok=True)

    with open(tpath, "r", encoding="utf-8") as f:
        tmpl = yaml.safe_load(f)

    weight = weight_path(run_name)

    # ---- 只允许修改的三个字段 ----
    tmpl["model"] = weight
    tmpl["project"] = a.project
    tmpl["name"] = a.name_override or run_name

    kwargs = {k: v for k, v in tmpl.items() if k not in EXCLUDE}

    from ultralytics import YOLO

    print("[R20] ================================================", flush=True)
    print("[R20] start %s" % time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    print("[R20] run_name=%s" % run_name, flush=True)
    print("[R20] template=%s" % tpath, flush=True)
    print("[R20] model=%s exists=%s" % (weight, os.path.exists(weight)), flush=True)
    print("[R20] data=%s" % tmpl.get("data"), flush=True)
    print("[R20] seed=%s" % tmpl.get("seed"), flush=True)
    print("[R20] project=%s" % a.project, flush=True)
    print("[R20] log=%s" % logf, flush=True)
    print("[R20] 只改三个字段: model / project / name;其余字段原样继承模板 args.yaml", flush=True)

    t0 = time.time()
    status = "OK"
    try:
        m = YOLO(weight)
        m.train(**kwargs)
    except Exception as e:
        status = "FAILED: %r" % (e,)
        import traceback
        traceback.print_exc()

    dt = time.time() - t0
    print("[R20] end %s" % time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    print("[R20] exit_status=%s" % status, flush=True)
    print("[R20] elapsed_sec=%.1f" % dt, flush=True)
    print("[R20] save_dir=%s" % os.path.join(a.project, tmpl["name"]), flush=True)
    return 0 if status == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
