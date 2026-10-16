#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Round 21: train ONE YOLOv8n run from its YOLO11n template args.yaml.

做法沿用第二十轮 round20_yolov8_20261008/scripts/r20_train_one.py
(读模板 args.yaml -> 修改字段 -> 调用 YOLO(...).train(...))。
本脚本只改三个字段:
    model   -> 本轮目录内 yolov8n.pt 的绝对路径
    project -> round21_v8data_20261009
    name    -> 模板 run 名中的 yolo11n 改为 yolov8n
data / seed 及其余全部字段原样取自模板 args.yaml,不做任何改动。

Usage: r21_train_one.py <run_name>
"""
import argparse
import os
import sys
import time

import yaml

ROOT = "/home/DSJ44/paper2_yolov11_steel"
EXP = os.path.join(ROOT, "round21_v8data_20261009")
WEIGHT = os.path.join(EXP, "yolov8n.pt")

TEMPLATE_ROOT = {
    "A_arm": os.path.join(ROOT, "exp_apspc_dose_unified_20260915", "runs"),
    "XML": os.path.join(ROOT, "exp_gc10_label_20260914", "runs"),
    "DIR": os.path.join(ROOT, "exp_gc10_label_20260914", "runs"),
}

# (group, condition, seed, template_run, run_name)
JOBS = [
    ("A", "A_arm", 0, "apspc_yolo11n_leaktrain_unified_s0", "apspc_yolov8n_leaktrain_unified_s0"),
    ("A", "A_arm", 1, "apspc_yolo11n_leaktrain_unified_s1", "apspc_yolov8n_leaktrain_unified_s1"),
    ("A", "A_arm", 42, "apspc_yolo11n_leaktrain_unified_s42", "apspc_yolov8n_leaktrain_unified_s42"),
    ("B", "XML", 0, "gc10_yolo11n_xml_s0", "gc10_yolov8n_xml_s0"),
    ("B", "XML", 1, "gc10_yolo11n_xml_s1", "gc10_yolov8n_xml_s1"),
    ("B", "XML", 42, "gc10_yolo11n_xml_s42", "gc10_yolov8n_xml_s42"),
    ("B", "DIR", 0, "gc10_yolo11n_dir_s0", "gc10_yolov8n_dir_s0"),
    ("B", "DIR", 1, "gc10_yolo11n_dir_s1", "gc10_yolov8n_dir_s1"),
    ("B", "DIR", 42, "gc10_yolo11n_dir_s42", "gc10_yolov8n_dir_s42"),
]

BY_RUN = {j[4]: j for j in JOBS}

# 这些键不作为 train() 的 kwargs 传入(由程序自行管理/派生)
EXCLUDE = {"task", "mode", "save_dir", "pretrained"}


def job_of(run_name):
    if run_name not in BY_RUN:
        raise ValueError("未知 run_name: %s" % run_name)
    return BY_RUN[run_name]


def template_path(run_name):
    group, cond, seed, tname, _ = job_of(run_name)
    return os.path.join(TEMPLATE_ROOT[cond], tname, "args.yaml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_name")
    ap.add_argument("--project", default=EXP)
    ap.add_argument("--log", default=None)
    ap.add_argument("--name-override", default=None,
                    help="空跑时用于改 save_dir 名(加 dryrun_ 前缀)")
    a = ap.parse_args()

    run_name = a.run_name
    group, cond, seed, tname, _ = job_of(run_name)
    tpath = template_path(run_name)
    logf = a.log or os.path.join(EXP, "logs", run_name + ".log")
    os.makedirs(os.path.dirname(logf), exist_ok=True)

    with open(tpath, "r", encoding="utf-8") as f:
        tmpl = yaml.safe_load(f)

    # ---- 只允许修改的三个字段 ----
    tmpl["model"] = WEIGHT
    tmpl["project"] = a.project
    tmpl["name"] = a.name_override or run_name

    kwargs = {k: v for k, v in tmpl.items() if k not in EXCLUDE}

    from ultralytics import YOLO

    print("[R21] ================================================", flush=True)
    print("[R21] start %s" % time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    print("[R21] run_name=%s group=%s condition=%s seed=%s" % (run_name, group, cond, seed), flush=True)
    print("[R21] template=%s" % tpath, flush=True)
    print("[R21] template_run=%s" % tname, flush=True)
    print("[R21] model=%s exists=%s" % (WEIGHT, os.path.exists(WEIGHT)), flush=True)
    print("[R21] data=%s" % tmpl.get("data"), flush=True)
    print("[R21] seed=%s" % tmpl.get("seed"), flush=True)
    print("[R21] project=%s" % a.project, flush=True)
    print("[R21] log=%s" % logf, flush=True)
    print("[R21] 只改三个字段: model / project / name;其余字段原样继承模板 args.yaml", flush=True)

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
    print("[R21] end %s" % time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    print("[R21] exit_status=%s" % status, flush=True)
    print("[R21] elapsed_sec=%.1f" % dt, flush=True)
    print("[R21] save_dir=%s" % os.path.join(a.project, tmpl["name"]), flush=True)
    return 0 if status == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
