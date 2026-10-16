#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Round 22 第5步 · 训练单个 run。

读模板 args.yaml -> 只改第2步表格允许的字段 -> 调 RTDETR(...).train(...)。

相对模板修改的字段(其余全部原样继承,data 与 seed 不动):
    model          -> 本轮目录内 rtdetr-l.pt 的绝对路径
    project / name -> 本轮目录;name 中 yolo11n 改为 rtdetr-l
    optimizer      -> AdamW(显式指定,严禁 auto)
    lr0            -> 0.0001
    momentum       -> 0.9
    weight_decay   -> 0.0001
    epochs / batch -> 取自第3步 decision.txt

模型类说明: rtdetr-l.pt 是 Transformer 检测器,必须走 RTDETR 的 trainer/validator
(YOLO(rtdetr-l.pt) 会套用 YOLO 的 DetectionTrainer/DetectionValidator);
RTDETR(...).train() 即 RT-DETR 的真实 train() 路径。

用法: r22_train_one.py <run_name> [--project DIR] [--log FILE] [--name-override NAME]
"""
import argparse
import os
import sys
import time

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r22_common import (EXCLUDE, EXP, FIXED_MODS, RTDETR_WEIGHT,  # noqa: E402
                        job_of, read_decision, template_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_name")
    ap.add_argument("--project", default=EXP)
    ap.add_argument("--log", default=None)
    ap.add_argument("--name-override", default=None,
                    help="空跑时用于改 save_dir 名(加 dryrun_ 前缀)")
    a = ap.parse_args()

    run_name = a.run_name
    ds, cond, seed, tname, _ = job_of(run_name)
    tpath = template_path(run_name)
    logf = a.log or os.path.join(EXP, "logs", run_name + ".log")
    os.makedirs(os.path.dirname(logf), exist_ok=True)

    dec = read_decision()

    with open(tpath, "r", encoding="utf-8") as f:
        m = yaml.safe_load(f)

    m["model"] = RTDETR_WEIGHT
    m["project"] = a.project
    m["name"] = a.name_override or run_name
    m.update(FIXED_MODS)
    m["epochs"] = int(dec["epochs"])
    m["batch"] = int(dec["batch"])

    kwargs = {k: v for k, v in m.items() if k not in EXCLUDE}

    from ultralytics import RTDETR

    print("[R22] ================================================", flush=True)
    print("[R22] start %s" % time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    print("[R22] run_name=%s dataset=%s condition=%s seed=%s" % (run_name, ds, cond, seed),
          flush=True)
    print("[R22] template=%s" % tpath, flush=True)
    print("[R22] template_run=%s" % tname, flush=True)
    print("[R22] model=%s exists=%s" % (RTDETR_WEIGHT, os.path.exists(RTDETR_WEIGHT)), flush=True)
    print("[R22] data=%s(不得改)" % m.get("data"), flush=True)
    print("[R22] seed=%s(不得改)" % m.get("seed"), flush=True)
    print("[R22] project=%s" % a.project, flush=True)
    print("[R22] name=%s" % m["name"], flush=True)
    print("[R22] optimizer=%s lr0=%s momentum=%s weight_decay=%s epochs=%s batch=%s"
          % (m["optimizer"], m["lr0"], m["momentum"], m["weight_decay"],
             m["epochs"], m["batch"]), flush=True)
    print("[R22] log=%s" % logf, flush=True)
    print("[R22] 只改第2步表格允许的字段: model/project/name/save_dir/optimizer/"
          "lr0/momentum/weight_decay/epochs/batch", flush=True)
    print("[R22] 模型类: ultralytics.RTDETR(RT-DETR 真实 train() 路径)", flush=True)

    t0 = time.time()
    status = "OK"
    try:
        RTDETR(RTDETR_WEIGHT).train(**kwargs)
    except BaseException as e:                                  # noqa: BLE001
        status = "FAILED: %r" % (e,)
        import traceback
        traceback.print_exc()

    dt = time.time() - t0
    print("[R22] end %s" % time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    print("[R22] exit_status=%s" % status, flush=True)
    print("[R22] elapsed_sec=%.1f" % dt, flush=True)
    print("[R22] save_dir=%s" % os.path.join(a.project, m["name"]), flush=True)
    return 0 if status == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
