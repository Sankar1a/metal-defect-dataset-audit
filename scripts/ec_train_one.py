#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Line EC: train ONE run (cond, seed) for the GC10 label-ambiguity experiment.

Called as a subprocess by ec_run_all.py so each training gets a fresh process.

Usage: ec_train_one.py <cond> <seed> [--epochs N] [--project P] [--log P]
"""
import argparse
import os
import sys
import time

ROOT = "/home/DSJ44/paper2_yolov11_steel"
EXP = os.path.join(ROOT, "exp_gc10_label_20260914")

COND_YAML = {
    "xml": os.path.join(EXP, "configs", "gc10_xml.yaml"),
    "dir": os.path.join(EXP, "configs", "gc10_dir.yaml"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cond", choices=["xml", "dir"])
    ap.add_argument("seed", type=int)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--project", default=os.path.join(EXP, "runs"))
    ap.add_argument("--log", default=None)
    a = ap.parse_args()

    name = "gc10_yolo11n_%s_s%d" % (a.cond, a.seed)
    logf = a.log or os.path.join(EXP, "logs", name + ".log")
    os.makedirs(os.path.dirname(logf), exist_ok=True)

    from ultralytics import YOLO

    print("[EC] ================================================", flush=True)
    print("[EC] start %s" % time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    print("[EC] run_name=%s cond=%s seed=%d epochs=%d" % (name, a.cond, a.seed, a.epochs), flush=True)
    print("[EC] data=%s" % COND_YAML[a.cond], flush=True)
    print("[EC] model=%s" % os.path.join(ROOT, "yolo11n.pt"), flush=True)
    print("[EC] project=%s" % a.project, flush=True)
    print("[EC] 统一基准参数: imgsz=640 batch=16 patience=50 resume=False exist_ok=False device=0", flush=True)
    print("[EC] 补充说明新增七项: optimizer=SGD degrees=10.0 flipud=0.5 mixup=0.1 conf=0.001 warmup_epochs=3 save_period=20", flush=True)
    print("[EC] log=%s" % logf, flush=True)

    t0 = time.time()
    status = "OK"
    try:
        m = YOLO(os.path.join(ROOT, "yolo11n.pt"))
        m.train(
            data=COND_YAML[a.cond],
            imgsz=640, epochs=a.epochs, batch=16, patience=50,
            seed=a.seed, resume=False, exist_ok=False, device=0,

            # ↓↓↓ 补充说明 §1 新增七项,与统一基准对齐 ↓↓↓
            optimizer='SGD',
            degrees=10.0,
            flipud=0.5,
            mixup=0.1,
            conf=0.001,
            warmup_epochs=3,
            save_period=20,
            # ↑↑↑ 新增结束 ↑↑↑

            project=a.project,
            name=name,
        )
    except Exception as e:
        status = "FAILED: %r" % (e,)
        import traceback
        traceback.print_exc()

    dt = time.time() - t0
    print("[EC] end %s" % time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    print("[EC] exit_status=%s" % status, flush=True)
    print("[EC] elapsed_sec=%.1f" % dt, flush=True)
    print("[EC] save_dir=%s" % os.path.join(a.project, name), flush=True)
    return 0 if status == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
