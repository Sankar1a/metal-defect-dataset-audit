#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Round 16: train ONE run. Called as a subprocess by run_all.py.

Usage: train_one.py <run_name> <seed> <arm> <data_yaml>
"""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import EXP, MODEL, RUNS, TRAIN_KWARGS, TEMPLATE

LOGS = os.path.join(EXP, "logs")


def main():
    run_name, seed, arm, data_yaml = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
    os.makedirs(LOGS, exist_ok=True)
    logf = os.path.join(LOGS, run_name + ".log")

    from ultralytics import YOLO

    print("[T16] ================================================", flush=True)
    print("[T16] run_name=%s arm=%s seed=%d" % (run_name, arm, seed), flush=True)
    print("[T16] start=%s" % time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    print("[T16] model=%s" % MODEL, flush=True)
    print("[T16] data=%s" % data_yaml, flush=True)
    print("[T16] template=%s" % TEMPLATE, flush=True)
    print("[T16] project=%s" % RUNS, flush=True)
    print("[T16] kwargs=%r" % (TRAIN_KWARGS,), flush=True)

    t0 = time.time()
    status = "OK"
    try:
        m = YOLO(MODEL)
        m.train(data=data_yaml, seed=seed, project=RUNS, name=run_name, **TRAIN_KWARGS)
    except Exception as e:
        status = "FAILED: %r" % (e,)
        import traceback
        traceback.print_exc()

    dt = time.time() - t0
    print("[T16] end=%s" % time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    print("[T16] exit_status=%s" % status, flush=True)
    print("[T16] wall_elapsed_sec=%.1f" % dt, flush=True)
    print("[T16] save_dir=%s" % os.path.join(RUNS, run_name), flush=True)
    return 0 if status == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
