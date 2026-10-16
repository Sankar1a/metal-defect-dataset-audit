#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared config for round 16 (APSPC dose experiment, unified protocol).

Every value in TRAIN_KWARGS was read from the ONLY template:
    runs_unified_20260803/apspc_yolo11n_s0/args.yaml
via the round-0.1 grep. Nothing comes from the task-book example table.
"""
import os

ROOT = "/home/DSJ44/paper2_yolov11_steel"
EXP = os.path.join(ROOT, "exp_apspc_dose_unified_20260915")
AUG = os.path.join(ROOT, "exp_apspc_dose_20260817")
TEMPLATE = os.path.join(ROOT, "runs_unified_20260803", "apspc_yolo11n_s0", "args.yaml")

# ---- data yamls ----
YAML_A = os.path.join(AUG, "configs", "apspc_d0.yaml")                       # train_leak, val_d0
YAML_B = os.path.join(ROOT, "configs", "apspc", "apspc_clean1711_seed0_dev.yaml")

# 9 val yamls re-used as-is from the August experiment.
# NOTE: filename rule is "apspc_" + val_set with the leading "val_" stripped:
#   val_d0 -> apspc_d0.yaml ; val_d20_same -> apspc_d20_same.yaml ; ...
# Verified against the actual dir listing of exp_apspc_dose_20260817/configs/.
VAL_SETS = ["val_d0",
            "val_d20_same", "val_d28_same", "val_d41_same", "val_d82_same",
            "val_d20_conf", "val_d28_conf", "val_d41_conf", "val_d82_conf"]
VAL_YAML = {v: os.path.join(AUG, "configs", "apspc_%s.yaml" % v[len("val_"):])
            for v in VAL_SETS}

SEEDS = [0, 1, 42]

ARM_RUNS = [
    # (arm, seed, data_yaml, run_name)
    ("A_leaktrain", 0, YAML_A, "apspc_yolo11n_leaktrain_unified_s0"),
    ("A_leaktrain", 1, YAML_A, "apspc_yolo11n_leaktrain_unified_s1"),
    ("A_leaktrain", 42, YAML_A, "apspc_yolo11n_leaktrain_unified_s42"),
    ("B_clean_repro", 0, YAML_B, "apspc_yolo11n_clean_repro_s0"),
    ("B_clean_repro", 1, YAML_B, "apspc_yolo11n_clean_repro_s1"),
    ("B_clean_repro", 42, YAML_B, "apspc_yolo11n_clean_repro_s42"),
]

# ---- the 23 fields read from the APSPC template by the 0.1 grep ----
# values are taken verbatim from runs_unified_20260803/apspc_yolo11n_s0/args.yaml
TRAIN_KWARGS = dict(
    imgsz=640, epochs=200, batch=16, patience=50,
    resume=False, exist_ok=False, device=0,
    optimizer="SGD",
    degrees=10.0,
    flipud=0.5,
    mixup=0.1,
    conf=0.001,
    warmup_epochs=3,
    save_period=20,
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    close_mosaic=10,
    mosaic=1.0,
    scale=0.5,
    translate=0.1,
    fliplr=0.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    erasing=0.4,
    box=7.5,
    cls=0.5,
    dfl=1.5,
)

MODEL = os.path.join(ROOT, "yolo11n.pt")
RUNS = os.path.join(EXP, "runs")
