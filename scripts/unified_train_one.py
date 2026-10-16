#!/usr/bin/env python3
"""Per-run training for the unified-protocol 18-run benchmark.
Usage: PYTHONNOUSERSITE=1 <yolov11 python> train_one.py <DS> <MODEL> <SEED> <RUN_NAME>
DS in {NEU, GC10, APSPC}; MODEL in {yolo11n, yolo11s}; SEED in {0,1,42}.
Protocol = runs/baseline/v11n_neudet_seed0/args.yaml (augmentation v2) verbatim,
except model/data/seed/project/name (and conf set to 0.001 = same as null default).
resume=False, exist_ok=False.
"""
import os, sys

DS = sys.argv[1]
MODEL = sys.argv[2]
SEED = int(sys.argv[3])
RUN_NAME = sys.argv[4]
ROOT = '/home/DSJ44/paper2_yolov11_steel'

DATA = {
    'NEU': os.path.join(ROOT, 'audits/leakage_sensitivity/neu_det_val359.yaml'),
    'GC10': os.path.join(ROOT, 'configs/gc10/gc10_det.yaml'),
    'APSPC': os.path.join(ROOT, 'configs/apspc/apspc_clean1711_seed0_dev.yaml'),
}[DS]
WEIGHTS = os.path.join(ROOT, f'{MODEL}.pt')
PROJECT = os.path.join(ROOT, 'runs_unified_20260803')

from ultralytics import YOLO

kwargs = dict(
    data=DATA, epochs=200, time=None, patience=50, batch=16, imgsz=640,
    save=True, save_period=20, cache=False, device=0, workers=8,
    pretrained=True, optimizer='SGD', verbose=True, seed=SEED, deterministic=True,
    single_cls=False, rect=False, cos_lr=False, close_mosaic=10, resume=False,
    amp=True, fraction=1.0, profile=False, freeze=None, multi_scale=False,
    overlap_mask=True, mask_ratio=4, dropout=0.0, val=True, split='val',
    save_json=False, save_hybrid=False, conf=0.001, iou=0.7, max_det=300,
    half=False, dnn=False, plots=True, agnostic_nms=False, classes=None,
    retina_masks=False, lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005,
    warmup_epochs=3, warmup_momentum=0.8, warmup_bias_lr=0.1, box=7.5, cls=0.5,
    dfl=1.5, pose=12.0, kobj=1.0, nbs=64, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    degrees=10.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0,
    flipud=0.5, fliplr=0.5, bgr=0.0, mosaic=1.0, mixup=0.1, copy_paste=0.0,
    copy_paste_mode='flip', auto_augment='randaugment', erasing=0.4,
    crop_fraction=1.0,
    project=PROJECT, name=RUN_NAME, exist_ok=False,
)

# allow a smoke override: optional 5th arg = epochs (for /tmp validation only)
if len(sys.argv) > 5 and sys.argv[5] == 'SMOKE':
    kwargs['epochs'] = 1
    kwargs['project'] = os.path.join(ROOT, 'runs_unified_20260803', '_smoke')
    kwargs['plots'] = False

print(f'[train_one] DS={DS} MODEL={MODEL} SEED={SEED} NAME={RUN_NAME} '
      f'data={DATA} weights={WEIGHTS} project={kwargs["project"]}', flush=True)
m = YOLO(WEIGHTS)
m.train(**kwargs)
print(f'[train_one] DONE {RUN_NAME}', flush=True)
