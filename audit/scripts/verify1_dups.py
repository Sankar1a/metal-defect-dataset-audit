#!/usr/bin/env python3
"""Independent blind verification: data-integrity audit (3 datasets) + 18-run cross-check.
Reads ONLY: dataset image files + YOLO label txt files; run dirs' args.yaml + results.csv.
Writes ONLY: verify_20260805/{all_image_hashes.csv, dup_groups.csv, cross_split_dups.csv,
verify_18rows.csv, verify_env.txt, conflict_pairs/*.png} + terminal counts.
No reading of audit/summary/markdown files. No modifications to runs_unified_20260803/.
"""
import os, sys, csv, hashlib, datetime, glob, shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

ROOT = '/home/DSJ44/paper2_yolov11_steel'
VERIFY = os.path.join(ROOT, 'verify_20260805')
RUNS = os.path.join(ROOT, 'runs_unified_20260803')
CPDIR = os.path.join(VERIFY, 'conflict_pairs')

DATASETS = {
    'NEU-DET': {
        'root': '/home/DSJ44/sr_project/yolo_dataset_original',
        'names': {0:'crazing',1:'inclusion',2:'patches',3:'pitted_surface',4:'rolled-in_scale',5:'scratches'},
    },
    'GC10-DET': {
        'root': '/home/DSJ44/sr_project/gc10_yolo',
        'names': {0:'chongkong',1:'hanfeng',2:'yueyawan',3:'shuiban',4:'youban',5:'siban',6:'yiwu',7:'yahen',8:'zhehen',9:'yaozhe'},
    },
    'APSPC': {
        'root': '/home/DSJ44/paper2_yolov11_steel/datasets/apspc_clean1711_seed0',
        'names': {0:'aoxian',1:'budaodian',2:'cahua',3:'jupi',4:'loudi',5:'pengshang',6:'qikeng',7:'tucengkailie',8:'tufen',9:'zangdian'},
    },
}

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 16), b''):
            h.update(chunk)
    return h.hexdigest()

def parse_labels(txt_path):
    rows = []
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            rows.append((int(float(parts[0])), float(parts[1]), float(parts[2]),
                         float(parts[3]), float(parts[4])))
    return sorted(rows)

def same_labels(a, b):
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        for u, v in zip(x, y):
            if abs(u - v) > 1e-6:
                return False
    return True

def discover_splits(images_dir):
    splits = []
    if not os.path.isdir(images_dir):
        return splits
    for d in sorted(os.listdir(images_dir)):
        dd = os.path.join(images_dir, d)
        if os.path.isdir(dd):
            files = [f for f in os.listdir(dd) if f.lower().endswith(IMG_EXTS)]
            if files:
                splits.append(d)
    return splits

# ---------------- Line A ----------------
all_rows = []
dup_groups = []   # list of dicts
summary_counts = {}

for ds, info in DATASETS.items():
    root = info['root']
    images_dir = os.path.join(root, 'images')
    splits = discover_splits(images_dir)
    # A1 hashes
    per_ds_rows = []
    for split in splits:
        sdir = os.path.join(images_dir, split)
        for fn in sorted(os.listdir(sdir)):
            if not fn.lower().endswith(IMG_EXTS):
                continue
            ap = os.path.join(sdir, fn)
            rel = os.path.join('images', split, fn)
            per_ds_rows.append({
                'dataset': ds, 'split': split, 'relative_path': rel,
                'sha256': sha256_file(ap), 'filesize_bytes': os.path.getsize(ap),
            })
    all_rows.extend(per_ds_rows)
    # A2 dup groups by sha256
    by_sha = {}
    for r in per_ds_rows:
        by_sha.setdefault(r['sha256'], []).append(r)
    groups = [v for v in by_sha.values() if len(v) >= 2]

    n_conf = n_ident = n_miss = 0
    for g in groups:
        splits_in = sorted({r['split'] for r in g})
        rels = [r['relative_path'] for r in g]
        # locate labels
        parsed = []
        missing = False
        for r in g:
            stem = os.path.splitext(os.path.basename(r['relative_path']))[0]
            lab = os.path.join(root, 'labels', r['split'], stem + '.txt')
            if not os.path.exists(lab):
                missing = True
                parsed.append(None)
            else:
                parsed.append(parse_labels(lab))
        if missing:
            status = 'missing'
            n_miss += 1
        elif all(same_labels(parsed[0], p) for p in parsed[1:]):
            status = 'identical'
            n_ident += 1
        else:
            status = 'conflicting'
            n_conf += 1
        dup_groups.append({
            'dataset': ds, 'sha256': g[0]['sha256'], 'n_files': len(g),
            'splits_involved': ';'.join(splits_in),
            'file_paths': ';'.join(rels),
            'label_status': status,
        })

    summary_counts[ds] = {
        'images_total': len(per_ds_rows),
        'dup_groups': len(groups),
        'dup_groups_conflicting': n_conf,
        'dup_groups_identical': n_ident,
        'dup_groups_missing_label': n_miss,
    }

# A4 cross-split dups
cross_rows = []
for g in dup_groups:
    sp = [s for s in g['splits_involved'].split(';') if s]
    if len(sp) >= 2:
        cross_rows.append({**g, 'split_pair': ';'.join(sorted(sp))})

# write A outputs
with open(os.path.join(VERIFY, 'all_image_hashes.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['dataset','split','relative_path','sha256','filesize_bytes'])
    w.writeheader(); w.writerows(all_rows)

with open(os.path.join(VERIFY, 'dup_groups.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['dataset','sha256','n_files','splits_involved','file_paths','label_status'])
    w.writeheader(); w.writerows(dup_groups)

with open(os.path.join(VERIFY, 'cross_split_dups.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['dataset','sha256','n_files','splits_involved','file_paths','label_status','split_pair'])
    w.writeheader(); w.writerows(cross_rows)

# A5 conflict PNGs
conf_groups = [g for g in dup_groups if g['label_status'] == 'conflicting']
os.makedirs(CPDIR, exist_ok=True)
for g in conf_groups:
    ds = g['dataset']; root = DATASETS[ds]['root']; names = DATASETS[ds]['names']
    rels = g['file_paths'].split(';')
    n = len(rels)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.6), dpi=110)
    if n == 1:
        axes = [axes]
    for ax, rel in zip(axes, rels):
        img = mpimg.imread(os.path.join(root, rel))
        ax.imshow(img)
        H, W = img.shape[0], img.shape[1]
        stem = os.path.splitext(os.path.basename(rel))[0]
        split = rel.split('/')[1]
        lab = os.path.join(root, 'labels', split, stem + '.txt')
        if os.path.exists(lab):
            for (cid, cx, cy, bw, bh) in parse_labels(lab):
                x1, y1 = (cx - bw/2) * W, (cy - bh/2) * H
                wpx, hpx = bw * W, bh * H
                ax.add_patch(plt.Rectangle((x1, y1), wpx, hpx, fill=False, edgecolor='lime', lw=1.5))
                lbl = names.get(cid, str(cid))
                ax.annotate(lbl, (x1, y1), color='yellow', fontsize=7,
                            xytext=(2, 2), textcoords='offset points')
        ax.set_xlabel(rel, fontsize=7)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(os.path.join(CPDIR, f'{ds}_{g["sha256"][:12]}.png'))
    plt.close(fig)

# A6 terminal print
for ds, c in summary_counts.items():
    print(f"{ds}: images_total={c['images_total']}, dup_groups={c['dup_groups']}, "
          f"dup_groups_conflicting={c['dup_groups_conflicting']}, "
          f"dup_groups_identical={c['dup_groups_identical']}, "
          f"dup_groups_missing_label={c['dup_groups_missing_label']}, "
          f"cross_split_dup_groups={c.get('cross_split_dup_groups', sum(1 for x in cross_rows if x['dataset']==ds))}")

# ---------------- Line B ----------------
all_subdirs = sorted(d for d in os.listdir(RUNS) if os.path.isdir(os.path.join(RUNS, d)))
missing_any = []
run_dirs = []
for d in all_subdirs:
    has_args = os.path.exists(os.path.join(RUNS, d, 'args.yaml'))
    has_res = os.path.exists(os.path.join(RUNS, d, 'results.csv'))
    if has_args and has_res:
        run_dirs.append(d)
    else:
        missing_any.append((d, 'args' if not has_args else 'ok', 'results' if not has_res else 'ok'))
print(f"B: run_dirs_with_args_and_results={len(run_dirs)}")
for md, sa, sr in missing_any:
    print(f"B: RUN_DIR_MISSING_FILE {md} args={sa} results={sr}")

import yaml
b_rows = []
for rd in run_dirs:
    with open(os.path.join(RUNS, rd, 'args.yaml')) as f:
        a = yaml.safe_load(f) or {}
    def g(k):
        return a.get(k)
    # results.csv
    with open(os.path.join(RUNS, rd, 'results.csv')) as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        norm = [''.join(h.split()) for h in header]   # strip spaces to tolerate ' (B)'
        data = list(reader)
    epochs_actual = len(data)
    def col(normalized_name):
        for h, nn in zip(header, norm):
            if nn == normalized_name:
                return h
        return None
    c50 = col('metrics/mAP50(B)')
    c5095 = col('metrics/mAP50-95(B)')
    last = data[-1] if data else []
    idx = {h: i for i, h in enumerate(header)}
    mAP50 = float(last[idx[c50]].strip()) if c50 is not None and idx.get(c50) is not None and len(last) > idx[c50] else None
    mAP5095 = float(last[idx[c5095]].strip()) if c5095 is not None and idx.get(c5095) is not None and len(last) > idx[c5095] else None
    b_rows.append({
        'run_name': g('name') or rd,
        'model': g('model'), 'data': g('data'), 'imgsz': g('imgsz'),
        'epochs': g('epochs'), 'batch': g('batch'), 'patience': g('patience'),
        'seed': g('seed'), 'resume': g('resume'), 'exist_ok': g('exist_ok'), 'device': g('device'),
        'epochs_actual': epochs_actual,
        'mAP50': mAP50, 'mAP50-95': mAP5095,
    })

with open(os.path.join(VERIFY, 'verify_18rows.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['run_name','model','data','imgsz','epochs','batch','patience',
                                      'seed','resume','exist_ok','device','epochs_actual','mAP50','mAP50-95'])
    w.writeheader(); w.writerows(b_rows)

print(f"B: rows_written={len(b_rows)}")
for r in b_rows:
    print(f"{r['run_name']}, {r['seed']}, {r['epochs_actual']}, {r['mAP50']}, {r['mAP50-95']}")

# ---------------- Line C ----------------
import torch, ultralytics
script_sha = sha256_file(os.path.abspath(__file__))
with open(os.path.join(VERIFY, 'verify_env.txt'), 'w') as f:
    f.write(f"python_version={sys.version}\n")
    f.write(f"torch_version={torch.__version__}\n")
    f.write(f"ultralytics_version={ultralytics.__version__}\n")
    f.write(f"script_sha256={script_sha}\n")
    f.write(f"timestamp={datetime.datetime.now().isoformat()}\n")
    for ds, info in DATASETS.items():
        f.write(f"dataset_root[{ds}]={info['root']}\n")

print("C: verify_env.txt written")
