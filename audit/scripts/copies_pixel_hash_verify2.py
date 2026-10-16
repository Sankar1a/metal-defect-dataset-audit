#!/usr/bin/env python3
"""Round-2 independent verification:
D pixel-level duplicate detection, E APSPC copy search, F metric reconciliation,
G environment record. Read-only on datasets and run dirs; writes verify2_20260805/.
Env: yolov11 (ultralytics 8.3.40). No markdown output; CSV/PNG/terminal only.
"""
import os, sys, csv, hashlib, datetime, subprocess, shutil, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import yaml

ROOT = '/home/DSJ44/paper2_yolov11_steel'
VERIFY2 = os.path.join(ROOT, 'verify2_20260805')
RUNS = os.path.join(ROOT, 'runs_unified_20260803')
CPDIR = os.path.join(VERIFY2, 'pixel_conflict_pairs')

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

def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()

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

# ================= Line D =================
all_rows = []        # D1 csv rows
imgmeta = {}         # rel -> (dataset, split, byte_sha, pixel_sha, w, h, mode)
decode_fail = 0
for ds, info in DATASETS.items():
    root = info['root']
    images_dir = os.path.join(root, 'images')
    for split in discover_splits(images_dir):
        sdir = os.path.join(images_dir, split)
        for fn in sorted(os.listdir(sdir)):
            if not fn.lower().endswith(IMG_EXTS):
                continue
            ap = os.path.join(sdir, fn)
            rel = os.path.join('images', split, fn)
            byte_sha = sha256_file(ap)
            try:
                with Image.open(ap) as im:
                    mode_orig = im.mode
                    w, h = im.size
                    rgb = im.convert('RGB')
                    pixel_sha = sha256_bytes(rgb.tobytes())
            except Exception:
                decode_fail += 1
                all_rows.append({'dataset': ds, 'split': split, 'relative_path': rel,
                                 'pixel_sha256': 'DECODE_FAILED', 'width': '', 'height': '', 'mode_original': ''})
                imgmeta[rel] = (ds, split, byte_sha, None, None, None, None)
                continue
            all_rows.append({'dataset': ds, 'split': split, 'relative_path': rel,
                             'pixel_sha256': pixel_sha, 'width': w, 'height': h, 'mode_original': mode_orig})
            imgmeta[rel] = (ds, split, byte_sha, pixel_sha, w, h, mode_orig)

with open(os.path.join(VERIFY2, 'all_pixel_hashes.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['dataset','split','relative_path','pixel_sha256','width','height','mode_original'])
    w.writeheader(); w.writerows(all_rows)

# D2 groups by (dataset, pixel_sha)
by_key = {}
for rel, (ds, split, byte_sha, pixel_sha, w, h, mo) in imgmeta.items():
    if pixel_sha is None:
        continue
    by_key.setdefault((ds, pixel_sha), []).append((rel, split, byte_sha))

dup_rows = []
for (ds, psha), members in by_key.items():
    if len(members) < 2:
        continue
    splits_in = sorted({sp for _, sp, _ in members})
    rels = [r for r, _, _ in members]
    byte_shas = {b for _, _, b in members}
    byte_distinct = len(byte_shas)
    # label status
    root = DATASETS[ds]['root']
    parsed = []; missing = False
    for r, sp, _ in members:
        stem = os.path.splitext(os.path.basename(r))[0]
        lab = os.path.join(root, 'labels', sp, stem + '.txt')
        if not os.path.exists(lab):
            missing = True; parsed.append(None)
        else:
            parsed.append(parse_labels(lab))
    if missing:
        status = 'missing'
    elif all(same_labels(parsed[0], p) for p in parsed[1:]):
        status = 'identical'
    else:
        status = 'conflicting'
    dup_rows.append({
        'dataset': ds, 'pixel_sha256': psha, 'n_files': len(members),
        'splits_involved': ';'.join(splits_in), 'file_paths': ';'.join(rels),
        'byte_hashes_distinct': byte_distinct, 'label_status': status,
    })

with open(os.path.join(VERIFY2, 'pixel_dup_groups.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['dataset','pixel_sha256','n_files','splits_involved','file_paths','byte_hashes_distinct','label_status'])
    w.writeheader(); w.writerows(dup_rows)

# D4 conflict PNGs (cap 20)
conf = [g for g in dup_rows if g['label_status'] == 'conflicting']
os.makedirs(CPDIR, exist_ok=True)
png_count = 0
for g in conf:
    if png_count >= 20:
        break
    ds = g['dataset']; root = DATASETS[ds]['root']; names = DATASETS[ds]['names']
    rels = g['file_paths'].split(';')
    n = len(rels)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.6), dpi=110)
    if n == 1:
        axes = [axes]
    for ax, rel in zip(axes, rels):
        try:
            img = mpimg.imread(os.path.join(root, rel))
            ax.imshow(img)
            H, W = img.shape[0], img.shape[1]
        except Exception:
            ax.text(0.5, 0.5, 'DECODE_FAILED', ha='center', va='center'); H = W = 1
        stem = os.path.splitext(os.path.basename(rel))[0]
        split = rel.split('/')[1]
        lab = os.path.join(root, 'labels', split, stem + '.txt')
        if os.path.exists(lab):
            for (cid, cx, cy, bw, bh) in parse_labels(lab):
                x1, y1 = (cx - bw/2) * W, (cy - bh/2) * H
                ax.add_patch(plt.Rectangle((x1, y1), bw*W, bh*H, fill=False, edgecolor='lime', lw=1.5))
                ax.annotate(names.get(cid, str(cid)), (x1, y1), color='yellow', fontsize=7,
                            xytext=(2, 2), textcoords='offset points')
        ax.set_xlabel(rel, fontsize=7)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(os.path.join(CPDIR, f'{ds}_{g["pixel_sha256"][:12]}.png'))
    plt.close(fig)
    png_count += 1

# D5 terminal
for ds in DATASETS:
    dd = [g for g in dup_rows if g['dataset'] == ds]
    n_conf = sum(1 for g in dd if g['label_status'] == 'conflicting')
    n_ident = sum(1 for g in dd if g['label_status'] == 'identical')
    n_miss = sum(1 for g in dd if g['label_status'] == 'missing')
    n_cross = sum(1 for g in dd if len(g['splits_involved'].split(';')) >= 2)
    n_reenc = sum(1 for g in dd if g['byte_hashes_distinct'] > 1)
    print(f"{ds}: pixel_dup_groups={len(dd)}, conflicting={n_conf}, identical={n_ident}, "
          f"missing_label={n_miss}, cross_split={n_cross}, groups_with_reencoding={n_reenc}")

# ================= Line E =================
def count_images(path):
    n = 0
    for dp, _, files in os.walk(path):
        n += sum(1 for f in files if f.lower().endswith(IMG_EXTS))
    return n

def dir_size(path):
    return subprocess.run(['du', '-sb', path], capture_output=True, text=True).stdout.split()[0]

SEARCH_ROOTS = ['/home/DSJ44', '/tmp']
candidates = []
for sroot in SEARCH_ROOTS:
    out = subprocess.run(['find', sroot, '-type', 'd', '-iname', '*apspc*'], capture_output=True, text=True).stdout
    for line in out.splitlines():
        p = line.strip()
        if not p:
            continue
        comps = p.split('/')
        if any(c.startswith('runs_') for c in comps):
            continue
        if p not in candidates:
            candidates.append(p)
candidates.sort()
e_lines = []
for p in candidates:
    n_img = count_images(p)
    try:
        sz = dir_size(p)
    except Exception:
        sz = 'N/A'
    mtime = datetime.datetime.fromtimestamp(os.path.getmtime(p)).isoformat()
    line = f"{p}\timages={n_img}\tsize_bytes={sz}\tlast_modified={mtime}"
    e_lines.append(line)
    print("E:", line)
with open(os.path.join(VERIFY2, 'apspc_copies.txt'), 'w') as f:
    f.write("search_roots=" + ",".join(SEARCH_ROOTS) + "\n")
    f.write("excluded=runs_* directories\n")
    for l in e_lines:
        f.write(l + "\n")

# ================= Line F =================
from ultralytics import YOLO
run_dirs = sorted(d for d in os.listdir(RUNS) if os.path.isdir(os.path.join(RUNS, d)))
valid = []
for d in run_dirs:
    if os.path.exists(os.path.join(RUNS, d, 'args.yaml')) and os.path.exists(os.path.join(RUNS, d, 'results.csv')):
        valid.append(d)
    else:
        print(f"F: RUN_DIR_MISSING_FILE {d}")
print(f"F: run_dirs_with_args_and_results={len(valid)}")

# pre-check data yaml test keys
yaml_test_check = {}
for d in valid:
    with open(os.path.join(RUNS, d, 'args.yaml')) as f:
        a = yaml.safe_load(f) or {}
    dy = a.get('data')
    if dy:
        with open(dy) as f:
            dcfg = yaml.safe_load(f) or {}
        t = dcfg.get('test')
        yaml_test_check[d] = (dy, t)
        if t:
            print(f"F: DATA_YAML_TEST_KEY_NONEMPTY {d} data={dy} test={t}")

f_rows = []
for d in valid:
    run_dir = os.path.join(RUNS, d)
    with open(os.path.join(run_dir, 'args.yaml')) as f:
        a = yaml.safe_load(f) or {}
    seed = a.get('seed'); data = a.get('data'); imgsz = a.get('imgsz')
    # results.csv
    with open(os.path.join(run_dir, 'results.csv')) as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        norm = [''.join(h.split()) for h in header]
        rows = list(reader)
    epochs_total = len(rows)
    def col(nn):
        for h, n2 in zip(header, norm):
            if n2 == nn:
                return h
        return None
    c50 = col('metrics/mAP50(B)'); c5095 = col('metrics/mAP50-95(B)'); cep = col('epoch')
    idx = {h: i for i, h in enumerate(header)}
    def val(row, cn):
        if cn is None or idx.get(cn) is None or len(row) <= idx[cn]:
            return None
        v = row[idx[cn]].strip()
        try:
            return float(v)
        except ValueError:
            return None
    last = rows[-1]
    last50 = val(last, c50); last5095 = val(last, c5095)
    # best row by mAP50-95
    best_row = None; best_epoch = None
    for r in rows:
        v95 = val(r, c5095)
        if v95 is None:
            continue
        if best_row is None or v95 > val(best_row, c5095):
            best_row = r; best_epoch = val(r, cep) if cep is not None and idx.get(cep) is not None and len(r) > idx[cep] else len(rows) - rows.index(r)
    if best_row is None:
        best_row = rows[-1]; best_epoch = None
    best_epoch = val(best_row, cep)
    best50 = val(best_row, c50); best5095 = val(best_row, c5095)
    # F3 val
    bestpt = os.path.join(run_dir, 'weights', 'best.pt')
    v50 = v5095 = None; val_note = ''
    if os.path.exists(bestpt):
        dy_cfg_test = yaml_test_check.get(d, (None, None))[1]
        if dy_cfg_test:
            v50 = v5095 = None; val_note = 'SKIPPED_TEST_KEY_NONEMPTY'
        else:
            m = YOLO(bestpt)
            r = m.val(data=data, imgsz=imgsz, split='val', batch=16, device=0, verbose=False,
                      project='/tmp/verify2_f_val', name=d)
            v50 = float(r.box.map50); v5095 = float(r.box.map)
    else:
        val_note = 'BEST_PT_MISSING'
    f_rows.append({
        'run_name': d, 'seed': seed, 'epochs_total': epochs_total,
        'last_map50': last50, 'last_map5095': last5095,
        'best_row_epoch': best_epoch, 'best_row_map50': best50, 'best_row_map5095': best5095,
        'bestpt_val_map50': v50, 'bestpt_val_map5095': v5095, 'note': val_note,
    })

with open(os.path.join(VERIFY2, 'metric_reconciliation.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['run_name','seed','epochs_total','last_map50','last_map5095',
                                      'best_row_epoch','best_row_map50','best_row_map5095',
                                      'bestpt_val_map50','bestpt_val_map5095','note'])
    w.writeheader(); w.writerows(f_rows)
print("F: rows_written=%d" % len(f_rows))
for r in f_rows:
    print(f"F: {r['run_name']}, seed={r['seed']}, epochs_total={r['epochs_total']}, "
          f"last_map50={r['last_map50']}, last_map5095={r['last_map5095']}, "
          f"best_row_epoch={r['best_row_epoch']}, best_row_map50={r['best_row_map50']}, best_row_map5095={r['best_row_map5095']}, "
          f"bestpt_val_map50={r['bestpt_val_map50']}, bestpt_val_map5095={r['bestpt_val_map5095']}, note={r['note']}")

# ================= Line G =================
import torch, ultralytics
gpu = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip().splitlines()
gpu_model = gpu[0] if gpu else 'N/A'
script_sha = sha256_file(os.path.abspath(__file__))
with open(os.path.join(VERIFY2, 'verify2_env.txt'), 'w') as f:
    f.write(f"python_version={sys.version}\n")
    f.write(f"torch_version={torch.__version__}\n")
    f.write(f"ultralytics_version={ultralytics.__version__}\n")
    f.write(f"script_sha256={script_sha}\n")
    f.write(f"timestamp={datetime.datetime.now().isoformat()}\n")
    f.write(f"gpu_model={gpu_model}\n")
    for ds, info in DATASETS.items():
        f.write(f"dataset_root[{ds}]={info['root']}\n")
print("G: verify2_env.txt written")
