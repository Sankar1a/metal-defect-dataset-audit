#!/usr/bin/env python3
"""Round-3 Line M: metric reconciliation for the 18 runs in runs_unified_20260803/.
Must run in ultralytics 8.3.40 env. M0 env check, M1 last row, M2 best row, M3 best.pt val.
Writes verify3_20260805/metric_reconciliation.csv. split='val' only; skips if data yaml test key non-empty.
"""
import os, sys, csv, yaml

RUNS = '/home/DSJ44/paper2_yolov11_steel/runs_unified_20260803'
VERIFY3 = '/home/DSJ44/paper2_yolov11_steel/verify3_20260805'

# M0 env check
import torch, ultralytics
print(f"M0 python={sys.version.split()[0]} torch={torch.__version__} ultralytics={ultralytics.__version__}")
if ultralytics.__version__ != '8.3.40':
    r = os.popen('conda env list 2>/dev/null || echo no-conda').read()
    print("M0 ENV_MISMATCH_STOP current_ultralytics=%s\n%s" % (ultralytics.__version__, r))
    sys.exit(2)

from ultralytics import YOLO

run_dirs = sorted(d for d in os.listdir(RUNS) if os.path.isdir(os.path.join(RUNS, d)))
valid = [d for d in run_dirs
         if os.path.exists(os.path.join(RUNS, d, 'args.yaml')) and os.path.exists(os.path.join(RUNS, d, 'results.csv'))]
print(f"M1 run_dirs_with_args_and_results={len(valid)}")

def col_index(header, norm, name):
    for h, n2 in zip(header, norm):
        if n2 == name:
            return h
    return None

f_rows = []
for d in valid:
    rd = os.path.join(RUNS, d)
    with open(os.path.join(rd, 'args.yaml')) as f:
        a = yaml.safe_load(f) or {}
    seed = a.get('seed'); data = a.get('data'); imgsz = a.get('imgsz')
    # data yaml test-key pre-check
    test_val = None
    if data:
        with open(data) as f:
            dcfg = yaml.safe_load(f) or {}
        test_val = dcfg.get('test')
    # results.csv
    with open(os.path.join(rd, 'results.csv')) as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        norm = [''.join(h.split()) for h in header]
        rows = list(reader)
    epochs_total = len(rows)
    idx = {h: i for i, h in enumerate(header)}
    c50 = col_index(header, norm, 'metrics/mAP50(B)')
    c5095 = col_index(header, norm, 'metrics/mAP50-95(B)')
    cep = col_index(header, norm, 'epoch')
    def val(row, cn):
        if cn is None or idx.get(cn) is None or len(row) <= idx[cn]:
            return None
        try:
            return float(row[idx[cn]].strip())
        except ValueError:
            return None
    last = rows[-1]
    last50 = val(last, c50); last5095 = val(last, c5095)
    best_row = max(rows, key=lambda r: val(r, c5095) if val(r, c5095) is not None else -1)
    best50 = val(best_row, c50); best5095 = val(best_row, c5095)
    best_epoch = val(best_row, cep)
    # M3 best.pt val
    bestpt = os.path.join(rd, 'weights', 'best.pt')
    v50 = v5095 = None; note = ''
    if os.path.exists(bestpt):
        if test_val:
            note = 'SKIPPED_TEST_KEY_NONEMPTY'
            print(f"M3 SKIP {d} data={data} test={test_val}")
        else:
            m = YOLO(bestpt)
            r = m.val(data=data, imgsz=imgsz, split='val', batch=16, device=0, verbose=False,
                      project='/tmp/verify3_f_val', name=d)
            v50 = float(r.box.map50); v5095 = float(r.box.map)
    else:
        note = 'BEST_PT_MISSING'
    f_rows.append({
        'run_name': d, 'seed': seed, 'epochs_total': epochs_total,
        'last_map50': last50, 'last_map5095': last5095,
        'best_row_epoch': best_epoch, 'best_row_map50': best50, 'best_row_map5095': best5095,
        'bestpt_val_map50': v50, 'bestpt_val_map5095': v5095, 'note': note,
    })

with open(os.path.join(VERIFY3, 'metric_reconciliation.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['run_name','seed','epochs_total','last_map50','last_map5095',
                                      'best_row_epoch','best_row_map50','best_row_map5095',
                                      'bestpt_val_map50','bestpt_val_map5095','note'])
    w.writeheader(); w.writerows(f_rows)
print(f"M5 rows={len(f_rows)}")
for r in f_rows:
    print(f"M5 {r['run_name']}, seed={r['seed']}, epochs_total={r['epochs_total']}, "
          f"last_map50={r['last_map50']}, last_map5095={r['last_map5095']}, "
          f"best_row_epoch={r['best_row_epoch']}, best_row_map50={r['best_row_map50']}, best_row_map5095={r['best_row_map5095']}, "
          f"bestpt_val_map50={r['bestpt_val_map50']}, bestpt_val_map5095={r['bestpt_val_map5095']}, note={r['note']}")
