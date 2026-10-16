#!/usr/bin/env python3
"""Round-3 Line N: post-hoc snapshot of prior audit files (after L6 credential).
Prints content as-is (no interpretation) and copies to prior_audit_snapshot/.
CSVs: print all rows (both <200 lines). JSONs: print each fully.
"""
import os, shutil, glob

VERIFY3 = '/home/DSJ44/paper2_yolov11_steel/verify3_20260805'
SNAP = os.path.join(VERIFY3, 'prior_audit_snapshot')
os.makedirs(SNAP, exist_ok=True)

csvs = [
    'audits/q2_benchmark_readiness_20260730_v2/duplicate_conflict_summary.csv',
    'audits/gc10_det/duplicate_audit.csv',
]
for f in csvs:
    print(f"=== {f} ===")
    with open(f) as fh:
        lines = fh.read().splitlines()
    if len(lines) > 200:
        print('\n'.join(lines[:21]))
        print(f"... (total lines: {len(lines)})")
    else:
        print('\n'.join(lines))
    shutil.copy2(f, os.path.join(SNAP, os.path.basename(f)))

jsons = sorted(glob.glob('audits/q2_overnight_evidence_20260802_v1/reval_*.json'))
for j in jsons:
    print(f"=== {j} ===")
    with open(j) as fh:
        print(fh.read(), end='')
    shutil.copy2(j, os.path.join(SNAP, os.path.basename(j)))

print("N: prior_audit_snapshot_files=" + ",".join(sorted(os.listdir(SNAP))))
