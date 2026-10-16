#!/usr/bin/env python3
"""Round-2 (revised) Line H: locate dataset copies under /home/DSJ44 + list audits dir file paths.
Writes verify2_20260805/{dataset_copies.csv, audits_filelist.txt}; then STOPS per taskbook.
No audit computation on copies; no opening of any file under audits/.
"""
import os, csv, subprocess, datetime

VERIFY2 = '/home/DSJ44/paper2_yolov11_steel/verify2_20260805'
SEARCH_ROOT = '/home/DSJ44'
KEYWORDS = ['apspc', 'neu', 'gc10', 'steel', 'raw', 'original']
EXCLUDE = ('runs_', 'verify_', '.git', '__pycache__', 'weights', 'cache')
IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')

def prune(name):
    return any(name.startswith(x) or name == x for x in EXCLUDE) or \
           any(x in name for x in ('weights', 'cache'))

# H1: find candidate dirs
expr = ['-type', 'd', '(']
for i, kw in enumerate(KEYWORDS):
    if i:
        expr += ['-o']
    expr += ['-iname', f'*{kw}*']
expr += [')']
out = subprocess.run(['find', SEARCH_ROOT] + expr, capture_output=True, text=True).stdout
cands = [l.strip() for l in out.splitlines()
         if l.strip() and not any(prune(c) for c in l.strip().split('/'))]
cands.sort()

rows = []
for p in cands:
    n_img = 0; n_lbl = 0; mtime_latest = 0
    for dp, dnames, fnames in os.walk(p):
        dnames[:] = [d for d in dnames if not prune(d)]
        for f in fnames:
            low = f.lower()
            if low.endswith(IMG_EXTS):
                n_img += 1
            elif low.endswith('.txt'):
                n_lbl += 1
            try:
                mtime_latest = max(mtime_latest, os.path.getmtime(os.path.join(dp, f)))
            except OSError:
                pass
        try:
            mtime_latest = max(mtime_latest, os.path.getmtime(dp))
        except OSError:
            pass
    try:
        sz = subprocess.run(['du', '-sb', p], capture_output=True, text=True).stdout.split()[0]
    except Exception:
        sz = 'N/A'
    has_images = os.path.isdir(os.path.join(p, 'images'))
    rows.append({
        'absolute_path': p,
        'n_image_files': n_img,
        'n_label_files': n_lbl,
        'total_size_bytes': sz,
        'mtime_latest': datetime.datetime.fromtimestamp(mtime_latest).isoformat() if mtime_latest else 'N/A',
        'depth_has_images_subdir': 'yes' if has_images else 'no',
    })

with open(os.path.join(VERIFY2, 'dataset_copies.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['absolute_path','n_image_files','n_label_files','total_size_bytes','mtime_latest','depth_has_images_subdir'])
    w.writeheader(); w.writerows(rows)

# H2: audits file list (do NOT open files)
out2 = subprocess.run(['find', '/home/DSJ44/paper2_yolov11_steel/audits', '-type', 'f'], capture_output=True, text=True).stdout
files = sorted(l for l in out2.splitlines() if l.strip())
with open(os.path.join(VERIFY2, 'audits_filelist.txt'), 'w') as f:
    f.write('\n'.join(files) + '\n')

# H3: terminal
print('H1: dataset copy search results (rows=%d)' % len(rows))
for r in rows:
    print("H1:", r['absolute_path'], "| images=%d" % r['n_image_files'],
          "| labels=%d" % r['n_label_files'], "| size_bytes=%s" % r['total_size_bytes'],
          "| mtime_latest=%s" % r['mtime_latest'],
          "| has_images_subdir=%s" % r['depth_has_images_subdir'])
print('H2: audits files count=%d' % len(files))
print('H: LINE_H_DONE_STOPPING')
