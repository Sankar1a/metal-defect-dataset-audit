#!/usr/bin/env python3
"""Round-3 Line L: audit three RAW dataset copies (APSPC_RAW, GC10_RAW, NEU_RAW).
L0 annotation inventory, L1 dual hash, L2 dup groups (byte+pixel), L3 label conflicts,
L4 conflict PNGs, L5 counts, L6 disk credential. Read-only on raw dirs; writes verify3_20260805/.
Env: yolov11 (PIL/matplotlib). No audit computation on unrelated data.
"""
import os, csv, hashlib, subprocess, datetime
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

VERIFY3 = '/home/DSJ44/paper2_yolov11_steel/verify3_20260805'
CPDIR = os.path.join(VERIFY3, 'raw_conflict_pairs')

DATASETS = {
    'APSPC_RAW': {
        'root': '/home/DSJ44/paper2_yolov11_steel/APSPC',
        'expected_images': 1885,
        'label_source': 'voc',
        'annot_dir': '/home/DSJ44/paper2_yolov11_steel/APSPC/extracted/annotations/APSPC-Annotations/Annotations',
        'class_map': {'none': 0},
    },
    'GC10_RAW': {
        'root': '/home/DSJ44/sr_project/gc10_raw',
        'expected_images': 2312,
        'label_source': 'voc',
        'annot_dir': '/home/DSJ44/sr_project/gc10_raw/lable',
        'class_map': {'none': 0},
    },
    'NEU_RAW': {
        'root': '/home/DSJ44/sr_project/raw_neu_images',
        'expected_images': 1800,
        'label_source': 'none',
        'annot_dir': None,
        'class_map': {},
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

def parse_voc(xml_path):
    rows = []
    try:
        tree = ET.parse(xml_path)
    except Exception:
        return rows
    root = tree.getroot()
    for obj in root.iter('object'):
        name = (obj.findtext('name') or '').strip()
        bb = obj.find('bndbox')
        if bb is None:
            continue
        def g(tag):
            t = bb.findtext(tag)
            return float(t.strip()) if t is not None and t.strip() else 0.0
        rows.append((name, g('xmin'), g('ymin'), g('xmax'), g('ymax')))
    return sorted(rows)

def labels_equal(a, b):
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x[0] != y[0]:
            return False
        for u, v in zip(x[1:], y[1:]):
            if abs(u - v) > 1e-6:
                return False
    return True

# ---------- L0 inventory ----------
inv_rows = []
for ds, info in DATASETS.items():
    root = info['root']
    from collections import Counter
    cnt = Counter(); ex = {}
    for dp, dnames, fnames in os.walk(root):
        for f in fnames:
            ext = os.path.splitext(f)[1].lower() or '(noext)'
            cnt[ext] += 1
            ex.setdefault(ext, os.path.join(dp, f))
    for ext in sorted(cnt, key=lambda e: -cnt[e]):
        inv_rows.append({'dataset': ds, 'extension': ext, 'count': cnt[ext], 'example_path': ex[ext]})
    print(f"L0 {ds}: total_files={sum(cnt.values())}")
    for ext in sorted(cnt, key=lambda e: -cnt[e]):
        print(f"L0   {ext}: {cnt[ext]} example={ex[ext]}")
with open(os.path.join(VERIFY3, 'raw_annotation_inventory.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['dataset','extension','count','example_path'])
    w.writeheader(); w.writerows(inv_rows)

# ---------- L1 dual hashes ----------
imgmeta = {}  # ds -> list of (rel, abs, byte_sha, pixel_sha, w, h, size)
for ds, info in DATASETS.items():
    root = info['root']
    files = []
    for dp, dnames, fnames in os.walk(root):
        for f in fnames:
            if f.lower().endswith(IMG_EXTS):
                files.append(os.path.join(dp, f))
    files.sort()
    print(f"L1 {ds}: images_found={len(files)} expected={info['expected_images']}")
    for ap in files:
        rel = os.path.relpath(ap, root)
        byte_sha = sha256_file(ap)
        with Image.open(ap) as im:
            w, h = im.size
            pixel_sha = sha256_bytes(im.convert('RGB').tobytes())
        imgmeta.setdefault(ds, []).append((rel, ap, byte_sha, pixel_sha, w, h, os.path.getsize(ap)))

with open(os.path.join(VERIFY3, 'raw_all_hashes.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['dataset','relative_path','byte_sha256','pixel_sha256','width','height','filesize_bytes'])
    w.writeheader()
    for ds, lst in imgmeta.items():
        for rel, ap, bs, ps, iw, ih, sz in lst:
            w.writerow({'dataset':ds,'relative_path':rel,'byte_sha256':bs,'pixel_sha256':ps,'width':iw,'height':ih,'filesize_bytes':sz})

# ---------- L2 dup groups (byte + pixel) ----------
def build_groups(ds, key_index):
    d = {}
    for rel, ap, bs, ps, w, h, sz in imgmeta[ds]:
        key = (bs, ps)[key_index]
        d.setdefault(key, []).append((rel, ap, bs, ps))
    return [v for v in d.values() if len(v) >= 2]

def annot_map_for(ds):
    info = DATASETS[ds]
    m = {}
    if info['annot_dir'] and os.path.isdir(info['annot_dir']):
        for dp, dnames, fnames in os.walk(info['annot_dir']):
            for f in fnames:
                if f.lower().endswith('.xml'):
                    m[os.path.splitext(f)[0]] = os.path.join(dp, f)
    return m

# pre-build annotation maps
ann_map = {ds: annot_map_for(ds) for ds in DATASETS}

def group_label_status(ds, members):
    info = DATASETS[ds]
    if info['label_source'] == 'none':
        return 'none'
    if info['label_source'] == 'voc':
        parsed = []; missing = False
        for rel, ap, bs, ps in members:
            base = os.path.splitext(os.path.basename(rel))[0]
            xml_path = ann_map[ds].get(base)
            if xml_path is None:
                missing = True; parsed.append(None)
            else:
                parsed.append(parse_voc(xml_path))
        if missing:
            return 'missing'
        if all(labels_equal(parsed[0], p) for p in parsed[1:]):
            return 'identical'
        return 'conflicting'
    return 'none'

dup_rows = []
for ds in DATASETS:
    for hash_type, ki in (('byte', 0), ('pixel', 1)):
        for g in build_groups(ds, ki):
            key = g[0][2] if ki == 0 else g[0][3]
            dup_rows.append({
                'dataset': ds, 'hash_type': hash_type, 'hash': key, 'n_files': len(g),
                'file_paths': ';'.join(r[0] for r in g),
                'byte_hashes_distinct': len({r[2] for r in g}),
                'label_status': group_label_status(ds, g),
            })

with open(os.path.join(VERIFY3, 'raw_dup_groups.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['dataset','hash_type','hash','n_files','file_paths','byte_hashes_distinct','label_status'])
    w.writeheader(); w.writerows(dup_rows)

# ---------- L4 conflict PNGs ----------
os.makedirs(CPDIR, exist_ok=True)
conf = [r for r in dup_rows if r['label_status'] == 'conflicting']
png_count = 0
for r in conf:
    if png_count >= 20:
        break
    ds = r['dataset']; root = DATASETS[ds]['root']; info = DATASETS[ds]
    rels = r['file_paths'].split(';')
    n = len(rels)
    fig, axes = plt.subplots(1, n, figsize=(3.0*n, 3.6), dpi=110)
    if n == 1:
        axes = [axes]
    for ax, rel in zip(axes, rels):
        ap = os.path.join(root, rel)
        try:
            img = mpimg.imread(ap)
            ax.imshow(img)
        except Exception:
            ax.text(0.5, 0.5, 'DECODE_FAILED', ha='center', va='center'); img = None
        base = os.path.splitext(os.path.basename(rel))[0]
        xml_path = ann_map[ds].get(base) if info['label_source'] == 'voc' else None
        if xml_path:
            for (name, xmin, ymin, xmax, ymax) in parse_voc(xml_path):
                ax.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                                           fill=False, edgecolor='lime', lw=1.5))
                ax.annotate(name, (xmin, ymin), color='yellow', fontsize=7,
                            xytext=(2, 2), textcoords='offset points')
        ax.set_xlabel(rel, fontsize=7)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(os.path.join(CPDIR, f'{ds}_{r["hash"][:12]}.png'))
    plt.close(fig)
    png_count += 1

# ---------- L5 counts ----------
for ds in DATASETS:
    for ht in ('byte', 'pixel'):
        dd = [r for r in dup_rows if r['dataset'] == ds and r['hash_type'] == ht]
        n_conf = sum(1 for r in dd if r['label_status'] == 'conflicting')
        n_ident = sum(1 for r in dd if r['label_status'] == 'identical')
        n_miss = sum(1 for r in dd if r['label_status'] == 'missing')
        n_reenc = sum(1 for r in dd if r['byte_hashes_distinct'] > 1)
        removable = sum(r['n_files'] - 1 for r in dd)
        print(f"L5 {ds} {ht}: images_total={len(imgmeta[ds])}, dup_groups={len(dd)}, "
              f"conflicting={n_conf}, identical={n_ident}, missing_label={n_miss}, "
              f"groups_with_reencoding={n_reenc}, files_removable={removable}")

# ---------- L6 disk credential ----------
dup_sha = sha256_file(os.path.join(VERIFY3, 'raw_dup_groups.csv'))
dup_lines = sum(1 for _ in open(os.path.join(VERIFY3, 'raw_dup_groups.csv'))) - 1
print(f"L6 raw_dup_groups.csv sha256={dup_sha} rows={dup_lines}")
