#!/usr/bin/env python3
import os
import sys
import csv
import io
import math
import random
import shutil
import hashlib
import difflib
import datetime
import subprocess
from pathlib import Path
from collections import Counter, OrderedDict

import yaml
from PIL import Image
import xml.etree.ElementTree as ET

PROJECT = Path('/home/DSJ44/paper2_yolov11_steel')
APSPC = PROJECT / 'APSPC'
EXTRACTED = APSPC / 'extracted'
ANN_DIR = EXTRACTED / 'annotations' / 'APSPC-Annotations' / 'Annotations'
CLEAN = PROJECT / 'datasets' / 'apspc_clean1711_seed0'
VERIFY3 = PROJECT / 'verify3_20260805' / 'raw_dup_groups.csv'
CONFIG_CA1 = PROJECT / 'configs' / 'apspc' / 'apspc_clean1711_seed0_dev.yaml'
EXP = PROJECT / 'exp_apspc_dose_20260817'
DATA = EXP / 'data'
CONFIGS_OUT = EXP / 'configs'
CA_TXT = EXP / 'CA_preflight.txt'
CB_TXT = EXP / 'CB_build.txt'

RNG = random.Random(0)


def emit(line, ca_lines=None, cb_lines=None):
    print(line, flush=True)
    if ca_lines is not None:
        ca_lines.append(line)
    if cb_lines is not None:
        cb_lines.append(line)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def xml_objects_and_size(xml_path):
    root = ET.parse(str(xml_path)).getroot()
    size_el = root.find('size')
    xml_w = None
    xml_h = None
    if size_el is not None:
        w_el = size_el.find('width')
        h_el = size_el.find('height')
        if w_el is not None and w_el.text is not None:
            xml_w = int(float(w_el.text.strip()))
        if h_el is not None and h_el.text is not None:
            xml_h = int(float(h_el.text.strip()))
    objs = []
    for obj in root.iter('object'):
        name_el = obj.find('name')
        bbox_el = obj.find('bndbox')
        if name_el is None or bbox_el is None:
            raise ValueError(f'{xml_path}: object missing name or bndbox')
        name = (name_el.text or '').strip()
        vals = []
        for tag in ('xmin', 'ymin', 'xmax', 'ymax'):
            el = bbox_el.find(tag)
            if el is None or el.text is None:
                raise ValueError(f'{xml_path}: bndbox missing {tag}')
            vals.append(float(el.text.strip()))
        objs.append((name, vals[0], vals[1], vals[2], vals[3]))
    objs.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]))
    return objs, xml_w, xml_h


def parse_voc_for_hash(xml_path):
    objs, _, _ = xml_objects_and_size(xml_path)
    return objs


def pixel_sha256(image_path):
    with Image.open(str(image_path)) as im:
        rgb = im.convert('RGB')
        return hashlib.sha256(rgb.tobytes()).hexdigest(), im.size


def convert_objects_to_yolo_lines(objs, name2idx, img_w, img_h):
    lines = []
    for name, xmin, ymin, xmax, ymax in objs:
        if name not in name2idx:
            raise ValueError(f'unknown class {name}')
        cls = name2idx[name]
        cx = (float(xmin) + float(xmax)) / 2.0 / float(img_w)
        cy = (float(ymin) + float(ymax)) / 2.0 / float(img_h)
        bw = (float(xmax) - float(xmin)) / float(img_w)
        bh = (float(ymax) - float(ymin)) / float(img_h)
        lines.append((cls, cx, cy, bw, bh))
    lines.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]))
    return lines


def parse_yolo_txt(txt_path):
    rows = []
    if not Path(txt_path).exists():
        return None
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f'{txt_path}: bad yolo line {line!r}')
            rows.append(tuple(float(x) for x in parts))
    rows.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]))
    return rows


def labels_equal(a, b, tol=1e-4):
    if len(a) != len(b):
        return False
    return all(len(x) == len(y) and all(abs(i - j) <= tol for i, j in zip(x, y)) for x, y in zip(a, b))


def find_image_in_extracted(stem):
    for part in ('part1', 'part2'):
        p = EXTRACTED / part / f'APSPC{part[-1]}' / f'{stem}.jpg'
        if p.exists():
            return p
    return None


def find_xml_in_extracted(stem):
    p = ANN_DIR / f'{stem}.xml'
    if p.exists():
        return p
    return None


def make_unique_target(target_dir, stem, ext):
    target = target_dir / f'{stem}.{ext}'
    if not target.exists():
        return target, target.name
    idx = 1
    while True:
        name = f'dup_{stem}.{ext}' if idx == 1 else f'dup{idx}_{stem}.{ext}'
        target = target_dir / name
        if not target.exists():
            return target, target.name
        idx += 1


def copy_and_convert(img_src, label_xml_src, img_dir, label_dir, name2idx, collisions):
    stem = Path(img_src).stem
    target_img, target_name = make_unique_target(img_dir, stem, 'jpg')
    target_label = label_dir / (Path(target_name).stem + '.txt')
    shutil.copy2(str(img_src), str(target_img))
    objs, xml_w, xml_h = xml_objects_and_size(label_xml_src)
    with Image.open(str(img_src)) as im:
        actual_w, actual_h = im.size
    lines = convert_objects_to_yolo_lines(objs, name2idx, actual_w, actual_h)
    with open(target_label, 'w', encoding='utf-8') as f:
        for cls, cx, cy, bw, bh in lines:
            f.write(f'{int(cls)} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}\n')
    if target_name != f'{stem}.jpg':
        collisions.append((str(img_src), target_name))
    return target_img, target_label, (xml_w != actual_w or xml_h != actual_h)


def main():
    t0 = datetime.datetime.now()
    ca_lines = []
    cb_lines = []

    # Environment at start
    sys_ver = subprocess.run([sys.executable, '-c', 'import sys; print(sys.version.split()[0])'], capture_output=True, text=True).stdout.strip()
    emit(f'python_version={sys_ver}', ca_lines=ca_lines)
    torch_out = subprocess.run([sys.executable, '-c', 'import torch; print(torch.__version__)'], capture_output=True, text=True).stdout.strip()
    emit(f'torch_version={torch_out}', ca_lines=ca_lines)
    ult_out = subprocess.run([sys.executable, '-c', 'import ultralytics; print(ultralytics.__version__)'], capture_output=True, text=True).stdout.strip()
    emit(f'ultralytics_version={ult_out}', ca_lines=ca_lines)
    smi = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu', '--format=csv'], capture_output=True, text=True)
    emit('nvidia_smi:', ca_lines=ca_lines)
    for line in smi.stdout.splitlines():
        emit(line, ca_lines=ca_lines)
    if smi.returncode != 0:
        emit(smi.stderr.strip(), ca_lines=ca_lines)

    # CA1
    cfg = yaml.safe_load(Path(CONFIG_CA1).read_text(encoding='utf-8'))
    names = OrderedDict((int(k), str(v)) for k, v in cfg['names'].items())
    nc = int(cfg['nc'])
    emit(f'CA1 nc={nc}', ca_lines=ca_lines)
    for idx in sorted(names):
        emit(f'CA1 names[{idx}]={names[idx]}', ca_lines=ca_lines)

    # CA2
    with open(VERIFY3, newline='') as f:
        raw_rows = list(csv.DictReader(f))
    pixel_rows = [r for r in raw_rows if r.get('dataset') == 'APSPC_RAW' and r.get('hash_type') == 'pixel']
    if len(pixel_rows) != 82:
        emit(f'CA2 FAIL pixel_group_count={len(pixel_rows)} expected=82', ca_lines=ca_lines)
        Path(CA_TXT).write_text('\n'.join(ca_lines) + '\n', encoding='utf-8')
        sys.exit(1)
    group_paths = []
    for r in pixel_rows:
        paths = []
        for rel in r['file_paths'].split(';'):
            p = (APSPC / rel).resolve()
            if not p.exists():
                emit(f'CA2 FAIL missing file {rel}', ca_lines=ca_lines)
                Path(CA_TXT).write_text('\n'.join(ca_lines) + '\n', encoding='utf-8')
                sys.exit(1)
            paths.append(str(p))
        group_paths.append(paths)
    all_files = [p for g in group_paths for p in g]
    if len(all_files) != 174:
        emit(f'CA2 FAIL file_total={len(all_files)} expected=174', ca_lines=ca_lines)
        Path(CA_TXT).write_text('\n'.join(ca_lines) + '\n', encoding='utf-8')
        sys.exit(1)
    name_counts = Counter()
    missing_xml = []
    for p in all_files:
        stem = Path(p).stem
        xml = find_xml_in_extracted(stem)
        if xml is None:
            missing_xml.append(stem)
            continue
        objs, _, _ = xml_objects_and_size(xml)
        for o in objs:
            name_counts[o[0]] += 1
    emit('CA2 xml_name_counts:', ca_lines=ca_lines)
    for name in sorted(name_counts):
        emit(f'CA2 name={name} count={name_counts[name]}', ca_lines=ca_lines)
    unknown = [n for n in name_counts if n not in names.values()]
    if missing_xml:
        emit(f'CA2 FAIL missing_xml={len(missing_xml)} {missing_xml[:20]}', ca_lines=ca_lines)
    if unknown:
        emit(f'CA2 FAIL unknown_names={unknown}', ca_lines=ca_lines)
    if missing_xml or unknown:
        emit('CA6 FAIL', ca_lines=ca_lines)
        Path(CA_TXT).write_text('\n'.join(ca_lines) + '\n', encoding='utf-8')
        sys.exit(1)

    # CA3
    name2idx = {v: k for k, v in names.items()}
    train_imgs = sorted((CLEAN / 'images' / 'train').glob('*.jpg'))
    sample_stems = [p.stem for p in RNG.sample(train_imgs, 10)]
    ca3_same = 0
    ca3_diff = 0
    ca3_size_mismatch = 0
    ca3_detail_lines = []
    for stem in sample_stems:
        clean_img = CLEAN / 'images' / 'train' / f'{stem}.jpg'
        clean_txt = CLEAN / 'labels' / 'train' / f'{stem}.txt'
        xml = find_xml_in_extracted(stem)
        orig_img = find_image_in_extracted(stem)
        if xml is None or orig_img is None or not clean_txt.exists():
            emit(f'CA3 FAIL missing source for {stem} xml={xml} orig_img={orig_img} label={clean_txt.exists()}', ca_lines=ca_lines)
            ca3_diff += 1
            continue
        objs, xml_w, xml_h = xml_objects_and_size(xml)
        with Image.open(str(orig_img)) as im:
            actual_w, actual_h = im.size
        if (xml_w != actual_w) or (xml_h != actual_h):
            ca3_size_mismatch += 1
            ca3_detail_lines.append(f'CA3 size_mismatch stem={stem} xml_size={xml_w}x{xml_h} actual_size={actual_w}x{actual_h}')
        conv = convert_objects_to_yolo_lines(objs, name2idx, actual_w, actual_h)
        existing = parse_yolo_txt(clean_txt)
        if existing is None:
            emit(f'CA3 FAIL missing existing label {clean_txt}', ca_lines=ca_lines)
            ca3_diff += 1
        elif labels_equal(conv, existing):
            ca3_same += 1
        else:
            ca3_diff += 1
            ca3_detail_lines.append(f'CA3 mismatch stem={stem}')
            ca3_detail_lines.append(f'CA3 xml={xml}')
            ca3_detail_lines.append(f'CA3 image={orig_img}')
            ca3_detail_lines.append(f'CA3 converted={conv}')
            ca3_detail_lines.append(f'CA3 existing={existing}')
    emit(f'CA3 往返校验: 10 张中完全一致={ca3_same}, 不一致={ca3_diff}', ca_lines=ca_lines)
    emit(f'CA3 xml_size 与实际图像尺寸不一致的张数={ca3_size_mismatch}', ca_lines=ca_lines)
    for line in ca3_detail_lines:
        emit(line, ca_lines=ca_lines)
    if ca3_diff != 0:
        emit('CA6 FAIL', ca_lines=ca_lines)
        Path(CA_TXT).write_text('\n'.join(ca_lines) + '\n', encoding='utf-8')
        sys.exit(1)

    # CA4
    parsed_groups = []
    for paths in group_paths:
        sorted_paths = sorted(paths)
        copy_a = sorted_paths[0]
        copy_b = sorted_paths[1]
        parsed_groups.append({'paths': sorted_paths, 'copy_A': copy_a, 'copy_B': copy_b})
    size_dist = Counter(len(g['paths']) for g in parsed_groups)
    emit(f'CA4 组数={len(parsed_groups)}, 组大小分布={dict(sorted(size_dist.items()))}', ca_lines=ca_lines)
    emit(f'CA4 copy_A 总数={len(parsed_groups)}, copy_B 总数={len(parsed_groups)}', ca_lines=ca_lines)
    pixel_same = 0
    label_diff = 0
    ca4_fail_lines = []
    for g in parsed_groups:
        ha, _ = pixel_sha256(g['copy_A'])
        hb, _ = pixel_sha256(g['copy_B'])
        if ha == hb:
            pixel_same += 1
        else:
            ca4_fail_lines.append(f'CA4 pixel_hash_diff group_A={g["copy_A"]} group_B={g["copy_B"]} A={ha} B={hb}')
        objs_a = parse_voc_for_hash(find_xml_in_extracted(Path(g['copy_A']).stem))
        objs_b = parse_voc_for_hash(find_xml_in_extracted(Path(g['copy_B']).stem))
        if objs_a != objs_b:
            label_diff += 1
        else:
            ca4_fail_lines.append(f'CA4 label_same group_A={g["copy_A"]} group_B={g["copy_B"]}')
    emit(f'CA4 copy_A 与 copy_B 像素哈希相同的组数={pixel_same}   （应等于组数）', ca_lines=ca_lines)
    emit(f'CA4 copy_A 与 copy_B 标注不同的组数={label_diff}       （应等于组数）', ca_lines=ca_lines)
    for line in ca4_fail_lines:
        emit(line, ca_lines=ca_lines)

    # CA5
    sorted_groups = sorted(parsed_groups, key=lambda g: Path(g['paths'][0]).name)
    # Sort by pixel_sha256: attach hash from verify3 rows in same order
    # parsed_groups order follows pixel_rows; use that for hash mapping.
    hash_for_first = {}
    for r, g in zip(pixel_rows, parsed_groups):
        first = sorted(g['paths'])[0]
        hash_for_first[first] = r['hash']
    for g in sorted_groups:
        first = sorted(g['paths'])[0]
        g['pixel_hash'] = hash_for_first[first]
    sorted_groups.sort(key=lambda g: g['pixel_hash'])
    S20 = sorted_groups[:20]
    S28 = sorted_groups[:28]
    S41 = sorted_groups[:41]
    S82 = sorted_groups[:82]
    emit(f'CA5 S20={len(S20)} S28={len(S28)} S41={len(S41)} S82={len(S82)}', ca_lines=ca_lines)
    emit(f'CA5 nested={set(g["pixel_hash"] for g in S20) <= set(g["pixel_hash"] for g in S28) <= set(g["pixel_hash"] for g in S41) <= set(g["pixel_hash"] for g in S82)}', ca_lines=ca_lines)

    ca4_ok = (pixel_same == 82 and label_diff == 82 and len(parsed_groups) == 82 and set(size_dist) <= {2, 3})
    ca5_ok = (len(S20) == 20 and len(S28) == 28 and len(S41) == 41 and len(S82) == 82)
    gpu_ok = True
    try:
        gpu_q = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True)
        used, util = [x.strip() for x in gpu_q.stdout.strip().split(',')]
        gpu_ok = (int(used) == 0)
        emit(f'CA6 gpu_memory_used_mib={used} gpu_utilization_pct={util} gpu_idle={gpu_ok}', ca_lines=ca_lines)
    except Exception as e:
        gpu_ok = False
        emit(f'CA6 gpu_check_error={e}', ca_lines=ca_lines)

    if ca4_ok and ca5_ok and gpu_ok:
        emit('CA6 PASS', ca_lines=ca_lines)
    else:
        emit(f'CA6 FAIL ca4_ok={ca4_ok} ca5_ok={ca5_ok} gpu_ok={gpu_ok}', ca_lines=ca_lines)
        Path(CA_TXT).write_text('\n'.join(ca_lines) + '\n', encoding='utf-8')
        sys.exit(1)

    Path(CA_TXT).write_text('\n'.join(ca_lines) + '\n', encoding='utf-8')

    # ------------------------------------------------------------------ CB
    if DATA.exists():
        emit(f'CB FAIL data dir already exists {DATA}', cb_lines=cb_lines)
        Path(CB_TXT).write_text('\n'.join(cb_lines) + '\n', encoding='utf-8')
        sys.exit(1)
    DATA.mkdir(parents=True)
    CONFIGS_OUT.mkdir(parents=True, exist_ok=True)

    # CB1 train_leak
    train_img_dir = DATA / 'train_leak' / 'images'
    train_lbl_dir = DATA / 'train_leak' / 'labels'
    train_img_dir.mkdir(parents=True)
    train_lbl_dir.mkdir(parents=True)
    clean_train_imgs = sorted((CLEAN / 'images' / 'train').glob('*.jpg'))
    clean_train_lbls = sorted((CLEAN / 'labels' / 'train').glob('*.txt'))
    for p in clean_train_imgs:
        shutil.copy2(str(p), str(train_img_dir / p.name))
    for p in clean_train_lbls:
        shutil.copy2(str(p), str(train_lbl_dir / p.name))
    collisions = []
    for g in sorted_groups:
        copy_a = g['copy_A']
        xml_a = find_xml_in_extracted(Path(copy_a).stem)
        if xml_a is None:
            emit(f'CB FAIL copy_A xml missing {copy_a}', cb_lines=cb_lines)
            sys.exit(1)
        _, _, mismatch = copy_and_convert(copy_a, xml_a, train_img_dir, train_lbl_dir, name2idx, collisions)
        if mismatch:
            emit(f'CB copy_A size_mismatch image={copy_a} xml={xml_a}', cb_lines=cb_lines)
    train_img_count = len(list(train_img_dir.glob('*.jpg')))
    train_lbl_count = len(list(train_lbl_dir.glob('*.txt')))
    emit(f'CB1 train 图像数={train_img_count} (应为 1280), labels 数={train_lbl_count}', cb_lines=cb_lines)
    emit(f'CB1 copy_A_collisions={len(collisions)}', cb_lines=cb_lines)
    for row in collisions:
        emit(f'CB1 collision_map={row[0]} -> {row[1]}', cb_lines=cb_lines)
    if train_img_count != 1280 or train_lbl_count != 1280:
        emit(f'CB1 FAIL counts', cb_lines=cb_lines)
        Path(CB_TXT).write_text('\n'.join(cb_lines) + '\n', encoding='utf-8')
        sys.exit(1)

    # CB2 validation sets
    clean_val_imgs = sorted((CLEAN / 'images' / 'val').glob('*.jpg'))
    clean_val_lbls = sorted((CLEAN / 'labels' / 'val').glob('*.txt'))
    val_specs = [('val_d0', 0, None)]
    for d in (20, 28, 41, 82):
        val_specs.append((f'val_d{d}_same', d, 'same'))
        val_specs.append((f'val_d{d}_conf', d, 'conf'))

    val_counts = {}
    val_collisions = []
    for val_name, dose, variant in val_specs:
        img_dir = DATA / val_name / 'images'
        lbl_dir = DATA / val_name / 'labels'
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        for p in clean_val_imgs:
            shutil.copy2(str(p), str(img_dir / p.name))
        for p in clean_val_lbls:
            shutil.copy2(str(p), str(lbl_dir / p.name))
        if dose > 0:
            groups = sorted_groups[:dose]
            for g in groups:
                copy_b = g['copy_B']
                if variant == 'same':
                    label_src_path = g['copy_A']
                else:
                    label_src_path = g['copy_B']
                xml_src = find_xml_in_extracted(Path(label_src_path).stem)
                if xml_src is None:
                    emit(f'CB FAIL val label xml missing {label_src_path} for {val_name}', cb_lines=cb_lines)
                    sys.exit(1)
                target_img, target_name = make_unique_target(img_dir, Path(copy_b).stem, 'jpg')
                shutil.copy2(str(copy_b), str(target_img))
                target_label = lbl_dir / (Path(target_name).stem + '.txt')
                objs, _, _ = xml_objects_and_size(xml_src)
                with Image.open(str(copy_b)) as im:
                    actual_w, actual_h = im.size
                lines = convert_objects_to_yolo_lines(objs, name2idx, actual_w, actual_h)
                with open(target_label, 'w', encoding='utf-8') as f:
                    for cls, cx, cy, bw, bh in lines:
                        f.write(f'{int(cls)} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}\n')
                if target_name != f'{Path(copy_b).stem}.jpg':
                    val_collisions.append((val_name, str(copy_b), target_name))
        n_img = len(list(img_dir.glob('*.jpg')))
        n_lbl = len(list(lbl_dir.glob('*.txt')))
        val_counts[val_name] = (n_img, n_lbl)
        emit(f'CB2 {val_name} images={n_img} labels={n_lbl}', cb_lines=cb_lines)
    emit(f'CB2 val_collisions={len(val_collisions)}', cb_lines=cb_lines)
    for row in val_collisions:
        emit(f'CB2 val_collision val={row[0]} source={row[1]} -> {row[2]}', cb_lines=cb_lines)

    # CB3 yaml
    yaml_dir = CONFIGS_OUT
    common = {
        'path': str(DATA),
        'train': 'train_leak/images',
        'test': '',
        'nc': nc,
        'names': dict(names),
    }
    val_yaml_names = [name for name, _, _ in val_specs]
    for val_name in val_yaml_names:
        common['val'] = f'{val_name}/images'
        yaml_text = yaml.safe_dump(common, allow_unicode=True, sort_keys=False, default_flow_style=None)
        (yaml_dir / f'apspc_{val_name.removeprefix("val_")}.yaml').write_text(yaml_text, encoding='utf-8')
    d0_yaml = yaml_dir / 'apspc_d0.yaml'
    d82_yaml = yaml_dir / 'apspc_d82_conf.yaml'
    emit('CB3 val_d0 yaml:', cb_lines=cb_lines)
    emit(d0_yaml.read_text(encoding='utf-8').rstrip('\n'), cb_lines=cb_lines)
    diff = difflib.unified_diff(
        d0_yaml.read_text(encoding='utf-8').splitlines(),
        d82_yaml.read_text(encoding='utf-8').splitlines(),
        fromfile='apspc_d0.yaml', tofile='apspc_d82_conf.yaml', lineterm='')
    emit('CB3 diff d0.yaml vs d82_conf.yaml:', cb_lines=cb_lines)
    for line in diff:
        emit(line, cb_lines=cb_lines)

    # train_leak manifest
    manifest_path = DATA / 'train_leak_manifest.csv'
    with open(manifest_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['image', 'sha256'])
        for p in sorted(train_img_dir.glob('*.jpg')):
            w.writerow([f'train_leak/images/{p.name}', sha256_file(p)])
    emit(f'CB3 train_leak_manifest={manifest_path}', cb_lines=cb_lines)
    emit(f'CB3 train_leak_manifest_sha256={sha256_file(manifest_path)}', cb_lines=cb_lines)
    emit(f'CB end_timestamp={datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")}', cb_lines=cb_lines)

    Path(CB_TXT).write_text('\n'.join(cb_lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
