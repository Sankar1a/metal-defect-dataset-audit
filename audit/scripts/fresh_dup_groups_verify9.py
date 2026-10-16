#!/usr/bin/env python3
import os
import sys
import csv
import re
import hashlib
import shutil
import zipfile
import datetime
import subprocess
from pathlib import Path
from collections import Counter, defaultdict

from PIL import Image
import xml.etree.ElementTree as ET

HOME = Path('/home/DSJ44')
ROOT = Path('/home/DSJ44/paper2_yolov11_steel')
APSPC = ROOT / 'APSPC'
EXISTING = APSPC / 'extracted'
VERIFY = ROOT / 'verify9_20260816'
FRESH = Path('/tmp/apspc_fresh')
GC10_RAW = HOME / 'sr_project' / 'gc10_raw'

VERIFY.mkdir(parents=True, exist_ok=True)

ARCHIVE_EXT_PATTERNS = [
    '.zip', '.rar', '.tar', '.tar.gz', '.tar.bz2', '.tar.xz',
    '.tgz', '.tbz2', '.txz', '.7z'
]

FORBIDDEN_PRUNES = [
    str(ROOT / 'audits'),
    str(APSPC / 'audit'),
    str(APSPC / 'conflict_exclusion'),
]

BYTE_CACHE = {}


def emit(line):
    sys.stdout.write(line + '\n')
    sys.stdout.flush()


def sha256_file(path):
    p = str(path)
    if p in BYTE_CACHE:
        return BYTE_CACHE[p]
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    BYTE_CACHE[p] = h.hexdigest()
    return BYTE_CACHE[p]


def format_mtime(path):
    try:
        return datetime.datetime.fromtimestamp(path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
    except Exception as e:
        return f'mtime_error={e}'


def find_archive_paths(search_root, name_substr):
    name_exprs = []
    for ext in ARCHIVE_EXT_PATTERNS:
        name_exprs.append(f"-iname '*{name_substr.lower()}*{ext}'")
    name_expr = ' -o '.join(name_exprs)
    prune_args = [
        r"\( -type d -name 'runs_*' -prune \)",
        r"\( -type d -name 'verify*' -prune \)",
        r"\( -type d -name '.conda' -prune \)",
        r"\( -type d -name 'site-packages' -prune \)",
        r"\( -type d -name '.cache' -prune \)",
    ]
    for p in FORBIDDEN_PRUNES:
        prune_args.append(f"\\( -type d -path '{p}' -prune \\)")
    cmd = (
        "find " + shlex_quote(str(search_root)) + " "
        + " ".join(prune_args) + " -o "
        + r"\(" + " -type f " + r" \(" + " " + name_expr + r" \) -print0 \)"
    )
    try:
        cp = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as e:
        emit(f'find_error={e}')
        return []
    if cp.returncode != 0:
        emit(f'find_returncode={cp.returncode} stderr={cp.stderr.decode("utf-8", "replace").strip()}')
    out = cp.stdout.decode('utf-8', 'replace').split('\0')
    paths = sorted(Path(x).resolve() for x in out if x)
    return paths


def shlex_quote(s):
    return "'" + str(s).replace("'", "'\\''") + "'"


def archive_entry_stats(path):
    p = str(path)
    try:
        if path.suffix.lower() == '.zip':
            zf = zipfile.ZipFile(p)
            infos = zf.infolist()
            files = [i for i in infos if not i.is_dir()]
            counter = Counter()
            for i in files:
                ext = os.path.splitext(i.filename)[1].lower() or '(no_ext)'
                counter[ext] += 1
            zf.close()
            total = len(files)
            ext_text = ' '.join(f'{e}={counter[e]}' for e in sorted(counter))
            return total, ext_text
        elif '.tar' in path.name.lower():
            tf = tarfile.open(p, 'r:*')
            names = tf.getnames()
            counter = Counter()
            total = 0
            for n in names:
                if n.endswith('/'):
                    continue
                total += 1
                ext = os.path.splitext(n)[1].lower() or '(no_ext)'
                counter[ext] += 1
            tf.close()
            ext_text = ' '.join(f'{e}={counter[e]}' for e in sorted(counter))
            return total, ext_text
        else:
            return None, None
    except Exception as e:
        return None, f'read_error={e}'


def collect_files_by_type(root, exts):
    result = defaultdict(list)
    if not Path(root).exists():
        return result
    for p in sorted(Path(root).rglob('*')):
        if p.is_file() and p.suffix.lower() in exts:
            result[p.suffix.lower()].append(p)
    return result


def pair_and_compare(a_paths, b_paths, stem_to_paths):
    pass


def path_tree_lines(root, max_depth=2):
    root = Path(root)
    lines = []
    lines.append(str(root))

    def rec(path, prefix, depth):
        if depth >= max_depth:
            return
        children = sorted([c for c in path.iterdir()], key=lambda c: (c.is_file(), c.name.lower()))
        for i, c in enumerate(children):
            last = (i == len(children) - 1)
            connector = '└── ' if last else '├── '
            lines.append(prefix + connector + c.name + ('/' if c.is_dir() else ''))
            if c.is_dir():
                rec(c, prefix + ('    ' if last else '│   '), depth + 1)

    rec(root, '', 0)
    return '\n'.join(lines)


def parse_voc(path):
    try:
        tree = ET.parse(str(path))
        root = tree.getroot()
        objects = []
        for obj in root.iter('object'):
            name_el = obj.find('name')
            bbox_el = obj.find('bndbox')
            if name_el is None or bbox_el is None:
                return None
            vals = []
            for tag in ('name',):
                t = name_el.text
                vals.append('' if t is None else t.strip())
            for tag in ('xmin', 'ymin', 'xmax', 'ymax'):
                el = bbox_el.find(tag)
                if el is None or el.text is None:
                    return None
                vals.append(el.text.strip())
            objects.append(tuple(vals))
        objects.sort()
        return tuple(objects)
    except Exception:
        return None


def main():
    start_time = datetime.datetime.now()

    # ---------------------------------------------------------------- BA
    ba1_archives = find_archive_paths(APSPC, '')
    # find_archive_paths with empty substring would match all archives; scope below is APSPC only.
    ba1_archives = [p for p in ba1_archives if str(p).startswith(str(APSPC) + os.sep)]
    ba1_archives = sorted(ba1_archives)

    archive_lines = []
    if not ba1_archives:
        msg = 'BA1 未找到 APSPC 压缩包'
        archive_lines.append(msg)
        emit(msg)
    for p in ba1_archives:
        st = p.stat()
        sha = sha256_file(p)
        total, ext_text = archive_entry_stats(p)
        line1 = f'BA1 path={p} size={st.st_size} mtime={format_mtime(p)} sha256={sha}'
        if total is None:
            line2 = f'BA1 archive={p} internal_file_stats={ext_text}'
        else:
            line2 = f'BA1 archive={p} internal_total_files={total} ext_counts={ext_text}'
        archive_lines.extend([line1, line2])
        emit(line1)
        emit(line2)

    ba2_archives = find_archive_paths(HOME, 'apspc')
    ba2_header = 'BA2 search_root=/home/DSJ44 name_substr=apspc exclusions=runs_*,verify*,.conda,site-packages,.cache'
    archive_lines.append(ba2_header)
    emit(ba2_header)
    if not ba2_archives:
        msg = 'BA2 未找到 APSPC 压缩包'
        archive_lines.append(msg)
        emit(msg)
    else:
        for p in ba2_archives:
            try:
                size = p.stat().st_size
            except Exception:
                size = 'stat_error'
            line = f'BA2 path={p} size={size}'
            archive_lines.append(line)
            emit(line)

    with open(VERIFY / 'archive_inventory.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(archive_lines) + '\n')

    # ---------------------------------------------------------------- BB
    if FRESH.exists():
        shutil.rmtree(FRESH)
    FRESH.mkdir(parents=True, exist_ok=False)

    fresh_lines = []
    for p in ba1_archives:
        before = {q for q in FRESH.rglob('*') if q.is_file()}
        cp = subprocess.run(
            ['unzip', '-q', '-o', str(p), '-d', str(FRESH)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stderr_text = cp.stderr.decode('utf-8', 'replace').strip()
        warnings = stderr_text.splitlines() if stderr_text else []
        total, ext_text = archive_entry_stats(p)
        after = {q for q in FRESH.rglob('*') if q.is_file()}
        new_files = after - before
        line = (
            f'BB1 archive={p} zip_file_entries={total} extracted_files_on_disk={len(new_files)} '
            f'returncode={cp.returncode} warning_count={len(warnings)}'
        )
        fresh_lines.append(line)
        emit(line)
        for w in warnings:
            wl = f'BB1 archive={p} stderr={w}'
            fresh_lines.append(wl)
            emit(wl)

    fresh_jpg = sorted([p for p in FRESH.rglob('*') if p.is_file() and p.suffix.lower() == '.jpg'])
    fresh_xml = sorted([p for p in FRESH.rglob('*') if p.is_file() and p.suffix.lower() == '.xml'])
    line = f'BB2 jpg_total={len(fresh_jpg)} xml_total={len(fresh_xml)}'
    fresh_lines.append(line)
    emit(line)
    tree_text = path_tree_lines(FRESH, 2)
    fresh_lines.append(tree_text)
    emit(tree_text)

    with open(VERIFY / 'fresh_extract_summary.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(fresh_lines) + '\n')

    # ---------------------------------------------------------------- BC
    a_collect = collect_files_by_type(FRESH, {'.jpg', '.xml'})
    b_collect = collect_files_by_type(EXISTING, {'.jpg', '.xml'})
    a_jpg = a_collect.get('.jpg', [])
    a_xml = a_collect.get('.xml', [])
    b_jpg = b_collect.get('.jpg', [])
    b_xml = b_collect.get('.xml', [])

    def stem_map(paths):
        m = defaultdict(list)
        for p in paths:
            m[Path(p).stem].append(p)
        return m

    compare_lines = []
    line = (
        f'BC1 A jpg={len(a_jpg)}, B jpg={len(b_jpg)} | '
        f'A xml={len(a_xml)}, B xml={len(b_xml)}'
    )
    compare_lines.append(line)
    emit(line)

    def pair_stats(a_paths, b_paths):
        am = stem_map(a_paths)
        bm = stem_map(b_paths)
        stems = sorted(set(am) | set(bm))
        paired = 0
        only_a = 0
        only_b = 0
        same = 0
        diff = 0
        diff_rows = []
        only_rows = []
        for s in stems:
            al = sorted(am.get(s, []))
            bl = sorted(bm.get(s, []))
            n = min(len(al), len(bl))
            paired += n
            only_a += max(0, len(al) - n)
            only_b += max(0, len(bl) - n)
            for k in range(n):
                ah = sha256_file(al[k])
                bh = sha256_file(bl[k])
                if ah == bh:
                    same += 1
                else:
                    diff += 1
                    diff_rows.append((s, ah, bh, str(al[k]), str(bl[k])))
            for p in al[n:]:
                only_rows.append((s, 'A', str(p)))
            for p in bl[n:]:
                only_rows.append((s, 'B', str(p)))
        return paired, only_a, only_b, same, diff, diff_rows, only_rows

    paired_j, only_a_j, only_b_j, same_j, diff_j, diff_rows_j, only_rows_j = pair_stats(a_jpg, b_jpg)
    paired_x, only_a_x, only_b_x, same_x, diff_x, diff_rows_x, only_rows_x = pair_stats(a_xml, b_xml)

    paired = paired_j + paired_x
    only_a = only_a_j + only_a_x
    only_b = only_b_j + only_b_x
    same = same_j + same_x
    different = diff_j + diff_x
    diff_rows = diff_rows_j + diff_rows_x
    only_rows = only_rows_j + only_rows_x

    line = f'BC1 配对成功={paired}, 仅在A={only_a}, 仅在B={only_b}'
    compare_lines.append(line)
    emit(line)
    line = f'BC1 配对文件中 byte 完全相同={same}, 不同={different}'
    compare_lines.append(line)
    emit(line)

    if diff_rows:
        compare_lines.append('BC1 different_files:')
        for row in diff_rows[:20]:
            line = f'BC1 different stem={row[0]} A_sha256={row[1]} B_sha256={row[2]} A_path={row[3]} B_path={row[4]}'
            compare_lines.append(line)
            emit(line)
        if len(diff_rows) > 20:
            line = f'BC1 different_files_truncated total={len(diff_rows)} shown=20'
            compare_lines.append(line)
            emit(line)
    if only_rows:
        compare_lines.append('BC1 only_one_side_files:')
        for row in only_rows[:20]:
            line = f'BC1 only_one_side stem={row[0]} side={row[1]} path={row[2]}'
            compare_lines.append(line)
            emit(line)
        if len(only_rows) > 20:
            line = f'BC1 only_one_side_files_truncated total={len(only_rows)} shown=20'
            compare_lines.append(line)
            emit(line)

    with open(VERIFY / 'fresh_vs_existing.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(compare_lines) + '\n')

    # ---------------------------------------------------------------- BD
    pixel_groups = defaultdict(list)
    byte_by_path = {}
    pixel_errors = []
    for p in fresh_jpg:
        bhash = sha256_file(p)
        byte_by_path[str(p)] = bhash
        try:
            with Image.open(str(p)) as im:
                rgb = im.convert('RGB')
                data = rgb.tobytes()
            phash = hashlib.sha256(data).hexdigest()
            pixel_groups[phash].append((p, bhash))
        except Exception as e:
            pixel_errors.append((str(p), str(e)))

    dup_items = sorted(
        ((ph, lst) for ph, lst in pixel_groups.items() if len(lst) >= 2),
        key=lambda kv: (kv[0], kv[1][0][0].name.lower()),
    )

    xml_paths = sorted([p for p in FRESH.rglob('*') if p.is_file() and p.suffix.lower() == '.xml'])
    xml_by_stem = defaultdict(list)
    for p in xml_paths:
        xml_by_stem[p.stem].append(p)

    csv_path = VERIFY / 'fresh_dup_groups.csv'
    conflicting = 0
    identical = 0
    missing_label = 0
    files_involved = 0
    files_removable = 0

    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['pixel_sha256', 'n_files', 'file_paths', 'byte_hashes_distinct', 'label_status'])
        for ph, lst in dup_items:
            n_files = len(lst)
            paths = [str(p) for p, _ in lst]
            distinct = len({b for _, b in lst})
            statuses = []
            parsed_labels = []
            for p, _ in lst:
                stem = p.stem
                xp = xml_by_stem.get(stem, [])
                if not xp:
                    statuses.append('missing')
                    parsed_labels.append(None)
                    continue
                parsed = parse_voc(xp[0])
                if parsed is None:
                    statuses.append('missing')
                    parsed_labels.append(None)
                else:
                    statuses.append('found')
                    parsed_labels.append(parsed)
            if 'missing' in statuses:
                status = 'missing'
                missing_label += 1
            elif all(x == parsed_labels[0] for x in parsed_labels):
                status = 'identical'
                identical += 1
            else:
                status = 'conflicting'
                conflicting += 1
            files_involved += n_files
            files_removable += (n_files - 1)
            writer.writerow([ph, n_files, ';'.join(paths), distinct, status])

    images_total = len(fresh_jpg)
    dup_groups = len(dup_items)
    line = (
        f'BD5: images_total={images_total}, dup_groups={dup_groups}, '
        f'conflicting={conflicting}, identical={identical}, '
        f'missing_label={missing_label}, files_involved={files_involved}, '
        f'files_removable={files_removable}'
    )
    emit(line)
    if pixel_errors:
        for e in pixel_errors[:20]:
            emit(f'BD1 pixel_hash_error path={e[0]} error={e[1]}')
        emit(f'BD1 pixel_hash_error_total={len(pixel_errors)}')

    # ---------------------------------------------------------------- BE
    gc_lines = []
    gc10_archives = find_archive_paths(HOME, 'gc10')
    if not gc10_archives:
        msg = 'BE1 未找到 GC10 压缩包'
        gc_lines.append(msg)
        emit(msg)
    else:
        for p in gc10_archives:
            st = p.stat()
            sha = sha256_file(p)
            total, ext_text = archive_entry_stats(p)
            line = f'BE1 path={p} size={st.st_size} mtime={format_mtime(p)} sha256={sha}'
            gc_lines.append(line)
            emit(line)
            line = f'BE1 archive={p} internal_total_files={total} ext_counts={ext_text}'
            gc_lines.append(line)
            emit(line)

    subdir_names = [str(i) for i in range(1, 11)]
    total_jpg = 0
    unique_stems = set()
    for d in subdir_names:
        dpath = GC10_RAW / d
        jpgs = sorted([p for p in dpath.iterdir() if p.is_file() and p.suffix.lower() == '.jpg']) if dpath.exists() else []
        cnt = len(jpgs)
        total_jpg += cnt
        for p in jpgs:
            unique_stems.add(p.stem)
        line = f'BE2 subdir={d} jpg={cnt}'
        gc_lines.append(line)
        emit(line)
    line = f'BE2 subdir_jpg_total={total_jpg}'
    gc_lines.append(line)
    emit(line)
    unique_count = len(unique_stems)
    line = f'BE2 unique_jpg_stems={unique_count}'
    gc_lines.append(line)
    emit(line)
    lable_dir = GC10_RAW / 'lable'
    lable_xmls = sorted([p for p in lable_dir.rglob('*') if p.is_file() and p.suffix.lower() == '.xml']) if lable_dir.exists() else []
    lable_xml_count = len(lable_xmls)
    line = f'BE2 lable_xml_count={lable_xml_count}'
    gc_lines.append(line)
    emit(line)
    line = f'BE2 2312-subdir_jpg_total={2312 - total_jpg}'
    gc_lines.append(line)
    emit(line)
    line = f'BE2 2294-subdir_jpg_total={2294 - total_jpg}'
    gc_lines.append(line)
    emit(line)
    line = f'BE2 2312-unique_jpg_stems={2312 - unique_count}'
    gc_lines.append(line)
    emit(line)
    line = f'BE2 2294-unique_jpg_stems={2294 - unique_count}'
    gc_lines.append(line)
    emit(line)
    line = f'BE2 2294-lable_xml_count={2294 - lable_xml_count}'
    gc_lines.append(line)
    emit(line)

    with open(VERIFY / 'gc10_count_check.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(gc_lines) + '\n')

    # ---------------------------------------------------------------- BF
    end_time = datetime.datetime.now()
    script_path = Path(__file__).resolve()
    script_sha = sha256_file(script_path)
    env_lines = [
        f'python_version={sys.version.split()[0]}',
        f'script_path={script_path}',
        f'script_sha256={script_sha}',
        f'timestamp_start={start_time.strftime("%Y-%m-%d %H:%M:%S.%f")}',
        f'timestamp_end={end_time.strftime("%Y-%m-%d %H:%M:%S.%f")}',
        f'conda_env_name={os.environ.get("CONDA_DEFAULT_ENV", "")}',
        f'python_executable={sys.executable}',
        'involved_directories:',
        f'{APSPC}',
        f'{EXISTING}',
        f'{FRESH}',
        f'{VERIFY}',
        f'{GC10_RAW}',
        f'{HOME}',
    ]
    with open(VERIFY / 'verify9_env.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(env_lines) + '\n')
    for line in env_lines:
        emit(line)


if __name__ == '__main__':
    main()
