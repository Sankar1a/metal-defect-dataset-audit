#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Round 18 · 第2步: 生成 groups.csv / boxes.csv / png_map.csv(只读原始解压副本)。

分组来源 = 第1步找到的分组文件,不自行重新分组。
本脚本同时从原始解压副本独立复算 byte/pixel SHA256,用于交叉核对分组来源。
只写 round18_groups_20261006/ 下的产物。
"""
import collections
import csv
import hashlib
import os
import xml.etree.ElementTree as ET

from PIL import Image

ROOT = "/home/DSJ44/paper2_yolov11_steel"
D = os.path.join(ROOT, "round18_groups_20261006")
IMG_DIRS = {"APSPC1": "APSPC/extracted/part1/APSPC1",
            "APSPC2": "APSPC/extracted/part2/APSPC2"}
ANNOT_DIR = "APSPC/extracted/annotations/APSPC-Annotations/Annotations"
GROUP_SRC = "verify9_20260816/fresh_dup_groups.csv"
CROSS_SRC = ["APSPC/dedup_audit/conflicting_duplicate_groups.csv",
             "APSPC/conflict_exclusion/excluded_conflict_groups.csv"]

os.chdir(ROOT)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def num(s):
    """XML 原值 -> int/float,不换算不取整(非数值则原样返回)。"""
    try:
        f = float(s)
    except (TypeError, ValueError):
        return s
    return int(f) if f.is_integer() else f


def parse_voc(path):
    """按 XML 文档顺序返回 [(name, xmin, ymin, xmax, ymax)] 原值字符串。"""
    root = ET.parse(path).getroot()
    out = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        bb = obj.find("bndbox")
        if bb is None:
            continue
        out.append(tuple([name] + [(bb.findtext(t) or "").strip()
                                   for t in ("xmin", "ymin", "xmax", "ymax")]))
    return out


# ---------- 1. 原始解压副本独立复算 ----------
meta = {}
for part, d in IMG_DIRS.items():
    for f in sorted(os.listdir(d)):
        if not f.lower().endswith(".jpg"):
            continue
        p = os.path.join(d, f)
        with Image.open(p) as im:
            ps = hashlib.sha256(im.convert("RGB").tobytes()).hexdigest()
        meta[f] = {"part": part, "byte_sha256": sha256_file(p), "pixel_sha256": ps}
assert len(meta) == 1885, len(meta)

# ---------- 2. 分组来源(第1步找到的文件) ----------
groups = []
for r in csv.DictReader(open(GROUP_SRC, encoding="utf-8")):
    mem = sorted(os.path.basename(p) for p in r["file_paths"].split(";"))
    calc = {meta[f]["pixel_sha256"] for f in mem}
    assert len(calc) == 1, (r["pixel_sha256"], mem)
    assert calc.pop() == r["pixel_sha256"], "复算 pixel_sha256 与分组文件键不符"
    groups.append((r["pixel_sha256"], mem))

ngroups = len(groups)
nfiles = sum(len(m) for _, m in groups)
dist = collections.Counter(len(m) for _, m in groups)
allf = [f for _, m in groups for f in m]
assert len(set(allf)) == nfiles, "存在跨组重复文件"

# ---------- 3. 与其它视图交叉核对(成员集合两两比较) ----------
def members_of_dedup(path, keycol, memcol, sep, suffix):
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        out[r[keycol]] = frozenset(x + suffix for x in r[memcol].split(sep))
    return out

views = {"fresh(pixel)": {k: frozenset(v) for k, v in groups}}
views["dedup_audit"] = members_of_dedup(CROSS_SRC[0], "sha", "stems", "|", ".jpg")
views["conflict_exclusion"] = members_of_dedup(CROSS_SRC[1], "sha256", "stems", "|", ".jpg")
for nm, v in views.items():
    sz = {len(s) for s in v.values()}
    print("view %-20s groups=%d files=%d sizes=%s" % (nm, len(v), sum(len(s) for s in v.values()), sorted(sz)))
base = frozenset(views["fresh(pixel)"].values())
for nm, v in views.items():
    print("  %-20s member-set identical to fresh: %s" % (nm, frozenset(v.values()) == base))

# ---------- 4. 编号 G01-G82(按 pixel_sha256 字典序) ----------
groups.sort(key=lambda t: t[0])
gid = {h: "G%02d" % (i + 1) for i, (h, _) in enumerate(groups)}

# ---------- 5. groups.csv / boxes.csv ----------
groups_rows, boxes_rows = [], []
for h, mem in groups:
    for f in mem:
        stem = os.path.splitext(f)[0]
        xml_name = stem + ".xml"
        objs = parse_voc(os.path.join(ANNOT_DIR, xml_name))
        groups_rows.append({
            "group_id": gid[h], "pixel_sha256": h, "n_copies": len(mem),
            "filename": f, "part": meta[f]["part"], "xml_file": xml_name,
            "n_objects": len(objs),
            "classes_sorted": "|".join(sorted(o[0] for o in objs)),
        })
        for i, (name, xmin, ymin, xmax, ymax) in enumerate(objs):
            boxes_rows.append({
                "group_id": gid[h], "filename": f, "obj_idx": i, "class_name": name,
                "xmin": num(xmin), "ymin": num(ymin), "xmax": num(xmax), "ymax": num(ymax),
                "width": num(xmax) - num(xmin), "height": num(ymax) - num(ymin),
            })

# ---------- 6. png_map.csv ----------
pngs = [l.strip() for l in open(os.path.join(D, "step1_png_list.txt"), encoding="utf-8") if l.strip()]
byte_of = {}
for h, mem in groups:
    bs = {meta[f]["byte_sha256"] for f in mem}
    assert len(bs) == 1, mem
    byte_of[h] = bs.pop()
by_prefix = collections.defaultdict(list)
for h, b in byte_of.items():
    by_prefix[b[:12]].append(h)

png_rows, unmatched = [], 0
for p in pngs:
    ds, _, h12 = p[:-4].rpartition("_")
    hit = by_prefix.get(h12, [])
    if ds == "APSPC_RAW" and len(hit) == 1:
        h = hit[0]
        png_rows.append({"png_filename": p, "group_id": gid[h],
                         "member_filenames": "|".join(dict(groups)[h])})
    else:
        unmatched += 1
        png_rows.append({"png_filename": p, "group_id": "UNMATCHED", "member_filenames": ""})

# ---------- 7. 写出 ----------
def write_csv(name, cols, rows):
    with open(os.path.join(D, name), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

write_csv("groups.csv",
          ["group_id", "pixel_sha256", "n_copies", "filename", "part",
           "xml_file", "n_objects", "classes_sorted"], groups_rows)
write_csv("boxes.csv",
          ["group_id", "filename", "obj_idx", "class_name",
           "xmin", "ymin", "xmax", "ymax", "width", "height"], boxes_rows)
write_csv("png_map.csv",
          ["png_filename", "group_id", "member_filenames"], png_rows)

print("\nRESULT groups=%d files=%d dist=%s" % (ngroups, nfiles, dict(sorted(dist.items()))))
print("RESULT groups.csv=%d boxes.csv=%d png_map.csv=%d UNMATCHED=%d"
      % (len(groups_rows), len(boxes_rows), len(png_rows), unmatched))
print("RESULT gid %s..%s" % (min(gid.values()), max(gid.values())))
