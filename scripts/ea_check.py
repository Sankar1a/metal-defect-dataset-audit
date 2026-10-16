#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Line EA: GC10-DET xml/folder label-consistency recomputation + validation.

Read-only with respect to gc10_raw and gc10_yolo.
Writes only into exp_gc10_label_20260914/.
"""
import csv
import hashlib
import os
import sys
import xml.etree.ElementTree as ET
from collections import Counter

RAW = "/home/DSJ44/sr_project/gc10_raw"
LAB = os.path.join(RAW, "lable")
YOLO = "/home/DSJ44/sr_project/gc10_yolo"
OUT = "/home/DSJ44/paper2_yolov11_steel/exp_gc10_label_20260914"
YAML = "/home/DSJ44/paper2_yolov11_steel/configs/gc10/gc10_det.yaml"

FOLDER_CLASS = {
    1: "1_chongkong", 2: "2_hanfeng", 3: "3_yueyawan", 4: "4_shuiban",
    5: "5_youban", 6: "6_siban", 7: "7_yiwu", 8: "8_yahen",
    9: "9_zhehen", 10: "10_yaozhe",
}
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def pixel_sha256(path):
    from PIL import Image
    with Image.open(path) as im:
        return hashlib.sha256(im.convert("RGB").tobytes()).hexdigest()


def parse_xml_classes(path):
    """Return (sorted unique class ids as list of int, n_objects)."""
    tree = ET.parse(path)
    root = tree.getroot()
    objs = root.findall("object")
    ids = set()
    for o in objs:
        name_el = o.find("name")
        if name_el is None or name_el.text is None:
            continue
        name = name_el.text.strip()
        prefix = name.split("_")[0]
        if prefix.isdigit():
            ids.add(int(prefix))
    return sorted(ids), len(objs)


def main():
    print("EA0 python=%s" % sys.version.split()[0], flush=True)

    # ---------------- EA1 ----------------
    # index xml files by stem (report duplicate stems if any)
    xml_by_stem = {}
    dup_xml_stems = []
    for fn in sorted(os.listdir(LAB)):
        if not fn.lower().endswith(".xml"):
            continue
        stem = os.path.splitext(fn)[0]
        if stem in xml_by_stem:
            dup_xml_stems.append(stem)
        xml_by_stem.setdefault(stem, os.path.join(LAB, fn))

    rows = []          # folder_in_xml=no  -> EA_mismatch_list.csv
    missing_rows = []  # xml 不存在      -> 仅统计/打印,不进 mismatch 列表
    jpg_n = 0
    xml_exists_n = 0
    xml_missing_n = 0
    n_obj_zero = 0
    yes_n = 0
    no_n = 0
    per_folder = {}

    for fid in sorted(FOLDER_CLASS):
        d = os.path.join(RAW, str(fid))
        if not os.path.isdir(d):
            continue
        files = sorted(f for f in os.listdir(d) if f.lower().endswith(IMG_EXT))
        per_folder[fid] = {"jpg": len(files), "no": 0, "yes": 0, "missing": 0}
        for fn in files:
            jpg_n += 1
            stem = os.path.splitext(fn)[0]
            rel = os.path.join(str(fid), fn)
            xp = xml_by_stem.get(stem)
            if xp is None:
                xml_missing_n += 1
                per_folder[fid]["missing"] += 1
                missing_rows.append({
                    "relative_path": rel, "stem": stem, "folder_id": fid,
                    "folder_class": FOLDER_CLASS[fid], "xml_class_ids": "",
                    "n_objects": "", "xml_exists": "no",
                })
                continue
            xml_exists_n += 1
            ids, nobj = parse_xml_classes(xp)
            if nobj == 0:
                n_obj_zero += 1
            in_xml = fid in ids
            if in_xml:
                yes_n += 1
                per_folder[fid]["yes"] += 1
            else:
                no_n += 1
                per_folder[fid]["no"] += 1
                rows.append({
                    "relative_path": rel, "stem": stem, "folder_id": fid,
                    "folder_class": FOLDER_CLASS[fid],
                    "xml_class_ids": "|".join(str(i) for i in ids),
                    "n_objects": nobj, "xml_exists": "yes",
                })

    print("EA1 jpg=%d, xml_exists=%d, xml_missing=%d, n_objects==0 的张数=%d"
          % (jpg_n, xml_exists_n, xml_missing_n, n_obj_zero), flush=True)
    print("EA1 folder_in_xml_yes=%d, folder_in_xml_no=%d" % (yes_n, no_n), flush=True)
    print("EA1-ADD  yes + no = %d" % (yes_n + no_n), flush=True)
    print("EA1-ADD  对照口径甲(子目录文件总数 2312 - 无标注 6) = 2306", flush=True)
    print("EA1-ADD  对照口径乙(唯一图像 2300 - 无标注 6)      = 2294", flush=True)
    print("EA1-ADD  注: 第一项为本轮重算的 yes+no 实测值,后两项照抄任务书给定口径", flush=True)
    print("EA1 per-folder: folder_id jpg yes no missing", flush=True)
    for fid in sorted(per_folder):
        v = per_folder[fid]
        print("EA1 per-folder  %2d  jpg=%4d  folder_in_xml_yes=%4d  folder_in_xml_no=%3d  xml_missing=%d"
              % (fid, v["jpg"], v["yes"], v["no"], v["missing"]), flush=True)
    if dup_xml_stems:
        print("EA1 NOTE duplicate xml stems=%d: %s" % (len(dup_xml_stems), ",".join(sorted(set(dup_xml_stems)))), flush=True)
    else:
        print("EA1 NOTE duplicate xml stems=0", flush=True)
    print("EA1 NOTE xml_missing 的 %d 张(不进 mismatch 列表,仅列出):" % len(missing_rows), flush=True)
    for r in missing_rows:
        print("EA1 XML_MISSING %s" % r["relative_path"], flush=True)

    mism_path = os.path.join(OUT, "EA_mismatch_list.csv")
    with open(mism_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["relative_path", "stem", "folder_id",
                                          "folder_class", "xml_class_ids",
                                          "n_objects", "xml_exists"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("EA1 wrote %s rows=%d" % (mism_path, len(rows)), flush=True)

    # ---------------- EA2 ----------------
    print("", flush=True)
    import json
    cache_path = os.path.join(OUT, "scripts", ".hash_cache.json")
    cache = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    def cached_sha(path):
        key = "%s|%s" % (path, os.path.getmtime(path))
        v = cache.get(key)
        if v is None:
            v = pixel_sha256(path)
            cache[key] = v
        return v

    hash_to_loc = {}
    for split in ("train", "val"):
        idir = os.path.join(YOLO, "images", split)
        files = sorted(f for f in os.listdir(idir) if f.lower().endswith(IMG_EXT))
        print("EA2 hashing gc10_yolo/%s images=%d ..." % (split, len(files)), flush=True)
        for fn in files:
            h = cached_sha(os.path.join(idir, fn))
            hash_to_loc.setdefault(h, []).append((split, fn))

    located = []
    not_found = []
    for r in rows:
        p = os.path.join(RAW, r["relative_path"])
        h = cached_sha(p)
        loc = hash_to_loc.get(h)
        if loc:
            loc_sorted = sorted(loc, key=lambda t: (0 if t[0] == "train" else 1, t[1]))
            located.append((r, loc_sorted[0], loc_sorted))
        else:
            not_found.append(r)

    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    except Exception as e:
        print("EA2 NOTE hash cache write failed: %r" % (e,), flush=True)

    n_tr = sum(1 for _, pick, _ in located if pick[0] == "train")
    n_va = sum(1 for _, pick, _ in located if pick[0] == "val")
    print("EA2 不一致图像总数=%d" % len(rows), flush=True)
    print("EA2 在 gc10_yolo 中定位成功=%d (train=%d, val=%d), 未找到=%d"
          % (len(located), n_tr, n_va, len(not_found)), flush=True)
    amb = [x for x in located if len(x[2]) > 1]
    print("EA2 哈希在 gc10_yolo 中命中多处的图像数=%d" % len(amb), flush=True)
    if not_found:
        print("EA2 未找到清单:", flush=True)
        for r in not_found:
            print("EA2 NOT_FOUND %s (xml_exists=%s, n_objects=%s)"
                  % (r["relative_path"], r["xml_exists"], r["n_objects"]), flush=True)

    loc_path = os.path.join(OUT, "EA2_located.csv")
    with open(loc_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["relative_path", "stem", "folder_id", "split", "yolo_image",
                    "yolo_label", "n_hash_hits"])
        for r, pick, allhits in located:
            split, fn = pick
            stem_y = os.path.splitext(fn)[0]
            w.writerow([r["relative_path"], r["stem"], r["folder_id"], split, fn,
                        os.path.join(split, "labels", stem_y + ".txt"), len(allhits)])
    print("EA2 wrote %s rows=%d" % (loc_path, len(located)), flush=True)

    # ---------------- EA3 ----------------
    print("", flush=True)
    print("EA3 cat %s" % YAML, flush=True)
    with open(YAML, "r", encoding="utf-8") as f:
        ytext = f.read()
    print(ytext, flush=True)

    import yaml as _yaml
    yd = _yaml.safe_load(ytext)
    nc = yd.get("nc")
    names = yd.get("names")
    print("EA3 nc=%r" % (nc,), flush=True)
    print("EA3 names=%r" % (names,), flush=True)
    print("EA3 对应表 yaml_index -> yaml_name -> folder_id", flush=True)
    ok = True
    if isinstance(names, dict):
        items = sorted(((int(k), v) for k, v in names.items()))
    elif isinstance(names, list):
        items = list(enumerate(names))
    else:
        items = []
        ok = False
    if nc != len(items) or not items:
        ok = False
    for idx, nm in items:
        folder_id = idx + 1
        fc = FOLDER_CLASS.get(folder_id)
        # yaml 的 name 不带数字前缀(folder_class 形如 "1_chongkong"),比对时去前缀
        fc_suffix = fc.split("_", 1)[1] if fc else None
        match = "OK" if fc_suffix == nm else "MISMATCH"
        if fc_suffix != nm:
            ok = False
        print("EA3   %2d -> %-16s -> folder_id=%2d (%s) %s"
              % (idx, nm, folder_id, fc, match), flush=True)
    print("EA3 一一对应=%s" % ("YES" if ok else "NO"), flush=True)

    print("EA3 folder_class list for reference: %s"
          % ", ".join("%d=%s" % (k, FOLDER_CLASS[k]) for k in sorted(FOLDER_CLASS)), flush=True)

    # ---------------- EA4 ----------------
    print("", flush=True)
    print("EA4 通过条件检查:", flush=True)
    print("EA4  EA1 完成 = YES (jpg=%d, xml_exists=%d, xml_missing=%d)"
          % (jpg_n, xml_exists_n, xml_missing_n), flush=True)
    print("EA4  EA2 定位成功数 = %d (>0 : %s)" % (len(located), "YES" if len(located) > 0 else "NO"), flush=True)
    print("EA4  EA3 索引对应明确 = %s" % ("YES" if ok else "NO"), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
