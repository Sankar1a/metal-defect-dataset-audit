#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Line EB: build the two GC10 dataset copies (XML control / DIR treatment).

Never writes outside exp_gc10_label_20260914/.
Source gc10_yolo is opened read-only.
"""
import csv
import os
import shutil
import sys

SRC = "/home/DSJ44/sr_project/gc10_yolo"
OUT = "/home/DSJ44/paper2_yolov11_steel/exp_gc10_label_20260914"
DATA = os.path.join(OUT, "data")
XML = os.path.join(DATA, "gc10_xml")
DIRC = os.path.join(DATA, "gc10_dir")
LOC = os.path.join(OUT, "EA2_located.csv")
CFG = os.path.join(OUT, "configs")


def copy_split(src_root, dst_root):
    for kind in ("images", "labels"):
        for split in ("train", "val"):
            s = os.path.join(src_root, kind, split)
            d = os.path.join(dst_root, kind, split)
            os.makedirs(d, exist_ok=True)
            n = 0
            skipped = []
            for fn in sorted(os.listdir(s)):
                sp = os.path.join(s, fn)
                if not os.path.isfile(sp):
                    continue
                if fn.endswith(".cache"):
                    skipped.append(fn)
                    continue
                shutil.copy2(sp, os.path.join(d, fn))
                n += 1
            yield kind, split, n, skipped


def counts(root):
    out = {}
    for kind in ("images", "labels"):
        for split in ("train", "val"):
            p = os.path.join(root, kind, split)
            out["%s_%s" % (kind, split)] = len(os.listdir(p)) if os.path.isdir(p) else -1
    return out


def total_label_lines(root):
    tot = {}
    for split in ("train", "val"):
        d = os.path.join(root, "labels", split)
        t = 0
        for fn in os.listdir(d):
            if fn.endswith(".txt"):
                with open(os.path.join(d, fn), "r", encoding="utf-8", errors="replace") as f:
                    t += sum(1 for _ in f)
        tot[split] = t
    return tot


def main():
    print("EB0 python=%s" % sys.version.split()[0], flush=True)
    if os.path.exists(XML) or os.path.exists(DIRC):
        print("EB0 ABORT: target already exists -> %s / %s" % (XML, DIRC), flush=True)
        return 1

    # ---------------- EB1 ----------------
    print("", flush=True)
    print("EB1 复制 gc10_yolo -> data/gc10_xml", flush=True)
    print("EB1 NOTE 源目录含 labels/train.cache 与 labels/val.cache;"
          " .cache 为 ultralytics 自动生成、内含按目录计算的哈希,不复制到副本,"
          "以保证副本自洽且训练时不会回写源目录。", flush=True)
    for kind, split, n, skipped in copy_split(SRC, XML):
        print("EB1 copy %-6s %-5s files=%d (skipped .cache=%s)"
              % (kind, split, n, ",".join(skipped) if skipped else "none"), flush=True)
    c = counts(XML)
    print("EB1 train=%d val=%d labels_train=%d labels_val=%d"
          % (c["images_train"], c["images_val"], c["labels_train"], c["labels_val"]), flush=True)
    xml_lines_before = total_label_lines(XML)
    print("EB1 label 总行数 train=%d val=%d" % (xml_lines_before["train"], xml_lines_before["val"]), flush=True)

    # ---------------- EB2 ----------------
    print("", flush=True)
    print("EB2 复制 data/gc10_xml -> data/gc10_dir", flush=True)
    for kind, split, n, skipped in copy_split(XML, DIRC):
        print("EB2 copy %-6s %-5s files=%d" % (kind, split, n), flush=True)

    with open(LOC, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print("EB2 EA2_located 输入行数=%d" % len(rows), flush=True)

    n_tr = n_va = 0
    samples = []
    for r in rows:
        folder_id = int(r["folder_id"])
        yaml_index = folder_id - 1        # EA3 已核对: yaml_index + 1 == folder_id
        split = r["split"]
        rel = r["yolo_label"]             # 形如 train/labels/xxx.txt
        lp = os.path.join(DIRC, "labels", split, os.path.basename(rel))
        before = ""
        if os.path.exists(lp):
            with open(lp, "r", encoding="utf-8", errors="replace") as f:
                before = f.read()
        else:
            print("EB2 WARN label 不存在: %s" % lp, flush=True)
        after = before + ("%d 0.5 0.5 1.0 1.0\n" % yaml_index)
        with open(lp, "w", encoding="utf-8") as f:
            f.write(after)
        if len(samples) < 3:
            samples.append((lp, folder_id, yaml_index, before, after))
        if split == "train":
            n_tr += 1
        else:
            n_va += 1

    print("EB2 修改的 label 文件数=%d (train=%d, val=%d)" % (n_tr + n_va, n_tr, n_va), flush=True)
    dir_lines_after = total_label_lines(DIRC)
    print("EB2 修改前后行数变化: 总行数 从 %d 增至 %d"
          % (xml_lines_before["train"] + xml_lines_before["val"],
             dir_lines_after["train"] + dir_lines_after["val"]), flush=True)
    print("EB2 修改前后行数变化(train): 从 %d 增至 %d" % (xml_lines_before["train"], dir_lines_after["train"]), flush=True)
    print("EB2 修改前后行数变化(val)  : 从 %d 增至 %d" % (xml_lines_before["val"], dir_lines_after["val"]), flush=True)
    print("EB2 抽样 3 个被修改文件:", flush=True)
    for lp, fid, yi, before, after in samples:
        print("EB2 SAMPLE %s (folder_id=%d -> yaml_index=%d)" % (lp, fid, yi), flush=True)
        print("EB2 SAMPLE --- 修改前 ---", flush=True)
        print(before if before else "<空文件>", flush=True)
        print("EB2 SAMPLE --- 修改后 ---", flush=True)
        print(after, flush=True)
        print("EB2 SAMPLE --- 行数 %d -> %d ---" % (before.count("\n"), after.count("\n")), flush=True)

    c2 = counts(DIRC)
    print("EB2 data/gc10_dir train=%d val=%d labels_train=%d labels_val=%d"
          % (c2["images_train"], c2["images_val"], c2["labels_train"], c2["labels_val"]), flush=True)

    # ---------------- EB3 ----------------
    print("", flush=True)
    os.makedirs(CFG, exist_ok=True)
    y_xml = os.path.join(CFG, "gc10_xml.yaml")
    y_dir = os.path.join(CFG, "gc10_dir.yaml")
    names = ["chongkong", "hanfeng", "yueyawan", "shuiban", "youban",
             "siban", "yiwu", "yahen", "zhehen", "yaozhe"]
    body_names = "\n".join("  %d: %s" % (i, n) for i, n in enumerate(names))

    with open(y_xml, "w", encoding="utf-8") as f:
        f.write("# GC10-DET condition XML (control) — exp_gc10_label_20260914\n"
                "path: %s\n"
                "train: images/train\n"
                "val: images/val\n"
                "test: ''\n"
                "nc: 10\n"
                "names:\n%s\n" % (XML, body_names))
    with open(y_dir, "w", encoding="utf-8") as f:
        f.write("# GC10-DET condition DIR (treatment) — exp_gc10_label_20260914\n"
                "path: %s\n"
                "train: images/train\n"
                "val: images/val\n"
                "test: ''\n"
                "nc: 10\n"
                "names:\n%s\n" % (DIRC, body_names))
    print("EB3 wrote %s" % y_xml, flush=True)
    print("EB3 wrote %s" % y_dir, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
