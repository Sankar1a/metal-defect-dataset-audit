#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Round 21 第5步:评估(9 次训练全部结束后执行)。

5.0 评估流程自检(必须先做,不通过则停止)
     APSPC: 对原有 A 臂 YOLO11n 种子 0 的 best.pt 在 d0 与 d28-same 上重新评估,
            与 exp_apspc_dose_unified_20260915/dose_results_unified.csv 对应行比较
            mAP50 / mAP50-95,须逐位相同
     GC10 : 对原有 XML 条件 YOLO11n 种子 0 的 best.pt 在 XML 验证集上重新评估,
            与 exp_gc10_label_20260914/gc10_label_results.csv 对应行比较,须逐位相同
     结果写入 eval_selfcheck.txt;任一不相同 -> 停止,报告,不做后续评估

5.1 APSPC 剂量评估:  A 组 3 个 best.pt × 9 个验证集 = 27 次 -> dose_results_v8.csv
5.2 GC10 2×2 交叉:  B 组 6 个 best.pt × (XML, DIR) = 12 次 -> gc10_label_results_v8.csv
                    四格全部记录(不沿用 ed_eval.py 第 106 行 if target == cond 的自尺子逻辑)
5.3 GC10 分类别:    B 组 6 个模型在 XML 验证集上的 ap50 / ap5095 -> gc10_per_class_v8.csv
                    一致性检查 -> gc10_per_class_check.txt

model.val(...) 参数抄自 exp_apspc_dose_unified_20260915/scripts/eval_dose.py 第 88-90 行
(原文见 eval_kwargs_source.txt),只改 data 与输出位置。

用法: r21_eval.py
"""
import csv
import hashlib
import os
import sys
import time

import yaml

ROOT = "/home/DSJ44/paper2_yolov11_steel"
EXP = os.path.join(ROOT, "round21_v8data_20261009")
AUG = os.path.join(ROOT, "exp_apspc_dose_20260817")
OLD_A = os.path.join(ROOT, "exp_apspc_dose_unified_20260915")
OLD_G = os.path.join(ROOT, "exp_gc10_label_20260914")
EVALDIR = os.path.join(EXP, "runs_eval")

VAL_SETS = ["val_d0",
            "val_d20_same", "val_d28_same", "val_d41_same", "val_d82_same",
            "val_d20_conf", "val_d28_conf", "val_d41_conf", "val_d82_conf"]
VAL_YAML = {v: os.path.join(AUG, "configs", "apspc_%s.yaml" % v[len("val_"):])
            for v in VAL_SETS}

GC10_YAML = {"xml": os.path.join(OLD_G, "configs", "gc10_xml.yaml"),
             "dir": os.path.join(OLD_G, "configs", "gc10_dir.yaml")}

# 本轮 A 组(condition=A_arm)与 B 组 run
A_RUNS = [("apspc_yolov8n_leaktrain_unified_s0", 0),
          ("apspc_yolov8n_leaktrain_unified_s1", 1),
          ("apspc_yolov8n_leaktrain_unified_s42", 42)]
B_RUNS = [("XML", "gc10_yolov8n_xml_s0", 0),
          ("XML", "gc10_yolov8n_xml_s1", 1),
          ("XML", "gc10_yolov8n_xml_s42", 42),
          ("DIR", "gc10_yolov8n_dir_s0", 0),
          ("DIR", "gc10_yolov8n_dir_s1", 1),
          ("DIR", "gc10_yolov8n_dir_s42", 42)]

SELFCHECK = os.path.join(EXP, "eval_selfcheck.txt")


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def august_meta():
    """val_set -> (dose_groups, dose_images, val_total_images, label_variant)
    取自 8 月实验 exp_apspc_dose_20260817/dose_results.csv(与 eval_dose.py 同源)。"""
    meta = {}
    with open(os.path.join(AUG, "dose_results.csv"), "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            meta[r["val_set"]] = (r["dose_groups"], r["dose_images"],
                                  r["val_total_images"], r["label_variant"])
    return meta


def read_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def val_once(weight, data, tag):
    """model.val(...) 参数与 eval_dose.py 第 88-90 行相同,只改 data 与输出位置。"""
    from ultralytics import YOLO
    r = YOLO(weight).val(data=data, imgsz=640, split="val", batch=16,
                         device=0, verbose=False,
                         project=EVALDIR, name=tag)
    return r


def class_names_of(data_yaml):
    with open(data_yaml, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    nm = d.get("names")
    if isinstance(nm, dict):
        return [nm[k] for k in sorted(nm)]
    return list(nm)


# ======================================================================
# 5.0 评估流程自检
# ======================================================================
def step50_selfcheck():
    L = []
    A = L.append
    ok = True
    A("eval_selfcheck.txt — Round 21 第5.0步 评估流程自检")
    A("生成时间: %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    A("评估参数: model.val(data=..., imgsz=640, split='val', batch=16, device=0, verbose=False)")
    A("          原文见 eval_kwargs_source.txt(eval_dose.py 第 88-90 行)")
    A("=" * 78)

    # ---------- APSPC ----------
    wA = os.path.join(OLD_A, "runs", "apspc_yolo11n_leaktrain_unified_s0", "weights", "best.pt")
    A("")
    A("[A] APSPC 原有 A 臂 YOLO11n 种子 0")
    A("    weights = %s" % wA)
    A("    exists  = %s" % os.path.exists(wA))
    A("    sha256  = %s" % (sha256_file(wA) if os.path.exists(wA) else "<缺失>"))
    old_dose = read_csv(os.path.join(OLD_A, "dose_results_unified.csv"))

    for vset in ("val_d0", "val_d28_same"):
        ref = next((r for r in old_dose
                    if r["val_set"] == vset and r["seed"] == "0"), None)
        r = val_once(wA, VAL_YAML[vset], "selfcheck_apspc_A_s0__on_%s" % vset)
        m50, m5095 = float(r.box.map50), float(r.box.map)
        new50, new5095 = "%.6f" % m50, "%.6f" % m5095
        ref50 = ref["map50"] if ref else "<无对应行>"
        ref5095 = ref["map5095"] if ref else "<无对应行>"
        same50 = (new50 == ref50)
        same5095 = (new5095 == ref5095)
        A("")
        A("    ---- val_set = %s ----" % vset)
        A("    dose_results_unified.csv 原值 : map50=%s  map5095=%s" % (ref50, ref5095))
        A("    本次重算值                    : map50=%s  map5095=%s" % (new50, new5095))
        A("    是否逐位相同                  : map50=%s  map5095=%s"
          % ("相同" if same50 else "不相同", "相同" if same5095 else "不相同"))
        if not (same50 and same5095):
            ok = False

    # ---------- GC10 ----------
    wG = os.path.join(OLD_G, "runs", "gc10_yolo11n_xml_s0", "weights", "best.pt")
    A("")
    A("[B] GC10 原有 XML 条件 YOLO11n 种子 0")
    A("    weights = %s" % wG)
    A("    exists  = %s" % os.path.exists(wG))
    A("    sha256  = %s" % (sha256_file(wG) if os.path.exists(wG) else "<缺失>"))
    old_g = read_csv(os.path.join(OLD_G, "gc10_label_results.csv"))
    refG = next((r for r in old_g if r["run_name"] == "gc10_yolo11n_xml_s0"), None)

    r = val_once(wG, GC10_YAML["xml"], "selfcheck_gc10_xml_s0__on_xml")
    g50, g5095 = float(r.box.map50), float(r.box.map)
    refG50 = float(refG["eval_on_xml_map50"]) if refG else None
    refG5095 = float(refG["eval_on_xml_map5095"]) if refG else None
    sameG50 = (refG50 is not None) and (g50 == refG50)
    sameG5095 = (refG5095 is not None) and (g5095 == refG5095)
    A("")
    A("    ---- 评估集 = XML 验证集 (configs/gc10_xml.yaml) ----")
    A("    gc10_label_results.csv 原值(eval_on_xml_map50)   : %s" % repr(refG50))
    A("    gc10_label_results.csv 原值(eval_on_xml_map5095) : %s" % repr(refG5095))
    A("    本次重算值                                       : %s" % repr(g50))
    A("    本次重算值                                       : %s" % repr(g5095))
    A("    浮点完全相等: map50=%s  map5095=%s"
      % ("相同" if sameG50 else "不相同", "相同" if sameG5095 else "不相同"))
    A("    6 位小数比较: map50=%s  map5095=%s"
      % ("相同" if ("%.6f" % g50) == ("%.6f" % (refG50 if refG50 is not None else -1)) else "不相同",
         "相同" if ("%.6f" % g5095) == ("%.6f" % (refG5095 if refG5095 is not None else -1)) else "不相同"))
    if not (sameG50 and sameG5095):
        ok = False

    A("")
    A("=" * 78)
    A("自检结论: %s" % ("全部逐位相同 PASS" if ok else "存在不相同 FAIL"))
    A("后续评估: %s" % ("继续执行 5.1 / 5.2 / 5.3" if ok else "停止,不执行后续评估"))
    A("=" * 78)

    with open(SELFCHECK, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L), flush=True)
    return ok


# ======================================================================
# 5.1 APSPC 剂量评估
# ======================================================================
def step51_dose(meta):
    cols = ["seed", "run_name", "val_set", "dose_groups", "dose_images",
            "val_total_images", "label_variant", "map50", "map5095",
            "best_pt_sha256"]
    out = os.path.join(EXP, "dose_results_v8.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        f.flush()
        os.fsync(f.fileno())

    n = 0
    for run_name, seed in A_RUNS:
        wpath = os.path.join(EXP, run_name, "weights", "best.pt")
        sha = sha256_file(wpath) if os.path.exists(wpath) else ""
        print("\n[R21-5.1] %s seed=%d best.pt exists=%s sha256=%s"
              % (run_name, seed, os.path.exists(wpath), sha), flush=True)
        for vset in VAL_SETS:
            r = val_once(wpath, VAL_YAML[vset], "%s__on_%s" % (run_name, vset))
            m50, m5095 = float(r.box.map50), float(r.box.map)
            dg, di, vti, lv = meta[vset]
            row = {"seed": seed, "run_name": run_name, "val_set": vset,
                   "dose_groups": dg, "dose_images": di, "val_total_images": vti,
                   "label_variant": lv, "map50": "%.6f" % m50,
                   "map5095": "%.6f" % m5095, "best_pt_sha256": sha}
            with open(out, "a", newline="", encoding="utf-8") as f:
                ww = csv.DictWriter(f, fieldnames=cols)
                ww.writerow(row)
                f.flush()
                os.fsync(f.fileno())
            n += 1
            print("[R21-5.1]   %-14s map50=%.6f map5095=%.6f" % (vset, m50, m5095), flush=True)
    print("\n[R21-5.1] wrote %s rows=%d" % (out, n), flush=True)
    return n


# ======================================================================
# 5.2 GC10 2×2 交叉评估  +  5.3 分类别
# ======================================================================
def step52_53():
    cols = ["train_condition", "seed", "run_name", "best_pt_sha256",
            "eval_on", "map50", "map5095"]
    out = os.path.join(EXP, "gc10_label_results_v8.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        f.flush()
        os.fsync(f.fileno())

    pc_cols = ["train_condition", "seed", "class_index", "class_name", "ap50", "ap5095"]
    pc_out = os.path.join(EXP, "gc10_per_class_v8.csv")
    with open(pc_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=pc_cols)
        w.writeheader()
        f.flush()
        os.fsync(f.fileno())

    names = class_names_of(GC10_YAML["xml"])
    check = []
    check.append("gc10_per_class_check.txt — Round 21 第5.3步 一致性检查")
    check.append("生成时间: %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    check.append("检查内容: 每个模型在 XML 验证集上十类 ap5095 的平均值 是否等于 该模型总体 map5095")
    check.append("=" * 78)

    n = 0
    for cond, run_name, seed in B_RUNS:
        wpath = os.path.join(EXP, run_name, "weights", "best.pt")
        sha = sha256_file(wpath) if os.path.exists(wpath) else ""
        print("\n[R21-5.2] %s condition=%s seed=%d best.pt exists=%s sha256=%s"
              % (run_name, cond, seed, os.path.exists(wpath), sha), flush=True)
        for target in ("xml", "dir"):
            r = val_once(wpath, GC10_YAML[target], "%s__on_%s" % (run_name, target))
            b = r.box
            m50, m5095 = float(b.map50), float(b.map)
            row = {"train_condition": cond, "seed": seed, "run_name": run_name,
                   "best_pt_sha256": sha, "eval_on": target,
                   "map50": "%.6f" % m50, "map5095": "%.6f" % m5095}
            with open(out, "a", newline="", encoding="utf-8") as f:
                ww = csv.DictWriter(f, fieldnames=cols)
                ww.writerow(row)
                f.flush()
                os.fsync(f.fileno())
            n += 1
            print("[R21-5.2]   eval_on=%-3s map50=%.6f map5095=%.6f" % (target, m50, m5095),
                  flush=True)

            # ---- 5.3 分类别:只在 XML 验证集上 ----
            if target == "xml":
                ap50 = list(b.ap50)
                ap = list(b.ap)
                idx = [int(c) for c in list(b.ap_class_index)]
                got50 = {c: float(ap50[i]) for i, c in enumerate(idx)}
                got5095 = {c: float(ap[i]) for i, c in enumerate(idx)}
                vals = []
                for ci in range(len(names)):
                    v50 = got50.get(ci)
                    v95 = got5095.get(ci)
                    vals.append(v95)
                    with open(pc_out, "a", newline="", encoding="utf-8") as f:
                        ww = csv.DictWriter(f, fieldnames=pc_cols)
                        ww.writerow({"train_condition": cond, "seed": seed,
                                     "class_index": ci, "class_name": names[ci],
                                     "ap50": ("%.6f" % v50) if v50 is not None else "",
                                     "ap5095": ("%.6f" % v95) if v95 is not None else ""})
                        f.flush()
                        os.fsync(f.fileno())
                present = [v for v in vals if v is not None]
                mean_present = sum(present) / len(present) if present else float("nan")
                check.append("")
                check.append("model=%s  train_condition=%s  seed=%d" % (run_name, cond, seed))
                check.append("    r.box.ap_class_index = %s" % idx)
                check.append("    类别数(共 %d 类)出现于评估结果的类别数 = %d"
                             % (len(names), len(present)))
                check.append("    十类中缺失的类别索引 = %s"
                             % [i for i, v in enumerate(vals) if v is None])
                check.append("    mean(ap5095 over 出现类别) = %.10f" % mean_present)
                check.append("    r.box.map (总体 map5095)   = %.10f" % m5095)
                check.append("    差值(mean - 总体)          = %.10e" % (mean_present - m5095))
                check.append("    是否相等                   = %s"
                             % ("相等" if mean_present == m5095 else "不相等"))
                if present and len(present) == len(names):
                    check.append("    (十类全部出现,均值即十类平均)")
                else:
                    check.append("    注意: 有类别未出现在 ap_class_index 中,"
                                 "'十类平均'按出现类别求均值")
    print("\n[R21-5.2] wrote %s rows=%d" % (out, n), flush=True)
    with open(os.path.join(EXP, "gc10_per_class_check.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(check) + "\n")
    print("\n".join(check), flush=True)
    return n


def main():
    os.makedirs(EVALDIR, exist_ok=True)
    print("[R21-EVAL] start=%s" % time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)

    # 记录用到的验证集 yaml 路径
    with open(os.path.join(EXP, "dose_val_yamls.txt"), "w", encoding="utf-8") as f:
        f.write("dose_val_yamls.txt — 第5.1步 APSPC 9 个验证集 data yaml(与原实验完全相同)\n")
        f.write("来源: exp_apspc_dose_unified_20260915/scripts/common.py 的 VAL_YAML\n")
        f.write("=" * 74 + "\n")
        for v in VAL_SETS:
            f.write("%-14s %s  exists=%s\n" % (v, VAL_YAML[v], os.path.exists(VAL_YAML[v])))
    with open(os.path.join(EXP, "gc10_val_yamls.txt"), "w", encoding="utf-8") as f:
        f.write("gc10_val_yamls.txt — 第5.2步 GC10 两个验证集 data yaml\n")
        f.write("来源: exp_gc10_label_20260914/scripts/ed_eval.py 的 YAML 字典\n")
        f.write("=" * 74 + "\n")
        for k in ("xml", "dir"):
            f.write("%-4s %s  exists=%s\n" % (k, GC10_YAML[k], os.path.exists(GC10_YAML[k])))

    ok = step50_selfcheck()
    if not ok:
        print("\n[R21-EVAL] 5.0 自检未通过 -> 停止,不做后续评估", flush=True)
        return 1

    meta = august_meta()
    step51_dose(meta)
    step52_53()
    print("\n[R21-EVAL] end=%s" % time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
