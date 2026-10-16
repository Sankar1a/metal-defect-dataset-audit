#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Round 22 第6步 · 评估(12 次训练全部结束后执行)。

评估参数沿用 exp_apspc_dose_unified_20260915/scripts/eval_dose.py 第 88-90 行
(原文抄入 eval_kwargs_source.txt),只改 data 与输出位置(本轮目录 runs_eval/)。

6.0 自检 : 先用同样参数重新评估**原有** A 臂 YOLO11n 种子 0 的 best.pt 在 d0 上的结果,
          须与 dose_results_unified.csv 逐位相同 -> eval_selfcheck.txt;不同 -> 停止
6.1 NEU 换卷 : 3 个 NEU 模型 × (val359, val360) -> neu_leak_rtdetr.csv
6.2 APSPC 剂量: 3 个 A 臂模型 × 9 个验证集 -> dose_results_rtdetr.csv
6.3 GC10 2×2 : 6 个模型 × (XML, DIR),四格全部记录 -> gc10_label_results_rtdetr.csv
6.4 GC10 分类别: 6 个模型在 XML 验证集上的 ap50 / ap -> gc10_per_class_rtdetr.csv
               十类 ap5095 均值与总体 map5095 的差值 -> gc10_per_class_check.txt

用法: python -u r22_eval.py > logs/_eval.log 2>&1
"""
import csv
import hashlib
import os
import sys
import time

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r22_common import EXP, JOBS   # noqa: E402

ROOT = "/home/DSJ44/paper2_yolov11_steel"
AUG = os.path.join(ROOT, "exp_apspc_dose_20260817")
OLD_A = os.path.join(ROOT, "exp_apspc_dose_unified_20260915")
OLD_G = os.path.join(ROOT, "exp_gc10_label_20260914")
EVALDIR = os.path.join(EXP, "runs_eval")

SELFCHECK_WEIGHT = os.path.join(
    OLD_A, "runs", "apspc_yolo11n_leaktrain_unified_s0", "weights", "best.pt")
SELFCHECK_REF = os.path.join(OLD_A, "dose_results_unified.csv")

NEU_VAL = [("val359", os.path.join(ROOT, "audits/leakage_sensitivity/neu_det_val359.yaml")),
           ("val360", os.path.join(ROOT, "exp_leakage_20260817/configs/neu_val360_controlled.yaml"))]

VAL_SETS = ["val_d0",
            "val_d20_same", "val_d28_same", "val_d41_same", "val_d82_same",
            "val_d20_conf", "val_d28_conf", "val_d41_conf", "val_d82_conf"]
VAL_YAML = {v: os.path.join(AUG, "configs", "apspc_%s.yaml" % v[len("val_"):])
            for v in VAL_SETS}

GC10_YAML = {"xml": os.path.join(OLD_G, "configs", "gc10_xml.yaml"),
             "dir": os.path.join(OLD_G, "configs", "gc10_dir.yaml")}

NEU_RUNS = [(seed, rn) for ds, cond, seed, tn, rn in JOBS if ds == "NEU"]
APSPC_RUNS = [(seed, rn) for ds, cond, seed, tn, rn in JOBS if ds == "APSPC"]
GC10_RUNS = [(cond, seed, rn) for ds, cond, seed, tn, rn in JOBS if ds == "GC10"]

EVAL_PARAMS = "imgsz=640, split='val', batch=16, device=0, verbose=False"


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def august_meta():
    meta = {}
    with open(os.path.join(AUG, "dose_results.csv"), "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            meta[r["val_set"]] = (r["dose_groups"], r["dose_images"],
                                  r["val_total_images"], r["label_variant"])
    return meta


def class_names_of(data_yaml):
    with open(data_yaml, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    nm = d.get("names")
    if isinstance(nm, dict):
        return [nm[k] for k in sorted(nm)]
    return list(nm)


def write_kwargs_source():
    src = os.path.join(OLD_A, "scripts", "eval_dose.py")
    L = []
    A = L.append
    A("eval_kwargs_source.txt — 第二十二轮 第6步 model.val(...) 参数来源")
    A("生成时间: %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    A("=" * 74)
    A("来源脚本(第十六轮 · exp_apspc_dose_unified_20260915),任务书指定第 88-90 行:")
    A("  %s" % src)
    A("行数: %d" % len(open(src, encoding="utf-8").read().splitlines()))
    A("sha256: %s" % sha256_file(src))
    A("=" * 74)
    A("")
    A("--------- 原文: eval_dose.py 第 87-92 行(含任务书指定的 88-90 行) ---------")
    for i, ln in enumerate(open(src, encoding="utf-8").read().splitlines(), 1):
        if 87 <= i <= 92:
            A("%4d| %s" % (i, ln))
    A("")
    A("=" * 74)
    A("本轮(round22)第6步实际使用的调用 —— 与上述参数完全相同,只改 data 与输出位置:")
    A("  模型类: 本轮 12 个 RT-DETR 模型用 ultralytics.RTDETR(<best.pt>);")
    A("          6.0 自检用的是 YOLO11n 权重,用 ultralytics.YOLO(<best.pt>)。")
    A("  r = model.val(data=<该次评估指定的 data yaml>, %s," % EVAL_PARAMS)
    A("                project=%s, name=\"<run>__on_<tag>\")" % EVALDIR)
    A("  map50 = float(r.box.map50);  map5095 = float(r.box.map)")
    A("")
    A("6.0 自检目标: %s(seed=0, val_d0)" % SELFCHECK_REF)
    A("6.1 NEU 换卷 yaml: %s" % [p for _, p in NEU_VAL])
    A("6.2 APSPC 9 个验证集 yaml 见 dose_val_yamls.txt")
    A("6.3 GC10 两个验证集 yaml 见 gc10_val_yamls.txt")
    with open(os.path.join(EXP, "eval_kwargs_source.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


# ======================================================================
# 6.0 自检
# ======================================================================
def step60():
    L = []
    A = L.append
    ok = True
    A("eval_selfcheck.txt — 第二十二轮(round22_rtdetr_20261010)第6.0步 评估流程自检")
    A("生成时间: %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    A("评估参数: model.val(data=..., %s)" % EVAL_PARAMS)
    A("          原文见 eval_kwargs_source.txt(eval_dose.py 第 88-90 行)")
    A("=" * 78)
    A("")
    A("[A] APSPC 原有 A 臂 YOLO11n 种子 0 · 在 d0 上重新评估")
    A("    weights = %s" % SELFCHECK_WEIGHT)
    A("    exists  = %s" % os.path.exists(SELFCHECK_WEIGHT))
    A("    sha256  = %s"
      % (sha256_file(SELFCHECK_WEIGHT) if os.path.exists(SELFCHECK_WEIGHT) else "<缺失>"))
    A("    参照文件= %s" % SELFCHECK_REF)

    ref = next((r for r in read_csv(SELFCHECK_REF)
                if r["val_set"] == "val_d0" and r["seed"] == "0"), None)
    from ultralytics import YOLO
    r = YOLO(SELFCHECK_WEIGHT).val(data=VAL_YAML["val_d0"], imgsz=640, split="val",
                                   batch=16, device=0, verbose=False,
                                   project=EVALDIR, name="selfcheck_apspc_A_s0__on_val_d0")
    m50, m5095 = float(r.box.map50), float(r.box.map)
    new50, new5095 = "%.6f" % m50, "%.6f" % m5095
    ref50 = ref["map50"] if ref else "<无对应行>"
    ref5095 = ref["map5095"] if ref else "<无对应行>"
    same50 = (new50 == ref50)
    same5095 = (new5095 == ref5095)
    A("")
    A("    ---- val_set = val_d0 ----")
    A("    dose_results_unified.csv 原值 : map50=%s  map5095=%s" % (ref50, ref5095))
    A("    本次重算值                    : map50=%s  map5095=%s" % (new50, new5095))
    A("    是否逐位相同                  : map50=%s  map5095=%s"
      % ("相同" if same50 else "不相同", "相同" if same5095 else "不相同"))
    if not (same50 and same5095):
        ok = False
    A("")
    A("=" * 78)
    A("自检结论: %s" % ("逐位相同 PASS" if ok else "不相同 FAIL"))
    A("后续评估: %s" % ("继续执行 6.1 / 6.2 / 6.3 / 6.4" if ok else "停止,不执行后续评估"))
    A("=" * 78)
    with open(os.path.join(EXP, "eval_selfcheck.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L), flush=True)
    return ok


def val_rtdetr(weight, data, tag):
    from ultralytics import RTDETR
    return RTDETR(weight).val(data=data, imgsz=640, split="val", batch=16,
                              device=0, verbose=False, project=EVALDIR, name=tag)


# ======================================================================
# 6.1 NEU 换卷
# ======================================================================
def step61():
    cols = ["seed", "run_name", "best_pt_sha256",
            "val359_map50", "val359_map5095", "val360_map50", "val360_map5095"]
    out = os.path.join(EXP, "neu_leak_rtdetr.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        f.flush()
        os.fsync(f.fileno())
    n = 0
    for seed, run_name in NEU_RUNS:
        wpath = os.path.join(EXP, run_name, "weights", "best.pt")
        sha = sha256_file(wpath) if os.path.exists(wpath) else ""
        print("\n[R22-6.1] %s seed=%d best.pt exists=%s sha256=%s"
              % (run_name, seed, os.path.exists(wpath), sha), flush=True)
        row = {"seed": seed, "run_name": run_name, "best_pt_sha256": sha}
        for tag, vyaml in NEU_VAL:
            r = val_rtdetr(wpath, vyaml, "%s__on_%s" % (run_name, tag))
            m50, m5095 = float(r.box.map50), float(r.box.map)
            row["%s_map50" % tag] = "%.6f" % m50
            row["%s_map5095" % tag] = "%.6f" % m5095
            print("[R22-6.1]   %-7s map50=%.6f map5095=%.6f" % (tag, m50, m5095), flush=True)
        with open(out, "a", newline="", encoding="utf-8") as f:
            ww = csv.DictWriter(f, fieldnames=cols)
            ww.writerow({k: row.get(k, "") for k in cols})
            f.flush()
            os.fsync(f.fileno())
        n += 1
    print("\n[R22-6.1] wrote %s rows=%d" % (out, n), flush=True)


# ======================================================================
# 6.2 APSPC 剂量
# ======================================================================
def step62(meta):
    cols = ["seed", "run_name", "val_set", "dose_groups", "dose_images",
            "val_total_images", "label_variant", "map50", "map5095",
            "best_pt_sha256"]
    out = os.path.join(EXP, "dose_results_rtdetr.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        f.flush()
        os.fsync(f.fileno())
    n = 0
    for seed, run_name in APSPC_RUNS:
        wpath = os.path.join(EXP, run_name, "weights", "best.pt")
        sha = sha256_file(wpath) if os.path.exists(wpath) else ""
        print("\n[R22-6.2] %s seed=%d best.pt exists=%s sha256=%s"
              % (run_name, seed, os.path.exists(wpath), sha), flush=True)
        for vset in VAL_SETS:
            r = val_rtdetr(wpath, VAL_YAML[vset], "%s__on_%s" % (run_name, vset))
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
            print("[R22-6.2]   %-14s map50=%.6f map5095=%.6f" % (vset, m50, m5095), flush=True)
    print("\n[R22-6.2] wrote %s rows=%d" % (out, n), flush=True)


# ======================================================================
# 6.3 GC10 2x2  +  6.4 分类别
# ======================================================================
def step63_64():
    cols = ["train_condition", "seed", "run_name", "best_pt_sha256",
            "eval_on", "map50", "map5095"]
    out = os.path.join(EXP, "gc10_label_results_rtdetr.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        f.flush()
        os.fsync(f.fileno())

    pc_cols = ["train_condition", "seed", "class_index", "class_name", "ap50", "ap5095"]
    pc_out = os.path.join(EXP, "gc10_per_class_rtdetr.csv")
    with open(pc_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=pc_cols)
        w.writeheader()
        f.flush()
        os.fsync(f.fileno())

    names = class_names_of(GC10_YAML["xml"])
    check = ["gc10_per_class_check.txt — 第二十二轮 第6.4步 一致性检查",
             "生成时间: %s" % time.strftime("%Y-%m-%d %H:%M:%S"),
             "检查内容: 每个模型在 XML 验证集上十类 ap5095 的平均值 是否等于 该模型总体 map5095",
             "=" * 78]
    n = 0
    for cond, seed, run_name in GC10_RUNS:
        wpath = os.path.join(EXP, run_name, "weights", "best.pt")
        sha = sha256_file(wpath) if os.path.exists(wpath) else ""
        print("\n[R22-6.3] %s condition=%s seed=%d best.pt exists=%s sha256=%s"
              % (run_name, cond, seed, os.path.exists(wpath), sha), flush=True)
        for target in ("xml", "dir"):
            r = val_rtdetr(wpath, GC10_YAML[target], "%s__on_%s" % (run_name, target))
            b = r.box
            m50, m5095 = float(b.map50), float(b.map)
            with open(out, "a", newline="", encoding="utf-8") as f:
                ww = csv.DictWriter(f, fieldnames=cols)
                ww.writerow({"train_condition": cond, "seed": seed, "run_name": run_name,
                             "best_pt_sha256": sha, "eval_on": target,
                             "map50": "%.6f" % m50, "map5095": "%.6f" % m5095})
                f.flush()
                os.fsync(f.fileno())
            n += 1
            print("[R22-6.3]   eval_on=%-3s map50=%.6f map5095=%.6f" % (target, m50, m5095),
                  flush=True)

            if target == "xml":
                ap50 = list(b.ap50)
                ap = list(b.ap)
                idx = [int(c) for c in list(b.ap_class_index)]
                got50 = {c: float(ap50[i]) for i, c in enumerate(idx)}
                got95 = {c: float(ap[i]) for i, c in enumerate(idx)}
                vals = []
                for ci in range(len(names)):
                    v50, v95 = got50.get(ci), got95.get(ci)
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
                mean_present = (sum(present) / len(present)) if present else float("nan")
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
    print("\n[R22-6.3] wrote %s rows=%d" % (out, n), flush=True)
    with open(os.path.join(EXP, "gc10_per_class_check.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(check) + "\n")
    print("\n".join(check), flush=True)


def main():
    os.makedirs(EVALDIR, exist_ok=True)
    print("[R22-EVAL] start=%s" % time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    write_kwargs_source()

    with open(os.path.join(EXP, "dose_val_yamls.txt"), "w", encoding="utf-8") as f:
        f.write("dose_val_yamls.txt — 第6.2步 APSPC 9 个验证集 data yaml(与原实验完全相同)\n")
        f.write("=" * 74 + "\n")
        for v in VAL_SETS:
            f.write("%-14s %s  exists=%s\n" % (v, VAL_YAML[v], os.path.exists(VAL_YAML[v])))
    with open(os.path.join(EXP, "gc10_val_yamls.txt"), "w", encoding="utf-8") as f:
        f.write("gc10_val_yamls.txt — 第6.3步 GC10 两个验证集 data yaml\n")
        f.write("=" * 74 + "\n")
        for k in ("xml", "dir"):
            f.write("%-4s %s  exists=%s\n" % (k, GC10_YAML[k], os.path.exists(GC10_YAML[k])))
    with open(os.path.join(EXP, "neu_val_yamls.txt"), "w", encoding="utf-8") as f:
        f.write("neu_val_yamls.txt — 第6.1步 NEU 两个验证集 data yaml\n")
        f.write("=" * 74 + "\n")
        for tag, p in NEU_VAL:
            f.write("%-7s %s  exists=%s\n" % (tag, p, os.path.exists(p)))

    if not step60():
        with open(os.path.join(EXP, "STOPPED_eval.txt"), "a", encoding="utf-8") as f:
            f.write("STOPPED_eval.txt — 第二十二轮 第6.0步 评估自检未通过\n")
            f.write("时间: %s\n" % time.strftime("%Y-%m-%d %H:%M:%S"))
            f.write("6.0 重算值与 dose_results_unified.csv 不逐位相同;\n")
            f.write("按任务书停止,不执行 6.1/6.2/6.3/6.4。\n")
        print("\n[R22-EVAL] 6.0 自检未通过 -> 停止", flush=True)
        return 1

    meta = august_meta()
    step61()
    step62(meta)
    step63_64()
    print("\n[R22-EVAL] end=%s" % time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
