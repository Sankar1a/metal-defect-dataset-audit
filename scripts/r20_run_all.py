#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Round 20 串行队列驱动 —— 18 次 YOLOv8n/YOLOv8s 训练 + 每次训练后立即执行第3步。

做法沿用第十七轮 round17_yolo11m_20261005/scripts/r17_run_all.py
(该批 9 次训练参数闸门全部 PASS),每次训练用独立子进程,单进程串行,device=0。

顺序: NEU-DET 6 个 -> GC10-DET 6 个 -> APSPC 6 个;
      每个数据集内先 v8n 后 v8s,种子顺序 0、1、42。

每次训练结束后立即:
  3a GATE   : diff <模板 args.yaml> round20_yolov8_20261008/<name>/args.yaml > gates/<name>.diff
              只允许 model / project / name / save_dir 四个字段差异;
              出现第五个字段差异 -> gates/<name>.txt 写 FAIL 及字段 -> 立即停止队列
  3b 优化器 : grep -i 'optimizer:' logs/<name>.log | head -3 > optimizer/<name>.txt
              需包含 SGD(lr=0.01, momentum=0.937);不包含 -> 停止队列
  3c 轮数   : results.csv 数据行数
  3d 实测   : best.pt 在本 run 自己的 data yaml 上 model.val(...),取 r.box.map50 / r.box.map
              另取 model.info() 的参数量与 GFLOPs
  3e 追加   : yolov8_results.csv 一行并 flush

训练中途报错退出(非闸门原因)-> 记录后停止队列。

用法: nohup python -u r20_run_all.py >> logs/_driver.log 2>&1 &
"""
import csv
import json
import os
import subprocess
import sys
import time

import yaml

ROOT = "/home/DSJ44/paper2_yolov11_steel"
EXP = os.path.join(ROOT, "round20_yolov8_20261008")
TEMPLATE_ROOT = os.path.join(ROOT, "runs_unified_20260803")
PY = "/home/DSJ44/miniconda3/envs/yolov11/bin/python"
SCRIPTS = os.path.join(EXP, "scripts")

GATES = os.path.join(EXP, "gates")
OPTIMIZER = os.path.join(EXP, "optimizer")
LOGS = os.path.join(EXP, "logs")
EVALLOGS = os.path.join(EXP, "logs_eval")
OUTCSV = os.path.join(EXP, "yolov8_results.csv")

ALLOWED = {"model", "project", "name", "save_dir"}
NUMEQ = {"warmup_epochs"}          # 3 vs 3.0 之类,数值等价,记录但不判 FAIL

COLS = ["dataset", "size", "seed", "run_name", "epochs_run", "best_pt_sha256",
        "map50", "map5095", "params", "gflops", "gate", "optimizer_line"]

DSKEY = {"NEU-DET": "neu", "GC10-DET": "gc10", "APSPC": "apspc"}

JOBS = []
for _ds in ("NEU-DET", "GC10-DET", "APSPC"):
    for _sz in ("n", "s"):
        for _sd in (0, 1, 42):
            JOBS.append((_ds, _sz, _sd))


def run_name_for(dataset, size, seed):
    return "%s_yolov8%s_s%d" % (DSKEY[dataset], size, seed)


def template_for(dataset, size, seed):
    return os.path.join(TEMPLATE_ROOT, "%s_yolo11%s_s%d" % (DSKEY[dataset], size, seed),
                        "args.yaml")


def num_eq(a, b):
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return False


def count_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    return max(0, len(lines) - 1)


def gate_check(run_args_path, template_path, name):
    """3a: 写 gates/<name>.diff,返回 (verdict, violations, numeq, allowed_hit, raw)"""
    with open(run_args_path, "r", encoding="utf-8") as f:
        run_a = yaml.safe_load(f)
    with open(template_path, "r", encoding="utf-8") as f:
        tmpl_a = yaml.safe_load(f)

    viol, numeq, allowed_hit = [], [], []
    for k in sorted(set(run_a) | set(tmpl_a)):
        a, b = run_a.get(k, "<MISSING>"), tmpl_a.get(k, "<MISSING>")
        if a == b:
            continue
        if k in ALLOWED:
            allowed_hit.append((k, b, a))
            continue
        if k in NUMEQ and num_eq(a, b):
            numeq.append((k, b, a))
            continue
        viol.append((k, b, a))

    p = subprocess.run(["diff", template_path, run_args_path],
                       capture_output=True, text=True)
    with open(os.path.join(GATES, name + ".diff"), "w", encoding="utf-8") as f:
        f.write("# diff %s %s\n" % (template_path, run_args_path))
        f.write("# exit_code=%d\n" % p.returncode)
        f.write(p.stdout if p.stdout else "(no differences)\n")

    verdict = "PASS" if not viol else "FAIL"
    with open(os.path.join(GATES, name + ".txt"), "w", encoding="utf-8") as f:
        f.write("GATE %s\n" % verdict)
        f.write("run_args=%s\n" % run_args_path)
        f.write("template=%s\n" % template_path)
        f.write("允许差异字段(model/project/name/save_dir)-> %d 个:\n" % len(allowed_hit))
        for k, b, a in allowed_hit:
            f.write("    ALLOWED %-9s 模板=%r 本次=%r\n" % (k, b, a))
        f.write("数值等价例外(不判 FAIL)-> %s\n" % (numeq if numeq else "无"))
        f.write("第五个字段差异(违规)-> %d 个\n" % len(viol))
        for k, b, a in viol:
            f.write("    VIOLATION %-9s 模板=%r 本次=%r\n" % (k, b, a))
        f.write("GATE 结果: %s\n" % verdict)
    return verdict, viol, numeq, allowed_hit, (p.stdout or "")


def append_csv(row):
    newfile = not os.path.exists(OUTCSV)
    with open(OUTCSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if newfile:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in COLS})
        f.flush()
        os.fsync(f.fileno())


def main():
    for d in (GATES, OPTIMIZER, LOGS, EVALLOGS):
        os.makedirs(d, exist_ok=True)

    print("[R20] driver start=%s" % time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    print("[R20] python=%s" % PY, flush=True)
    print("[R20] EXP=%s" % EXP, flush=True)
    print("[R20] 顺序: NEU-DET(n,s) -> GC10-DET(n,s) -> APSPC(n,s),每种种子 0/1/42", flush=True)
    print("[R20] 共 %d 次" % len(JOBS), flush=True)

    t_all = time.time()
    for i, (dataset, size, seed) in enumerate(JOBS, 1):
        name = run_name_for(dataset, size, seed)
        tpath = template_for(dataset, size, seed)
        run_dir = os.path.join(EXP, name)
        run_args = os.path.join(run_dir, "args.yaml")
        logf = os.path.join(LOGS, name + ".log")

        hdr = ("\n" + "=" * 78 +
               "\n[R20] job %d/%d  %s (dataset=%s size=%s seed=%d)  start=%s\n"
               % (i, len(JOBS), name, dataset, size, seed,
                  time.strftime("%Y-%m-%d %H:%M:%S")) +
               "=" * 78)
        print(hdr, flush=True)

        # ---------------- training ----------------
        cmd = [PY, os.path.join(SCRIPTS, "r20_train_one.py"), name,
               "--project", EXP, "--log", logf]
        with open(logf, "w", encoding="utf-8") as lf:
            lf.write("[R20] cmd=%s\n" % " ".join(cmd))
            lf.flush()
            proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
        rc = proc.returncode
        print("[R20] job %d/%d %s train exit_code=%d end=%s"
              % (i, len(JOBS), name, rc, time.strftime("%Y-%m-%d %H:%M:%S")), flush=True)

        with open(logf, "r", encoding="utf-8", errors="replace") as lf:
            logtxt = lf.read()

        # ---------------- 3b optimizer ----------------
        opt_hits = [ln for ln in logtxt.splitlines() if "optimizer:" in ln.lower()][:3]
        with open(os.path.join(OPTIMIZER, name + ".txt"), "w", encoding="utf-8") as f:
            f.write("grep -i 'optimizer:' logs/%s.log | head -3\n" % name)
            f.write("命中行数=%d\n" % len(opt_hits))
            f.write("---\n")
            for ln in opt_hits:
                f.write(ln + "\n")
            if not opt_hits:
                f.write("<none>\n")
        optimizer_line = opt_hits[0].strip() if opt_hits else ""
        print("[R20] 3b optimizer_line=%r" % optimizer_line, flush=True)
        opt_ok = ("SGD(lr=0.01, momentum=0.937)" in optimizer_line)

        # ---------------- 3a GATE ----------------
        if not os.path.exists(run_args):
            with open(os.path.join(GATES, name + ".txt"), "w", encoding="utf-8") as f:
                f.write("GATE FAIL\nargs.yaml 不存在: %s\ntrain exit_code=%d\n"
                        % (run_args, rc))
            print("[R20] 3a GATE FAIL: args.yaml 不存在 -> 停止队列,不启动下一次", flush=True)
            append_csv({"dataset": dataset, "size": size, "seed": seed, "run_name": name,
                        "epochs_run": 0, "best_pt_sha256": "", "map50": "", "map5095": "",
                        "params": "", "gflops": "", "gate": "FAIL",
                        "optimizer_line": optimizer_line})
            return 2

        verdict, viol, numeq, allowed_hit, rawdiff = gate_check(run_args, tpath, name)
        print("[R20] 3a GATE=%s allowed=%s numeq=%s violations=%s"
              % (verdict, [x[0] for x in allowed_hit], numeq, viol), flush=True)

        # ---------------- 3c epochs ----------------
        results_csv = os.path.join(run_dir, "results.csv")
        epochs_run = count_rows(results_csv) if os.path.exists(results_csv) else 0
        print("[R20] 3c epochs_run=%d" % epochs_run, flush=True)

        # ---------------- 3d best.pt 实测 ----------------
        rec = {}
        if os.path.exists(os.path.join(run_dir, "weights", "best.pt")):
            eval_log = os.path.join(EVALLOGS, name + ".log")
            ev_cmd = [PY, os.path.join(SCRIPTS, "r20_eval_one.py"), name]
            with open(eval_log, "w", encoding="utf-8") as lf:
                lf.write("[R20] cmd=%s\n" % " ".join(ev_cmd))
                lf.flush()
                evp = subprocess.run(ev_cmd, stdout=lf, stderr=subprocess.STDOUT)
            print("[R20] 3d eval exit_code=%d log=%s" % (evp.returncode, eval_log), flush=True)
            jp = os.path.join(EXP, "_eval", name + ".json")
            if os.path.exists(jp):
                with open(jp, "r", encoding="utf-8") as f:
                    rec = json.load(f)
        else:
            print("[R20] 3d SKIP: best.pt 缺失", flush=True)
        print("[R20] 3d map50=%s map5095=%s params=%s gflops=%s"
              % (rec.get("map50"), rec.get("map5095"), rec.get("params"), rec.get("gflops")),
              flush=True)

        # ---------------- 3e append row ----------------
        gate_val = "TRAIN_ERROR" if rc != 0 else verdict
        append_csv({
            "dataset": dataset, "size": size, "seed": seed, "run_name": name,
            "epochs_run": epochs_run,
            "best_pt_sha256": rec.get("best_pt_sha256", ""),
            "map50": ("%.6f" % rec["map50"]) if rec.get("map50") is not None else "",
            "map5095": ("%.6f" % rec["map5095"]) if rec.get("map5095") is not None else "",
            "params": rec.get("params", ""),
            "gflops": rec.get("gflops", ""),
            "gate": gate_val,
            "optimizer_line": optimizer_line,
        })
        print("[R20] 3e appended row -> %s" % OUTCSV, flush=True)

        # ---------------- 停止条件 ----------------
        if rc != 0:
            print("[R20] 训练报错退出 exit_code=%d(非闸门原因)-> 停止队列,不启动下一次" % rc,
                  flush=True)
            return 5
        if verdict == "FAIL":
            print("[R20] GATE FAIL -> 立即停止整个队列,不启动下一次训练", flush=True)
            return 3
        if not opt_ok:
            print("[R20] 优化器解析行不含 SGD(lr=0.01, momentum=0.937) -> 停止队列", flush=True)
            return 4

    total = time.time() - t_all
    print("\n[R20] all %d jobs finished, total_elapsed_sec=%.1f (%s)"
          % (len(JOBS), total, time.strftime("%Y-%m-%d %H:%M:%S")), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
