#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Round 22 第5步 · 12 次训练串行驱动。

顺序: NEU(0,1,42) -> APSPC A 臂(0,1,42) -> GC10 XML(0,1,42) -> GC10 DIR(0,1,42)。
单进程串行, device=0, 每次训练输出到 logs/<name>.log。

每次训练结束后立即:
  闸门     : diff <模板 args.yaml> <本次 args.yaml> > gates/<name>.diff
             只允许第2步表格的 10 个字段不同;出现其它字段 -> gates/<name>.txt 写 FAIL,停止整个链
  优化器行 : grep -i 'optimizer:' logs/<name>.log | head -3 > optimizer/<name>.txt
             须含 AdamW(lr=0.0001, momentum=0.9),否则停止
  确定性告警: grep -i -E 'determinis|warn' logs/<name>.log | head -20 > determinism/<name>.txt(只记录)
  追加一行 : train_results.csv 并 flush:
             dataset, condition, seed, run_name, template_run, epochs_run,
             best_pt_sha256, gate, optimizer_line, train_seconds
             (train_seconds = results.csv 末行 time 列,与第十六轮 runs_meta.csv 同口径)

用法: python -u r22_run_all.py >> logs/_driver.log 2>&1
"""
import csv
import hashlib
import os
import subprocess
import sys
import time

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r22_common import (ALLOWED, EXP, JOBS, NUMEQ, OPTIMIZER_EXPECT,  # noqa: E402
                        PY, SCRIPTS, num_eq, read_decision, template_path)

GATES = os.path.join(EXP, "gates")
OPTIMIZER = os.path.join(EXP, "optimizer")
DETERMINISM = os.path.join(EXP, "determinism")
LOGS = os.path.join(EXP, "logs")
OUTCSV = os.path.join(EXP, "train_results.csv")

COLS = ["dataset", "condition", "seed", "run_name", "template_run", "epochs_run",
        "best_pt_sha256", "gate", "optimizer_line", "train_seconds"]


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def count_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    return max(0, len(lines) - 1)


def last_time_seconds(path):
    """results.csv 末行 time 列(累计训练秒数),与第十六轮 train_seconds 同口径。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return ""
        for k in rows[-1]:
            if k.strip() == "time":
                return "%.1f" % float(rows[-1][k])
    except Exception:                                           # noqa: BLE001
        pass
    return ""


def gate_check(run_args_path, tpath, name):
    with open(run_args_path, "r", encoding="utf-8") as f:
        run_a = yaml.safe_load(f)
    with open(tpath, "r", encoding="utf-8") as f:
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

    p = subprocess.run(["diff", tpath, run_args_path], capture_output=True, text=True)
    with open(os.path.join(GATES, name + ".diff"), "w", encoding="utf-8") as f:
        f.write("# diff %s %s\n" % (tpath, run_args_path))
        f.write("# exit_code=%d\n" % p.returncode)
        f.write(p.stdout if p.stdout else "(no differences)\n")

    verdict = "PASS" if not viol else "FAIL"
    with open(os.path.join(GATES, name + ".txt"), "w", encoding="utf-8") as f:
        f.write("GATE %s\n" % verdict)
        f.write("run_args=%s\n" % run_args_path)
        f.write("template=%s\n" % tpath)
        f.write("允许差异字段(第2步表格 model/project/name/save_dir/optimizer/lr0/"
                "momentum/weight_decay/epochs/batch)-> %d 个:\n" % len(allowed_hit))
        for k, b, a in allowed_hit:
            f.write("    ALLOWED %-13s 模板=%r 本次=%r\n" % (k, b, a))
        f.write("数值等价例外(不判 FAIL)-> %s\n" % (numeq if numeq else "无"))
        f.write("其它字段差异(违规)-> %d 个\n" % len(viol))
        for k, b, a in viol:
            f.write("    VIOLATION %-13s 模板=%r 本次=%r\n" % (k, b, a))
        f.write("data 是否被改动: %s\n"
                % ("否(一致)" if run_a.get("data") == tmpl_a.get("data") else "是(不一致)"))
        f.write("seed 是否被改动: %s\n"
                % ("否(一致)" if run_a.get("seed") == tmpl_a.get("seed") else "是(不一致)"))
        f.write("GATE 结果: %s\n" % verdict)
    return verdict, viol, numeq, allowed_hit


def append_csv(row):
    newfile = not os.path.exists(OUTCSV)
    with open(OUTCSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if newfile:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in COLS})
        f.flush()
        os.fsync(f.fileno())


def write_stopped(step, lines):
    p = os.path.join(EXP, "STOPPED_%s.txt" % step)
    with open(p, "a", encoding="utf-8") as f:
        f.write("STOPPED_%s.txt — 第二十二轮 第5步 训练阶段终止\n" % step)
        f.write("时间: %s\n" % time.strftime("%Y-%m-%d %H:%M:%S"))
        for ln in lines:
            f.write(ln + "\n")
        f.write("后续训练与第6/7步全部不执行。\n")
    print("[R22] 已写 %s" % p, flush=True)
    for ln in lines:
        print("[R22]   %s" % ln, flush=True)


def main():
    for d in (GATES, OPTIMIZER, DETERMINISM, LOGS):
        os.makedirs(d, exist_ok=True)
    dec = read_decision()

    print("[R22] driver start=%s" % time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    print("[R22] EXP=%s" % EXP, flush=True)
    print("[R22] decision: batch=%s epochs=%s est_hours=%s"
          % (dec.get("batch"), dec.get("epochs"), dec.get("estimated_total_hours")), flush=True)
    print("[R22] 顺序: NEU(0,1,42) -> APSPC-A(0,1,42) -> GC10-XML(0,1,42) -> GC10-DIR(0,1,42)",
          flush=True)
    print("[R22] 共 %d 次" % len(JOBS), flush=True)

    t_all = time.time()
    for i, (ds, cond, seed, tname, name) in enumerate(JOBS, 1):
        tpath = template_path(name)
        run_dir = os.path.join(EXP, name)
        run_args = os.path.join(run_dir, "args.yaml")
        logf = os.path.join(LOGS, name + ".log")

        print("\n" + "=" * 78, flush=True)
        print("[R22] job %d/%d  %s (dataset=%s condition=%s seed=%d template=%s)  start=%s"
              % (i, len(JOBS), name, ds, cond, seed, tname,
                 time.strftime("%Y-%m-%d %H:%M:%S")), flush=True)
        print("=" * 78, flush=True)

        cmd = [PY, "-u", os.path.join(SCRIPTS, "r22_train_one.py"), name,
               "--project", EXP, "--log", logf]
        with open(logf, "w", encoding="utf-8") as lf:
            lf.write("[R22] cmd=%s\n" % " ".join(cmd))
            lf.flush()
            proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
        rc = proc.returncode
        print("[R22] job %d/%d %s train exit_code=%d end=%s"
              % (i, len(JOBS), name, rc, time.strftime("%Y-%m-%d %H:%M:%S")), flush=True)

        with open(logf, "r", encoding="utf-8", errors="replace") as lf:
            logtxt = lf.read()

        # ---- 优化器行 ----
        opt_hits = [ln for ln in logtxt.splitlines() if "optimizer:" in ln.lower()][:3]
        with open(os.path.join(OPTIMIZER, name + ".txt"), "w", encoding="utf-8") as f:
            f.write("grep -i 'optimizer:' logs/%s.log | head -3\n" % name)
            f.write("命中行数=%d\n---\n" % len(opt_hits))
            for ln in opt_hits:
                f.write(ln + "\n")
            if not opt_hits:
                f.write("<none>\n")
        optimizer_line = opt_hits[0].strip() if opt_hits else ""
        print("[R22] optimizer_line=%r" % optimizer_line, flush=True)
        opt_ok = OPTIMIZER_EXPECT in optimizer_line

        # ---- 确定性告警(只记录,不处理) ----
        det = [ln for ln in logtxt.splitlines()
               if ("determinis" in ln.lower() or "warn" in ln.lower())][:20]
        with open(os.path.join(DETERMINISM, name + ".txt"), "w", encoding="utf-8") as f:
            f.write("grep -i -E 'determinis|warn' logs/%s.log | head -20\n" % name)
            f.write("命中行数=%d\n---\n" % len(det))
            for ln in det:
                f.write(ln + "\n")
            if not det:
                f.write("<none>\n")
        print("[R22] determinism/warn 命中行数=%d(只记录)" % len(det), flush=True)

        # ---- 闸门 ----
        if not os.path.exists(run_args):
            write_stopped("train", ["args.yaml 不存在: %s" % run_args,
                                    "train exit_code=%d" % rc])
            append_csv({"dataset": ds, "condition": cond, "seed": seed, "run_name": name,
                        "template_run": tname, "epochs_run": 0, "best_pt_sha256": "",
                        "gate": "FAIL", "optimizer_line": optimizer_line, "train_seconds": ""})
            return 2
        verdict, viol, numeq, allowed_hit = gate_check(run_args, tpath, name)
        print("[R22] GATE=%s allowed=%s numeq=%s violations=%s"
              % (verdict, sorted(x[0] for x in allowed_hit), numeq, viol), flush=True)

        # ---- 轮数 / best.pt / train_seconds ----
        results_csv = os.path.join(run_dir, "results.csv")
        epochs_run = count_rows(results_csv) if os.path.exists(results_csv) else 0
        best = os.path.join(run_dir, "weights", "best.pt")
        best_sha = sha256_file(best) if os.path.exists(best) else ""
        train_seconds = last_time_seconds(results_csv) if os.path.exists(results_csv) else ""
        print("[R22] epochs_run=%d best.pt exists=%s sha256=%s train_seconds=%s"
              % (epochs_run, os.path.exists(best), best_sha, train_seconds), flush=True)

        gate_val = "TRAIN_ERROR" if rc != 0 else verdict
        append_csv({"dataset": ds, "condition": cond, "seed": seed, "run_name": name,
                    "template_run": tname, "epochs_run": epochs_run,
                    "best_pt_sha256": best_sha, "gate": gate_val,
                    "optimizer_line": optimizer_line, "train_seconds": train_seconds})
        print("[R22] appended row -> %s" % OUTCSV, flush=True)

        # ---- 停止条件 ----
        if rc != 0:
            write_stopped("train", ["训练报错退出(非闸门原因): %s exit_code=%d" % (name, rc),
                                    "日志: %s" % logf])
            return 5
        if verdict == "FAIL":
            write_stopped("gate", ["参数闸门 FAIL: %s" % name,
                                   "违规字段: %s" % viol,
                                   "详见 gates/%s.txt" % name])
            return 3
        if not opt_ok:
            write_stopped("optimizer", ["优化器解析行不含 %r: %s" % (OPTIMIZER_EXPECT, name),
                                        "实际 optimizer 行: %s" % optimizer_line,
                                        "详见 optimizer/%s.txt" % name])
            return 4

    total = time.time() - t_all
    print("\n[R22] all %d jobs finished, total_elapsed_sec=%.1f (%s)"
          % (len(JOBS), total, time.strftime("%Y-%m-%d %H:%M:%S")), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
