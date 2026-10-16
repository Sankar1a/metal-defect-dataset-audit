#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Line EC driver: run the 6 trainings serially + enforce the §2 args.yaml gate.

  * runs exactly one training at a time (subprocess per run)
  * after every run, diffs <run>/args.yaml against runs_unified_20260803/gc10_yolo11n_s0/args.yaml
  * allowed differing keys: data, project, name, seed, save_dir
  * warmup_epochs 3.0 vs 3 is numerically equivalent -> allowed but flagged
  * any other differing key -> STOP IMMEDIATELY, do not run remaining trainings

Writes: exp_gc10_label_20260914/args_diff.txt
        exp_gc10_label_20260914/line_EC_terminal.txt
        exp_gc10_label_20260914/optimizer_resolved.txt
"""
import os
import subprocess
import sys
import time

ROOT = "/home/DSJ44/paper2_yolov11_steel"
EXP = os.path.join(ROOT, "exp_gc10_label_20260914")
PY = "/home/DSJ44/miniconda3/envs/yolov11/bin/python"
TEMPLATE = os.path.join(ROOT, "runs_unified_20260803", "gc10_yolo11n_s0", "args.yaml")

ALLOWED = {"data", "project", "name", "seed", "save_dir"}
NUMEQ_ALLOWED = {"warmup_epochs"}   # 3.0 == 3

DIFF_TXT = os.path.join(EXP, "args_diff.txt")
EC_TXT = os.path.join(EXP, "line_EC_terminal.txt")
OPT_TXT = os.path.join(EXP, "optimizer_resolved.txt")

JOBS = [("xml", 0), ("xml", 1), ("xml", 42),
        ("dir", 0), ("dir", 1), ("dir", 42)]


def parse_args_yaml(path):
    d = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if ":" in line and not line.startswith(" "):
                k, v = line.split(":", 1)
                d[k.strip()] = v.strip()
    return d


def num_eq(a, b):
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return False


def check_diff(run_args, tmpl_args):
    """Return (violations, numeq_flags, raw_diff)."""
    viol, numeq = [], []
    keys = sorted(set(run_args) | set(tmpl_args))
    for k in keys:
        a, b = run_args.get(k, "<MISSING>"), tmpl_args.get(k, "<MISSING>")
        if a == b:
            continue
        if k in ALLOWED:
            continue
        if k in NUMEQ_ALLOWED and num_eq(a, b):
            numeq.append((k, b, a))
            continue
        viol.append((k, b, a))
    # raw unified diff for the record
    p = subprocess.run(["diff", run_args_path[0], TEMPLATE], capture_output=True, text=True)
    return viol, numeq, p.stdout


def append(path, text):
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)


def main():
    open(DIFF_TXT, "w", encoding="utf-8").close()
    open(EC_TXT, "w", encoding="utf-8").close()
    open(OPT_TXT, "w", encoding="utf-8").close()

    append(EC_TXT, "=== line_EC_terminal.txt — exp_gc10_label_20260914 ===\n")
    append(EC_TXT, "driver_start=%s\n" % time.strftime("%Y-%m-%d %H:%M:%S"))
    append(EC_TXT, "python=%s\n" % PY)
    append(EC_TXT, "template=%s\n\n" % TEMPLATE)

    t_all = time.time()
    for i, (cond, seed) in enumerate(JOBS, 1):
        name = "gc10_yolo11n_%s_s%d" % (cond, seed)
        run_dir = os.path.join(EXP, "runs", name)
        run_args = os.path.join(run_dir, "args.yaml")
        logf = os.path.join(EXP, "logs", name + ".log")
        os.makedirs(os.path.dirname(logf), exist_ok=True)
        os.makedirs(os.path.join(EXP, "runs"), exist_ok=True)

        header = ("\n" + "=" * 78 +
                  "\n[EC] job %d/6  %s   start=%s\n" % (i, name, time.strftime("%Y-%m-%d %H:%M:%S")) +
                  "=" * 78 + "\n")
        print(header, flush=True)
        append(EC_TXT, header)

        cmd = [PY, os.path.join(EXP, "scripts", "ec_train_one.py"), cond, str(seed),
               "--project", os.path.join(EXP, "runs"), "--log", logf]
        with open(logf, "w", encoding="utf-8") as lf:
            p = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
        rc = p.returncode

        # echo the run log into the EC terminal transcript
        with open(logf, "r", encoding="utf-8", errors="replace") as lf:
            txt = lf.read()
        append(EC_TXT, txt)

        line = "[EC] job %d/6 %s exit_code=%d end=%s\n" % (i, name, rc, time.strftime("%Y-%m-%d %H:%M:%S"))
        print(line, flush=True)
        append(EC_TXT, line)

        # ---- optimizer resolved line ----
        opt_line = "<not found>"
        for ln in txt.splitlines():
            if "optimizer:" in ln.lower():
                opt_line = ln.strip()
                break
        append(OPT_TXT, "%s\n%s\n\n" % (name, opt_line))
        print("[EC] optimizer_resolved: %s" % opt_line, flush=True)

        # ---- §2 args.yaml gate ----
        if not os.path.exists(run_args):
            msg = "[EC] GATE: args.yaml 不存在 -> %s ; 停止,不跑剩余训练\n" % run_args
            print(msg, flush=True)
            append(DIFF_TXT, msg)
            append(EC_TXT, msg)
            return 2

        global run_args_path
        run_args_path = (run_args,)
        viol, numeq, raw = check_diff(parse_args_yaml(run_args), parse_args_yaml(TEMPLATE))

        blk = ["=" * 78,
               "diff %s \\\n     %s" % (run_args, TEMPLATE),
               "exit_code=%d" % (0 if not raw else 1),
               raw.rstrip("\n") if raw else "(no differences)",
               "--- 允许差异键(data/project/name/seed/save_dir):已自动放行",
               "--- 数值等价例外: %s" % (numeq if numeq else "无"),
               "--- 违规差异: %d" % len(viol)]
        for k, b, a in viol:
            blk.append("    VIOLATION %s: 模板=%r 本次=%r" % (k, b, a))
        blk.append("--- GATE 结果: %s" % ("PASS" if not viol else "FAIL"))
        blk.append("=" * 78 + "\n")
        btxt = "\n".join(blk) + "\n"
        print(btxt, flush=True)
        append(DIFF_TXT, btxt)
        append(EC_TXT, btxt)

        if viol:
            msg = ("[EC] GATE FAIL -> 立即停止,不执行后续 %d 次训练\n" % (len(JOBS) - i))
            print(msg, flush=True)
            append(DIFF_TXT, msg)
            append(EC_TXT, msg)
            return 3

    total = time.time() - t_all
    msg = "\n[EC] all 6 jobs finished, total_elapsed_sec=%.1f (%s)\n" % (
        total, time.strftime("%Y-%m-%d %H:%M:%S"))
    print(msg, flush=True)
    append(EC_TXT, msg)
    append(DIFF_TXT, msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
