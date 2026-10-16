#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Round 16 driver: 6 serial trainings + round-0 acceptance diff after each.

  * A arm first (leaktrain), then B arm (clean repro)
  * after EVERY run, diff <run>/args.yaml against the APSPC template
  * allowed differing keys: data, project, name, seed, save_dir
  * ANY other differing key -> STOP IMMEDIATELY (do not run remaining seeds)

Outputs: args_diff.txt, line_train_terminal.txt, optimizer_resolved.txt
Note: no resume is ever used; a failed run's directory is left for manual removal.
"""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ARM_RUNS, EXP, RUNS, TEMPLATE

PY = "/home/DSJ44/miniconda3/envs/yolov11/bin/python"
ALLOWED = {"data", "project", "name", "seed", "save_dir"}

DIFF_TXT = os.path.join(EXP, "args_diff.txt")
TERM_TXT = os.path.join(EXP, "line_train_terminal.txt")
OPT_TXT = os.path.join(EXP, "optimizer_resolved.txt")


def parse(p):
    d = {}
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if ":" in line and not line.startswith(" "):
                k, v = line.split(":", 1)
                d[k.strip()] = v.strip()
    return d


def append(p, t):
    with open(p, "a", encoding="utf-8") as f:
        f.write(t)


def main():
    for p in (DIFF_TXT, TERM_TXT, OPT_TXT):
        open(p, "w", encoding="utf-8").close()
    os.makedirs(RUNS, exist_ok=True)
    os.makedirs(os.path.join(EXP, "logs"), exist_ok=True)

    append(TERM_TXT, "=== line_train_terminal.txt — exp_apspc_dose_unified_20260915 ===\n")
    append(TERM_TXT, "driver_start=%s\npython=%s\ntemplate=%s\n\n"
           % (time.strftime("%Y-%m-%d %H:%M:%S"), PY, TEMPLATE))

    t_all = time.time()
    for i, (arm, seed, data_yaml, run_name) in enumerate(ARM_RUNS, 1):
        run_dir = os.path.join(RUNS, run_name)
        run_args = os.path.join(run_dir, "args.yaml")
        logf = os.path.join(EXP, "logs", run_name + ".log")

        hdr = ("\n" + "=" * 78 +
               "\n[T16] job %d/6  %s  arm=%s seed=%d  start=%s\n"
               % (i, run_name, arm, seed, time.strftime("%Y-%m-%d %H:%M:%S")) +
               "=" * 78 + "\n")
        print(hdr, flush=True)
        append(TERM_TXT, hdr)

        cmd = [PY, os.path.join(EXP, "scripts", "train_one.py"),
               run_name, str(seed), arm, data_yaml]
        t0 = time.time()
        with open(logf, "w", encoding="utf-8") as lf:
            p = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
        rc = p.returncode
        wall = time.time() - t0

        with open(logf, "r", encoding="utf-8", errors="replace") as lf:
            txt = lf.read()
        append(TERM_TXT, txt)

        line = ("[T16] job %d/6 %s exit_code=%d wall_sec=%.1f end=%s\n"
                % (i, run_name, rc, wall, time.strftime("%Y-%m-%d %H:%M:%S")))
        print(line, flush=True)
        append(TERM_TXT, line)

        # optimizer resolved line
        opt = "<not found>"
        for ln in txt.splitlines():
            if "optimizer:" in ln.lower():
                opt = ln.strip()
                break
        append(OPT_TXT, "%s\n%s\n\n" % (run_name, opt))
        print("[T16] optimizer_resolved: %s" % opt, flush=True)

        # ---- round-0 acceptance gate ----
        if not os.path.exists(run_args):
            msg = "[T16] GATE: args.yaml 缺失 -> %s ; 停止\n" % run_args
            append(DIFF_TXT, msg); append(TERM_TXT, msg)
            print(msg, flush=True)
            return 2

        a, b = parse(run_args), parse(TEMPLATE)
        viol, allowed = [], []
        for k in sorted(set(a) | set(b)):
            x, y = a.get(k, "<MISSING>"), b.get(k, "<MISSING>")
            if x == y:
                continue
            if k in ALLOWED:
                allowed.append((k, y, x))
            else:
                viol.append((k, y, x))

        p2 = subprocess.run(["diff", run_args, TEMPLATE], capture_output=True, text=True)
        blk = ["=" * 78,
               "run      : %s" % run_name,
               "diff %s" % run_args,
               "     %s" % TEMPLATE,
               "exit_code=%d" % p2.returncode,
               p2.stdout.rstrip("\n") if p2.stdout else "(no differences)",
               "--- 允许差异键 (data/project/name/seed/save_dir) 实际命中: %d" % len(allowed)]
        for k, y, x in allowed:
            blk.append("      ALLOWED %s: 模板=%r 本轮=%r" % (k, y, x))
        blk.append("--- 违规差异行数: %d" % len(viol))
        for k, y, x in viol:
            blk.append("      VIOLATION %s: 模板=%r 本轮=%r" % (k, y, x))
        blk.append("--- GATE 结果: %s" % ("PASS" if not viol else "FAIL"))
        blk.append("=" * 78 + "\n")
        btxt = "\n".join(blk) + "\n"
        print(btxt, flush=True)
        append(DIFF_TXT, btxt)
        append(TERM_TXT, btxt)

        if viol:
            msg = "[T16] GATE FAIL -> 立即停止,不执行后续 %d 次训练\n" % (len(ARM_RUNS) - i)
            print(msg, flush=True)
            append(DIFF_TXT, msg); append(TERM_TXT, msg)
            return 3

    tot = time.time() - t_all
    msg = "\n[T16] all 6 jobs finished total_elapsed_sec=%.1f (%s)\n" % (
        tot, time.strftime("%Y-%m-%d %H:%M:%S"))
    print(msg, flush=True)
    append(TERM_TXT, msg); append(DIFF_TXT, msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
