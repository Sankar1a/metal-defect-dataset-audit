#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Round 21 串行队列驱动 —— 9 次 YOLOv8n 训练 + 每次训练后立即执行第4步。

做法沿用第二十轮 round20_yolov8_20261008/scripts/r20_run_all.py,
每次训练用独立子进程,单进程串行, device=0。

顺序: A 组 3 个(种子 0、1、42)-> B 组 XML 3 个 -> B 组 DIR 3 个。

每次训练结束后立即:
  4a GATE   : diff <模板 args.yaml> <本次 args.yaml> > gates/<name>.diff
              只允许 model / project / name / save_dir 四个字段差异;
              出现第五个字段差异 -> gates/<name>.txt 写 FAIL 及字段 -> 立即停止队列
  4b 优化器 : grep -i 'optimizer:' logs/<name>.log | head -3 > optimizer/<name>.txt
              须含 SGD(lr=0.01, momentum=0.937);否则停止队列
  4c 轮数   : results.csv 数据行数
  4d 追加一行到 train_results.csv 并 flush:
              group, condition, seed, run_name, template_run, epochs_run,
              best_pt_sha256, gate, optimizer_line

用法: nohup python -u r21_run_all.py >> logs/_driver.log 2>&1 &
"""
import csv
import hashlib
import os
import subprocess
import sys
import time

import yaml

ROOT = "/home/DSJ44/paper2_yolov11_steel"
EXP = os.path.join(ROOT, "round21_v8data_20261009")
PY = "/home/DSJ44/miniconda3/envs/yolov11/bin/python"
SCRIPTS = os.path.join(EXP, "scripts")

GATES = os.path.join(EXP, "gates")
OPTIMIZER = os.path.join(EXP, "optimizer")
LOGS = os.path.join(EXP, "logs")
OUTCSV = os.path.join(EXP, "train_results.csv")

sys.path.insert(0, SCRIPTS)
from r21_train_one import JOBS, template_path  # noqa: E402

ALLOWED = {"model", "project", "name", "save_dir"}
NUMEQ = {"warmup_epochs"}          # 3 vs 3.0 之类,数值等价,记录但不判 FAIL

COLS = ["group", "condition", "seed", "run_name", "template_run", "epochs_run",
        "best_pt_sha256", "gate", "optimizer_line"]


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def template_for(run_name):
    return template_path(run_name)


def num_eq(a, b):
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return False


def count_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    return max(0, len(lines) - 1)


def gate_check(run_args_path, template_path_, name):
    """4a: 写 gates/<name>.diff,返回 (verdict, viol, numeq, allowed_hit)"""
    with open(run_args_path, "r", encoding="utf-8") as f:
        run_a = yaml.safe_load(f)
    with open(template_path_, "r", encoding="utf-8") as f:
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

    p = subprocess.run(["diff", template_path_, run_args_path],
                       capture_output=True, text=True)
    with open(os.path.join(GATES, name + ".diff"), "w", encoding="utf-8") as f:
        f.write("# diff %s %s\n" % (template_path_, run_args_path))
        f.write("# exit_code=%d\n" % p.returncode)
        f.write(p.stdout if p.stdout else "(no differences)\n")

    verdict = "PASS" if not viol else "FAIL"
    with open(os.path.join(GATES, name + ".txt"), "w", encoding="utf-8") as f:
        f.write("GATE %s\n" % verdict)
        f.write("run_args=%s\n" % run_args_path)
        f.write("template=%s\n" % template_path_)
        f.write("允许差异字段(model/project/name/save_dir)-> %d 个:\n" % len(allowed_hit))
        for k, b, a in allowed_hit:
            f.write("    ALLOWED %-9s 模板=%r 本次=%r\n" % (k, b, a))
        f.write("数值等价例外(不判 FAIL)-> %s\n" % (numeq if numeq else "无"))
        f.write("第五个字段差异(违规)-> %d 个\n" % len(viol))
        for k, b, a in viol:
            f.write("    VIOLATION %-9s 模板=%r 本次=%r\n" % (k, b, a))
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


def main():
    for d in (GATES, OPTIMIZER, LOGS):
        os.makedirs(d, exist_ok=True)

    print("[R21] driver start=%s" % time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    print("[R21] python=%s" % PY, flush=True)
    print("[R21] EXP=%s" % EXP, flush=True)
    print("[R21] 顺序: A组(0,1,42) -> B组XML(0,1,42) -> B组DIR(0,1,42)", flush=True)
    print("[R21] 共 %d 次" % len(JOBS), flush=True)

    t_all = time.time()
    for i, (group, cond, seed, tname, name) in enumerate(JOBS, 1):
        tpath = template_for(name)
        run_dir = os.path.join(EXP, name)
        run_args = os.path.join(run_dir, "args.yaml")
        logf = os.path.join(LOGS, name + ".log")

        hdr = ("\n" + "=" * 78 +
               "\n[R21] job %d/%d  %s (group=%s condition=%s seed=%d template=%s)  start=%s\n"
               % (i, len(JOBS), name, group, cond, seed, tname,
                  time.strftime("%Y-%m-%d %H:%M:%S")) +
               "=" * 78)
        print(hdr, flush=True)

        # ---------------- training ----------------
        cmd = [PY, os.path.join(SCRIPTS, "r21_train_one.py"), name,
               "--project", EXP, "--log", logf]
        with open(logf, "w", encoding="utf-8") as lf:
            lf.write("[R21] cmd=%s\n" % " ".join(cmd))
            lf.flush()
            proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
        rc = proc.returncode
        print("[R21] job %d/%d %s train exit_code=%d end=%s"
              % (i, len(JOBS), name, rc, time.strftime("%Y-%m-%d %H:%M:%S")), flush=True)

        with open(logf, "r", encoding="utf-8", errors="replace") as lf:
            logtxt = lf.read()

        # ---------------- 4b optimizer ----------------
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
        print("[R21] 4b optimizer_line=%r" % optimizer_line, flush=True)
        opt_ok = ("SGD(lr=0.01, momentum=0.937)" in optimizer_line)

        # ---------------- 4a GATE ----------------
        if not os.path.exists(run_args):
            with open(os.path.join(GATES, name + ".txt"), "w", encoding="utf-8") as f:
                f.write("GATE FAIL\nargs.yaml 不存在: %s\ntrain exit_code=%d\n"
                        % (run_args, rc))
            print("[R21] 4a GATE FAIL: args.yaml 不存在 -> 停止队列,不启动下一次", flush=True)
            append_csv({"group": group, "condition": cond, "seed": seed, "run_name": name,
                        "template_run": tname, "epochs_run": 0, "best_pt_sha256": "",
                        "gate": "FAIL", "optimizer_line": optimizer_line})
            return 2

        verdict, viol, numeq, allowed_hit = gate_check(run_args, tpath, name)
        print("[R21] 4a GATE=%s allowed=%s numeq=%s violations=%s"
              % (verdict, [x[0] for x in allowed_hit], numeq, viol), flush=True)

        # ---------------- 4c epochs ----------------
        results_csv = os.path.join(run_dir, "results.csv")
        epochs_run = count_rows(results_csv) if os.path.exists(results_csv) else 0
        print("[R21] 4c epochs_run=%d" % epochs_run, flush=True)

        # ---------------- best.pt sha256 ----------------
        best = os.path.join(run_dir, "weights", "best.pt")
        best_sha = sha256_file(best) if os.path.exists(best) else ""
        print("[R21] best.pt=%s exists=%s sha256=%s"
              % (best, os.path.exists(best), best_sha), flush=True)

        # ---------------- 4d append row ----------------
        gate_val = "TRAIN_ERROR" if rc != 0 else verdict
        append_csv({
            "group": group, "condition": cond, "seed": seed, "run_name": name,
            "template_run": tname, "epochs_run": epochs_run,
            "best_pt_sha256": best_sha, "gate": gate_val,
            "optimizer_line": optimizer_line,
        })
        print("[R21] 4d appended row -> %s" % OUTCSV, flush=True)

        # ---------------- 停止条件 ----------------
        if rc != 0:
            print("[R21] 训练报错退出 exit_code=%d(非闸门原因)-> 停止队列,不启动下一次" % rc,
                  flush=True)
            return 5
        if verdict == "FAIL":
            print("[R21] GATE FAIL -> 立即停止整个队列,不启动下一次训练", flush=True)
            return 3
        if not opt_ok:
            print("[R21] 优化器解析行不含 SGD(lr=0.01, momentum=0.937) -> 停止队列", flush=True)
            return 4

    total = time.time() - t_all
    print("\n[R21] all %d jobs finished, total_elapsed_sec=%.1f (%s)"
          % (len(JOBS), total, time.strftime("%Y-%m-%d %H:%M:%S")), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
