#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Round 22 补救 · 第3c步 空跑(只针对第 12 个 run)。

与 r22_dryrun.py 的核对规则完全一致(同一 ALLOWED / NUMEQ / FIXED_MODS / EXCLUDE,
同一"真实 train() 路径 + 构造训练器后立即中止"做法),差异仅在于:
  - 只跑 gc10_rtdetr-l_dir_s42 一个 run
  - 空跑目录名前缀为 dryrun_fix_
  - 结果写入 fix_20261012/dryrun_check.txt
  - 空跑目录不删除,完成后移入 fix_20261012/

用法: r22_fix_dryrun.py
"""
import os
import shutil
import sys
import time

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r22_common import (ALLOWED, EXCLUDE, EXP, FIXED_MODS, JOBS, NUMEQ,  # noqa: E402
                        RTDETR_WEIGHT, num_eq, read_decision, template_path)

RUN_NAME = "gc10_rtdetr-l_dir_s42"
FIXDIR = os.path.join(EXP, "fix_20261012")
OUT = os.path.join(FIXDIR, "dryrun_check.txt")


def main():
    from ultralytics import RTDETR
    from ultralytics.engine.trainer import BaseTrainer

    dec = read_decision()
    E, B = int(dec["epochs"]), int(dec["batch"])

    job = [j for j in JOBS if j[4] == RUN_NAME]
    if len(job) != 1:
        sys.stderr.write("错误: 找不到唯一 run %s\n" % RUN_NAME)
        return 90
    ds, cond, seed, tname, run_name = job[0]
    tpath = template_path(run_name)
    dr_name = "dryrun_fix_" + run_name
    ddir = os.path.join(EXP, dr_name)

    lines = []
    A = lines.append
    A("dryrun_check.txt — 第二十二轮补救(任务书 10)第3c步 空跑核对")
    A("生成时间: %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    A("做法: 走真实 RTDETR(...).train(...) 路径,构造训练器(args.yaml 落盘)后立即中止,不训练。")
    A("核对规则与 r22_dryrun.py 完全一致(同一 ALLOWED / NUMEQ / FIXED_MODS / EXCLUDE)。")
    A("空跑目录: %s(前缀 dryrun_fix_,完成后移入 fix_20261012/)" % ddir)
    A("epochs=%d batch=%d(取自 decision.txt)" % (E, B))
    A("允许差异字段(任务书 9 第2步表格): %s" % sorted(ALLOWED))
    A("数值等价例外(不判 FAIL): %s" % sorted(NUMEQ))
    A("weight: %s (exists=%s)" % (RTDETR_WEIGHT, os.path.exists(RTDETR_WEIGHT)))
    A("=" * 96)
    A("")
    A("dataset/condition/seed : %s / %s / %s" % (ds, cond, seed))
    A("run_name               : %s" % run_name)
    A("template_run           : %s" % tname)
    A("dryrun name            : %s" % dr_name)
    A("template               : %s" % tpath)

    if os.path.exists(ddir):
        shutil.rmtree(ddir)

    with open(tpath, "r", encoding="utf-8") as f:
        tmpl_orig = yaml.safe_load(f)
    t = dict(tmpl_orig)
    t["model"] = RTDETR_WEIGHT
    t["project"] = EXP
    t["name"] = dr_name
    t.update(FIXED_MODS)
    t["epochs"] = E
    t["batch"] = B
    kwargs = {k: v for k, v in t.items() if k not in EXCLUDE}

    class Stop(Exception):
        pass

    orig_train = BaseTrainer.train
    BaseTrainer.train = lambda self, *a, **k: (_ for _ in ()).throw(Stop())
    err = None
    try:
        RTDETR(RTDETR_WEIGHT).train(**kwargs)
    except Stop:
        pass
    except BaseException as e:                                      # noqa: BLE001
        err = repr(e)
    finally:
        BaseTrainer.train = orig_train

    outp = os.path.join(ddir, "args.yaml")
    if err:
        A("异常                   : %s" % err)
    if not os.path.exists(outp):
        A("结果                   : FAIL — args.yaml 未生成")
        A("")
        A("=" * 96)
        A("结论: FAIL,不得进入重训")
        A("=" * 96)
        os.makedirs(FIXDIR, exist_ok=True)
        with open(OUT, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print("\n".join(lines), flush=True)
        return 1

    with open(outp, "r", encoding="utf-8") as f:
        got = yaml.safe_load(f)

    allowed_hit, numeq, viol, diff_keys = [], [], [], []
    for kk in sorted(set(got) | set(tmpl_orig)):
        a, b = got.get(kk, "<MISSING>"), tmpl_orig.get(kk, "<MISSING>")
        if a == b:
            continue
        diff_keys.append(kk)
        if kk in ALLOWED:
            allowed_hit.append((kk, b, a))
        elif kk in NUMEQ and num_eq(a, b):
            numeq.append((kk, b, a))
        else:
            viol.append((kk, b, a))

    verdict = "PASS" if not viol else "FAIL"
    A("差异键                 : %s" % diff_keys)
    A("  允许字段             : %s" % sorted(x[0] for x in allowed_hit))
    for k, b, a in allowed_hit:
        A("      %-13s 模板=%r" % (k, b))
        A("      %-13s 空跑=%r" % ("", a))
    A("  数值等价             : %s" % (numeq if numeq else "无"))
    A("  违规差异             : %s" % (viol if viol else "无(0 个)"))
    A("  data 是否一致        : %s"
      % ("一致" if got.get("data") == tmpl_orig.get("data") else "不一致"))
    A("  seed 是否一致        : %s"
      % ("一致" if got.get("seed") == tmpl_orig.get("seed") else "不一致"))
    A("结果                   : %s" % verdict)
    A("")
    A("=" * 96)
    A("结论: %s" % ("PASS,可以进入重训" if verdict == "PASS" else "FAIL,不得进入重训"))
    A("=" * 96)

    os.makedirs(FIXDIR, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)

    # 空跑目录不删,移入 fix_20261012/
    dest = os.path.join(FIXDIR, dr_name)
    if os.path.exists(dest):
        shutil.rmtree(dest)
    shutil.move(ddir, dest)
    print("\n[dryrun_fix] 已移入 %s" % dest, flush=True)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
