#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Round 22 补救 · 只跑第 12 个 run(gc10_rtdetr-l_dir_s42)。

任务书 10 第 3a 步要求:"能用 r22_run_all.py 只跑这一个 run 就用它"。
本脚本不重写任何训练/闸门/优化器/确定性/CSV 逻辑,做法是把
r22_run_all.py 的模块级 JOBS 收窄为这一个 run,然后**原样调用它的 main()**,
从而走完全相同的代码路径:
    - 同样的训练命令行(调用 r22_train_one.py,参数一致)
    - 同样的闸门 diff(gates/<name>.diff / .txt)
    - 同样的优化器解析行(optimizer/<name>.txt,须含 OPTIMIZER_EXPECT)
    - 同样的确定性告警(determinism/<name>.txt)
    - 同样追加 train_results.csv(append_csv)
    - 同样的停止条件(STOPPED_train/gate/optimizer.txt)

退出码语义与 r22_run_all.main() 相同:0 正常;2 args.yaml 缺失;3 闸门 FAIL;
4 优化器不含期望串;5 训练报错退出。

用法: r22_fix_one.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r22_run_all as R                                            # noqa: E402

RUN_NAME = "gc10_rtdetr-l_dir_s42"

_kept = [j for j in R.JOBS if j[4] == RUN_NAME]
if len(_kept) != 1:
    sys.stderr.write("错误: 在 JOBS 中找不到唯一 run %s(匹配 %d 个)\n" % (RUN_NAME, len(_kept)))
    sys.exit(90)

print("[R22-FIX] 复用 r22_run_all.py 的 main(),JOBS 收窄为 1 个:%s" % RUN_NAME, flush=True)
R.JOBS = _kept
sys.exit(R.main())
