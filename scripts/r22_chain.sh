#!/usr/bin/env bash
# Round 22 链式脚本 —— 下载(仅当需要)-> 测速 -> 空跑 -> 训练 -> 评估 -> 影响范围 -> 打包
#
# 启动方式(完全脱离会话,任务书"总体要求"):
#   cd /home/DSJ44/paper2_yolov11_steel/round22_rtdetr_20261010
#   setsid nohup bash /home/DSJ44/paper2_yolov11_steel/round22_rtdetr_20261010/scripts/r22_chain.sh \
#       > /home/DSJ44/paper2_yolov11_steel/round22_rtdetr_20261010/chain.out 2>&1 < /dev/null &
#
# 链中任何一步失败 -> 写 STOPPED_<步骤名>.txt 说明原因,链终止,不做后续步骤。
# 严禁 pkill -f / killall;不得对目录使用带通配符的 chmod。
set -u
ROOT=/home/DSJ44/paper2_yolov11_steel
D=$ROOT/round22_rtdetr_20261010
PY=/home/DSJ44/miniconda3/envs/yolov11/bin/python
S=$D/scripts
LOGS=$D/logs

mkdir -p "$LOGS" "$D/gates" "$D/optimizer" "$D/determinism" "$D/log_tails" \
         "$D/runs_eval" "$D/probe/logs"
cd "$D" || exit 1

stamp() { date '+%Y-%m-%d %H:%M:%S'; }

# 写 STOPPED_<step>.txt(追加,避免覆盖该步脚本自身写的更详细内容)并终止链
stop() {
  step="$1"; reason="$2"
  f="$D/STOPPED_${step}.txt"
  {
    echo "STOPPED_${step}.txt — 第二十二轮 链在第 ${step} 步终止"
    echo "时间: $(stamp)"
    echo "原因: ${reason}"
    echo "后续步骤全部不执行。"
  } >> "$f"
  echo "[chain] STOPPED at ${step}: ${reason}"
  exit 1
}

echo "################ r22_chain start $(stamp) ################"
echo "[chain] ROOT=$ROOT"
echo "[chain] D=$D"

# ---------------- 第1步 权重 ----------------
"$PY" -u "$S/r22_weights.py" > "$LOGS/_weights.log" 2>&1
RC=$?
echo "[chain] step1 weights rc=$RC $(stamp)"
[ "$RC" -eq 0 ] || stop weights "r22_weights.py rc=$RC;详见 weights.txt 与 logs/_weights.log"

# ---------------- 第2步 模板 ----------------
"$PY" -u "$S/r22_templates.py" > "$LOGS/_templates.log" 2>&1
RC=$?
echo "[chain] step2 templates rc=$RC $(stamp)"
[ "$RC" -eq 0 ] || stop templates "r22_templates.py rc=$RC;详见 templates.txt 与 logs/_templates.log"

# ---------------- 第3步 测速 ----------------
"$PY" -u "$S/r22_probe.py" > "$LOGS/_probe.log" 2>&1
RC=$?
echo "[chain] step3 probe rc=$RC $(stamp)"
[ "$RC" -eq 0 ] || stop probe "r22_probe.py rc=$RC;详见 probe_timing.txt / STOPPED_probe.txt / logs/_probe.log"

# ---------------- 第4步 空跑 ----------------
"$PY" -u "$S/r22_dryrun.py" > "$LOGS/_dryrun.log" 2>&1
RC=$?
echo "[chain] step4 dryrun rc=$RC $(stamp)"
[ "$RC" -eq 0 ] || stop dryrun "r22_dryrun.py rc=$RC;详见 dryrun_check.txt 与 logs/_dryrun.log"

# ---------------- 第5步 训练(12 次串行) ----------------
"$PY" -u "$S/r22_run_all.py" > "$LOGS/_driver.log" 2>&1
RC=$?
echo "[chain] step5 train rc=$RC $(stamp)"
[ "$RC" -eq 0 ] || stop train "r22_run_all.py rc=$RC;详见 train_results.csv / gates/ / optimizer/ / logs/_driver.log"

# ---------------- 第6步 评估 ----------------
"$PY" -u "$S/r22_eval.py" > "$LOGS/_eval.log" 2>&1
RC=$?
echo "[chain] step6 eval rc=$RC $(stamp)"
[ "$RC" -eq 0 ] || stop eval "r22_eval.py rc=$RC;详见 eval_selfcheck.txt / logs/_eval.log"

# ---------------- log_tails ----------------
"$PY" -u "$S/r22_logtails.py" > "$LOGS/_logtails.log" 2>&1
echo "[chain] logtails rc=$? $(stamp)"

# ---------------- 第7步 影响范围 ----------------
{
  echo "=== 第7步 $(stamp) ==="
  cd "$ROOT" || exit 1
  date '+%Y-%m-%d %H:%M:%S' > "$D/END.txt"
  START=$(cat "$D/START.txt")
  echo "START=$START"
  # 任务书第7步的 $D 是**相对**目录名。若误用绝对路径,
  # `-not -path "./$D/*"` 永远匹配不上,本轮目录自身会被算进影响面(第二十一轮出过此错)。
  DB=$(basename "$D")
  find . -newermt "$START" -type f -not -path "./$DB/*" -not -path './.git/*' > "$D/impact_project.txt"
  find -L . -newermt "$START" -type f -not -path "./$DB/*" -not -path './.git/*' > "$D/impact_project_followlinks.txt"
  find /home/DSJ44/.config/Ultralytics -newermt "$START" -type f > "$D/impact_ultralytics.txt"
  find /home/DSJ44/.config/Ultralytics -type f -printf '%T+ %p\n' | sort >> "$D/impact_ultralytics.txt"
  diff "$D/precheck_settings_before.json" /home/DSJ44/.config/Ultralytics/settings.json > "$D/impact_settings_diff.txt"
  echo "step7 rc=$?"
} >> "$LOGS/_step7.log" 2>&1
echo "[chain] step7 impact rc=$? $(stamp)"

# ---------------- 打包 ----------------
bash "$S/r22_package.sh" >> "$LOGS/_package.log" 2>&1
RC=$?
echo "[chain] package rc=$RC $(stamp)"
[ "$RC" -eq 0 ] || stop package "r22_package.sh rc=$RC;详见 logs/_package.log"

echo "################ r22_chain end $(stamp) ################"
