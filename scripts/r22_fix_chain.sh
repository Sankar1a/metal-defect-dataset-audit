#!/usr/bin/env bash
# Round 22 补救链(任务书 10)
#   3c 空跑(单 run)-> 3a 重训第 12 次(带看门狗,失败原样重训一次)
#   -> 校验闸门/优化器/CSV -> 3b 评估(带看门狗)-> log_tails -> 第7步影响范围 -> 打包
#
# 启动(任务书 10 第4步):
#   cd /home/DSJ44/paper2_yolov11_steel/round22_rtdetr_20261010
#   setsid nohup bash scripts/r22_fix_chain.sh > fix_20261012/fix_chain.out 2>&1 < /dev/null &
#
# 严禁 pkill -f / killall;只按具体 PID / PGID 结束进程。不得对目录使用带通配符的 chmod。
set -u
ROOT=/home/DSJ44/paper2_yolov11_steel
D=$ROOT/round22_rtdetr_20261010
PY=/home/DSJ44/miniconda3/envs/yolov11/bin/python
S=$D/scripts
LOGS=$D/logs
FIX=$D/fix_20261012
RUN=gc10_rtdetr-l_dir_s42
POLL=60            # 看门狗轮询间隔(秒)
STALL=1200         # 连续无更新阈值(秒)= 20 分钟

mkdir -p "$FIX" "$LOGS" "$D/aborted_20261012"
cd "$D" || exit 1

stamp() { date '+%Y-%m-%d %H:%M:%S'; }

stop() {
  step="$1"; reason="$2"
  f="$D/STOPPED_${step}.txt"
  {
    echo "STOPPED_${step}.txt — 第二十二轮补救 链终止"
    echo "时间: $(stamp)"
    echo "原因: ${reason}"
    echo "后续步骤全部不执行。"
  } >> "$f"
  echo "[fix] STOPPED at ${step}: ${reason}"
  exit 1
}

# ---------------- 看门狗:PID/PGID 与日志路径 ----------------
# $1 = 日志路径, $2 = 进程组 PGID(= setsid 首进程 PID), $3 = 尝试序号
watchdog_loop() {
  local log="$1" pgid="$2" n="$3" last_m last_c now m
  last_c=$(date +%s)
  last_m=$(stat -c %Y "$log" 2>/dev/null || echo 0)
  while kill -0 "$pgid" 2>/dev/null; do
    sleep "$POLL"
    kill -0 "$pgid" 2>/dev/null || break
    now=$(date +%s)
    m=$(stat -c %Y "$log" 2>/dev/null || echo 0)
    if [ "$m" != "$last_m" ]; then
      last_m="$m"; last_c="$now"; continue
    fi
    if [ $((now - last_c)) -ge "$STALL" ]; then
      {
        echo "[watchdog] 触发 $(stamp)"
        echo "    attempt=$n"
        echo "    日志=$log"
        echo "    日志最后修改=$(date -d @"$last_m" '+%F %T' 2>/dev/null || echo "$last_m")"
        echo "    连续无更新秒数=$((now - last_c))(阈值 $STALL,轮询 $POLL)"
        echo "    目标进程组 PGID=$pgid"
        echo "    命令: kill -TERM -$pgid"
      } >> "$FIX/watchdog.txt"
      kill -TERM -"$pgid" 2>/dev/null
      echo "[watchdog] $(stamp) kill -TERM -$pgid 已发送" >> "$FIX/watchdog.txt"
      sleep 30
      if kill -0 -"$pgid" 2>/dev/null; then
        kill -KILL -"$pgid" 2>/dev/null
        echo "[watchdog] $(stamp) 30 秒后进程组仍存活 -> kill -KILL -$pgid 已发送" >> "$FIX/watchdog.txt"
      else
        echo "[watchdog] $(stamp) 30 秒后进程组已不存在,无需 SIGKILL" >> "$FIX/watchdog.txt"
      fi
      touch "$FIX/watchdog_tripped.txt"
      return 0
    fi
  done
  return 0
}

# ---------------- 单次重训(训练 + 看门狗) ----------------
# 返回训练驱动 r22_fix_one.py 的退出码;看门狗触发时额外写 watchdog_tripped.txt
attempt_train() {
  local n="$1" log="$LOGS/$RUN.log" pid wpid rc
  rm -f "$FIX/watchdog_tripped.txt"
  echo "[fix] attempt=$n start=$(stamp)" >> "$FIX/train_attempts.txt"
  setsid "$PY" -u "$S/r22_fix_one.py" > "$log" 2>&1 &
  pid=$!
  echo "$pid" > "$FIX/train_pgid.txt"
  echo "[fix] attempt=$n pgid=$pid (已写入 fix_20261012/train_pgid.txt)" >> "$FIX/train_attempts.txt"
  watchdog_loop "$log" "$pid" "$n" &
  wpid=$!
  wait "$pid"
  rc=$?
  kill "$wpid" 2>/dev/null
  wait "$wpid" 2>/dev/null
  echo "[fix] attempt=$n rc=$rc end=$(stamp)" >> "$FIX/train_attempts.txt"
  return $rc
}

echo "################ r22_fix_chain start $(stamp) ################"
echo "[fix] RUN=$RUN  POLL=${POLL}s  STALL=${STALL}s"

# ---------------- 第3c步 空跑 ----------------
"$PY" -u "$S/r22_fix_dryrun.py" > "$LOGS/_fix_dryrun.log" 2>&1
RC=$?
echo "[fix] step3c dryrun rc=$RC $(stamp)"
[ "$RC" -eq 0 ] || stop fix_dryrun "r22_fix_dryrun.py rc=$RC;详见 fix_20261012/dryrun_check.txt"

# ---------------- 第3a步 重训(最多两次) ----------------
attempt_train 1
RC1=$?
echo "[fix] attempt1 rc=$RC1 $(stamp)"
if [ -f "$FIX/watchdog_tripped.txt" ]; then
  {
    echo "[fix] attempt1 看门狗触发 -> 产出移入 aborted_20261012/retry1/ 后原样重训一次"
    echo "      $(stamp)"
  } >> "$FIX/watchdog.txt"
  mkdir -p "$D/aborted_20261012/retry1"
  [ -e "$D/$RUN" ] && mv "$D/$RUN" "$D/aborted_20261012/retry1/"
  [ -e "$LOGS/$RUN.log" ] && mv "$LOGS/$RUN.log" "$D/aborted_20261012/retry1/"
  echo "[fix] attempt1 产出已移入 aborted_20261012/retry1/ $(stamp)"

  attempt_train 2
  RC2=$?
  echo "[fix] attempt2 rc=$RC2 $(stamp)"
  if [ -f "$FIX/watchdog_tripped.txt" ]; then
    mkdir -p "$D/aborted_20261012/retry2"
    [ -e "$D/$RUN" ] && mv "$D/$RUN" "$D/aborted_20261012/retry2/"
    [ -e "$LOGS/$RUN.log" ] && mv "$LOGS/$RUN.log" "$D/aborted_20261012/retry2/"
    stop fix_train "第二次重训仍被看门狗判定卡死;详见 fix_20261012/watchdog.txt"
  fi
  RC_TRAIN=$RC2
else
  RC_TRAIN=$RC1
fi

# ---------------- 校验:闸门 PASS / 优化器行 / CSV 12 行 ----------------
ROWS=$(wc -l < "$D/train_results.csv")
LAST=$(tail -1 "$D/train_results.csv")
OPT_OK="no"
[ -f "$D/optimizer/$RUN.txt" ] && grep -q 'AdamW(lr=0.0001, momentum=0.9)' "$D/optimizer/$RUN.txt" && OPT_OK="yes"
{
  echo "[fix] 训练后校验 $(stamp)"
  echo "  r22_fix_one.py rc=$RC_TRAIN"
  echo "  train_results.csv 行数=$ROWS(应为 13 = 表头 1 行 + 数据 12 行)"
  echo "  末行=$LAST"
  echo "  optimizer/$RUN.txt 含 AdamW(lr=0.0001, momentum=0.9) = $OPT_OK"
} >> "$FIX/train_attempts.txt"

if [ "$RC_TRAIN" -ne 0 ]; then
  stop fix_gate "训练驱动 rc=$RC_TRAIN(非 0);详见 fix_20261012/train_attempts.txt 与 gates/ optimizer/"
fi
if [ "$ROWS" -ne 13 ]; then
  stop fix_gate "train_results.csv 行数=$ROWS,不是 13(表头 + 12 行)"
fi
case "$LAST" in
  *,PASS,*) : ;;
  *) stop fix_gate "train_results.csv 末行 gate 不是 PASS: $LAST" ;;
esac
[ "$OPT_OK" = "yes" ] || stop fix_gate "optimizer/$RUN.txt 不含 AdamW(lr=0.0001, momentum=0.9)"
echo "[fix] 校验通过:闸门 PASS + 优化器行 OK + CSV 13 行 $(stamp)"

# ---------------- 第3b步 评估(带看门狗,不重试) ----------------
rm -f "$FIX/eval_tripped.txt"
setsid "$PY" -u "$S/r22_eval.py" > "$LOGS/_eval.log" 2>&1 &
EPID=$!
echo "$EPID" > "$FIX/eval_pgid.txt"
# 评估看门狗:日志 20 分钟无更新 -> 结束其进程组,写 STOPPED_fix_eval.txt(不重试)
(
  last_c=$(date +%s); last_m=$(stat -c %Y "$LOGS/_eval.log" 2>/dev/null || echo 0)
  while kill -0 "$EPID" 2>/dev/null; do
    sleep "$POLL"
    kill -0 "$EPID" 2>/dev/null || break
    now=$(date +%s); m=$(stat -c %Y "$LOGS/_eval.log" 2>/dev/null || echo 0)
    if [ "$m" != "$last_m" ]; then last_m="$m"; last_c="$now"; continue; fi
    if [ $((now - last_c)) -ge "$STALL" ]; then
      {
        echo "[watchdog-eval] 触发 $(stamp)"
        echo "    日志=$LOGS/_eval.log"
        echo "    连续无更新秒数=$((now - last_c))(阈值 $STALL)"
        echo "    目标进程组 PGID=$EPID"
      } >> "$FIX/watchdog.txt"
      kill -TERM -"$EPID" 2>/dev/null
      sleep 30
      kill -0 -"$EPID" 2>/dev/null && kill -KILL -"$EPID" 2>/dev/null
      touch "$FIX/eval_tripped.txt"
      return 0
    fi
  done
) &
EWPID=$!
wait "$EPID"
RC_EVAL=$?
kill "$EWPID" 2>/dev/null
wait "$EWPID" 2>/dev/null
echo "[fix] step3b eval rc=$RC_EVAL $(stamp)"
if [ -f "$FIX/eval_tripped.txt" ]; then
  stop fix_eval "评估输出连续 20 分钟无更新,已结束后按任务书不重试;详见 fix_20261012/watchdog.txt"
fi
[ "$RC_EVAL" -eq 0 ] || stop fix_eval "r22_eval.py rc=$RC_EVAL;详见 eval_selfcheck.txt 与 logs/_eval.log"

# ---------------- log_tails ----------------
"$PY" -u "$S/r22_logtails.py" > "$LOGS/_logtails.log" 2>&1
echo "[fix] logtails rc=$? $(stamp)"

# ---------------- 第7步 影响范围(沿用原 START.txt;D 用相对目录名) ----------------
{
  echo "=== 第7步 $(stamp) ==="
  cd "$ROOT" || exit 1
  date '+%Y-%m-%d %H:%M:%S' > "$D/END.txt"
  START=$(cat "$D/START.txt")
  echo "START=$START(原 START.txt,未改动)"
  DB=$(basename "$D")
  find . -newermt "$START" -type f -not -path "./$DB/*" -not -path './.git/*' > "$D/impact_project.txt"
  find -L . -newermt "$START" -type f -not -path "./$DB/*" -not -path './.git/*' > "$D/impact_project_followlinks.txt"
  find /home/DSJ44/.config/Ultralytics -newermt "$START" -type f > "$D/impact_ultralytics.txt"
  find /home/DSJ44/.config/Ultralytics -type f -printf '%T+ %p\n' | sort >> "$D/impact_ultralytics.txt"
  diff "$D/precheck_settings_before.json" /home/DSJ44/.config/Ultralytics/settings.json > "$D/impact_settings_diff.txt"
  echo "step7 rc=$?"
} >> "$LOGS/_step7.log" 2>&1
echo "[fix] step7 impact rc=$? $(stamp)"

# ---------------- 打包(原清单 + fix_20261012/ + aborted_20261012/ 的 .log 与 args.yaml) ----------------
bash "$S/r22_fix_package.sh" >> "$LOGS/_package.log" 2>&1
RC=$?
echo "[fix] package rc=$RC $(stamp)"
[ "$RC" -eq 0 ] || stop package "r22_fix_package.sh rc=$RC;详见 logs/_package.log"

echo "################ r22_fix_chain end $(stamp) ################"
