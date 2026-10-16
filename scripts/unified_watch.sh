#!/bin/bash
# Watcher for the unified 18-run driver. Polls train_log.txt every 300s.
# Exits when ALL_DONE appears in train_log.txt, or driver process disappears.
LOG=/home/DSJ44/paper2_yolov11_steel/runs_unified_20260803/train_log.txt
PIDF=/tmp/unified_driver.pid
OUT=/home/DSJ44/paper2_yolov11_steel/runs_unified_20260803
echo "watcher start $(date '+%Y-%m-%dT%H:%M:%S%z')"
while :; do
  if grep -q 'ALL_DONE' "$LOG" 2>/dev/null; then
    echo "ALL_DONE detected $(date '+%Y-%m-%dT%H:%M:%S%z')"
    break
  fi
  if [ -f "$PIDF" ]; then
    DPID=$(cat "$PIDF")
    if ! ps -p "$DPID" >/dev/null 2>&1; then
      echo "DRIVER_GONE_WITHOUT_ALLDONE $(date '+%Y-%m-%dT%H:%M:%S%z')"
      break
    fi
  fi
  # report current state
  LAST=$(grep -E ' START | END | FAILED ' "$LOG" 2>/dev/null | tail -1)
  GPU=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | head -1)
  CUR_EPOCH=""
  CUR=$(echo "$LAST" | awk '{print $2}')
  if [ -n "$CUR" ] && [ -f "$OUT/$CUR/results.csv" ]; then
    ROWS=$(wc -l < "$OUT/$CUR/results.csv")
    CUR_EPOCH="rows=$ROWS"
  fi
  echo "check $(date '+%Y-%m-%dT%H:%M:%S%z') last=$LAST gpu=[$GPU] $CUR_EPOCH"
  sleep 300
done
echo "watcher end $(date '+%Y-%m-%dT%H:%M:%S%z')"
