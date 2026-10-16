#!/bin/bash
# Unified-protocol 18-run benchmark driver (strictly serial, single GPU).
# env: /home/DSJ44/miniconda3/envs/yolov11/bin/python (py3.10/torch2.3.1+cu121/ul8.3.40)
# protocol: runs/baseline/v11n_neudet_seed0/args.yaml (aug v2) verbatim, override model/data/seed/project/name
# On any run failure: log to train_log.txt, skip, continue. No retry.
set -u

PY=/home/DSJ44/miniconda3/envs/yolov11/bin/python
ROOT=/home/DSJ44/paper2_yolov11_steel
OUT=$ROOT/runs_unified_20260803
LOG=$OUT/train_log.txt

# 18 runs: DS MODEL SEED NAME
RUNS=(
  "NEU   yolo11n 0  neu_yolo11n_s0"
  "NEU   yolo11n 1  neu_yolo11n_s1"
  "NEU   yolo11n 42 neu_yolo11n_s42"
  "NEU   yolo11s 0  neu_yolo11s_s0"
  "NEU   yolo11s 1  neu_yolo11s_s1"
  "NEU   yolo11s 42 neu_yolo11s_s42"
  "GC10  yolo11n 0  gc10_yolo11n_s0"
  "GC10  yolo11n 1  gc10_yolo11n_s1"
  "GC10  yolo11n 42 gc10_yolo11n_s42"
  "GC10  yolo11s 0  gc10_yolo11s_s0"
  "GC10  yolo11s 1  gc10_yolo11s_s1"
  "GC10  yolo11s 42 gc10_yolo11s_s42"
  "APSPC yolo11n 0  apspc_yolo11n_s0"
  "APSPC yolo11n 1  apspc_yolo11n_s1"
  "APSPC yolo11n 42 apspc_yolo11n_s42"
  "APSPC yolo11s 0  apspc_yolo11s_s0"
  "APSPC yolo11s 1  apspc_yolo11s_s1"
  "APSPC yolo11s 42 apspc_yolo11s_s42"
)

ts() { date '+%Y-%m-%dT%H:%M:%S%z'; }

{
  echo "# Unified 18-run training log"
  echo "# started: $(ts)"
  echo "# env: yolov11 py3.10.20 torch2.3.1+cu121 ultralytics8.3.40 PYTHONNOUSERSITE=1"
  echo "# protocol: runs/baseline/v11n_neudet_seed0/args.yaml (aug v2): epochs=200 imgsz=640 batch=16 patience=50 iou=0.70"
  echo "# output: runs_unified_20260803/"
} > "$LOG"

for spec in "${RUNS[@]}"; do
  read -r DS MODEL SEED NAME <<< "$spec"
  echo "$(ts) START $NAME DS=$DS MODEL=$MODEL SEED=$SEED" >> "$LOG"
  PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 "$PY" "$OUT/train_one.py" "$DS" "$MODEL" "$SEED" "$NAME" > "$OUT/logs/$NAME.out" 2>&1
  EC=$?
  if [ $EC -eq 0 ]; then
    echo "$(ts) END $NAME exit=$EC COMPLETE_AS_CONFIGURED" >> "$LOG"
  else
    LASTERR=$(grep -vE '^\s*$' "$OUT/logs/$NAME.out" | tail -3 | tr '\n' ' ' | cut -c1-400)
    echo "$(ts) FAILED $NAME exit=$EC lasterr=$LASTERR" >> "$LOG"
  fi
done
echo "$(ts) ALL_DONE exit_total_ok" >> "$LOG"
