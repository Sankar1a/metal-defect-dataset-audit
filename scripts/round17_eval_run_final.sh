#!/usr/bin/env bash
# Round 17 · final stage: row counts, impact-scope verification, packaging.
# Train nothing, modify nothing outside round17_eval_20261005/ (plus the requested tarball).

set -u
ROOT=/home/DSJ44/paper2_yolov11_steel
cd "$ROOT"
OUT=$ROOT/round17_eval_20261005

echo "########## row counts ##########"
cd "$OUT"
wc -l A_neu_overall.csv \
      A_neu_per_class.csv \
      B_dose_overall.csv \
      B_dose_per_class.csv \
      B_valset_class_counts.csv \
      C_gc10_overall.csv \
      C_gc10_per_class.csv > final_rowcounts.txt
cat final_rowcounts.txt

echo "########## DONE.txt ##########"
cat DONE.txt
echo "--- eval process exit code ---"
cat eval_exit_code.txt 2>/dev/null || echo "(eval_exit_code.txt missing)"

echo "########## last step · impact scope ##########"
cd "$ROOT"
START=$(cat round17_eval_20261005/START.txt)
echo "START=$START"

find . -newermt "$START" -type f -not -path './round17_eval_20261005/*' -not -path './.git/*' \
  > round17_eval_20261005/impact_project.txt
echo "impact_project.txt lines: $(wc -l < round17_eval_20261005/impact_project.txt)"

find /home/DSJ44/.config/Ultralytics -newermt "$START" -type f \
  > round17_eval_20261005/impact_ultralytics.txt
find /home/DSJ44/.config/Ultralytics -type f -printf '%T+ %p\n' | sort \
  >> round17_eval_20261005/impact_ultralytics.txt

diff round17_eval_20261005/precheck_settings_before.json ~/.config/Ultralytics/settings.json \
  > round17_eval_20261005/impact_settings_diff.txt
echo "settings diff exit=$?"
cat round17_eval_20261005/impact_settings_diff.txt

echo "########## package ##########"
cd "$ROOT"
rm -f round17_eval.tar.gz
tar --exclude='round17_eval_20261005/val_runs' -czf round17_eval.tar.gz round17_eval_20261005
echo "tar exit=$?"
ls -l round17_eval.tar.gz
echo "--- archive size (bytes) ---"
stat -c '%s' round17_eval.tar.gz
echo "--- top-level members ---"
tar -tzf round17_eval.tar.gz | awk -F/ '{print $1"/"$2}' | sort -u
echo "--- any val_runs leaked into archive? (expect none) ---"
tar -tzf round17_eval.tar.gz | grep -c 'val_runs' || echo "0"
echo "--- member count ---"
tar -tzf round17_eval.tar.gz | wc -l

echo "########## FINAL DONE ##########"
