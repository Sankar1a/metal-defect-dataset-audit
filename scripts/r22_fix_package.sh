#!/usr/bin/env bash
# Round 22 补救交付打包:round22_rtdetr.tar.gz
# 与 scripts/r22_package.sh 相同,仅按任务书 10 第3b步**追加**两项:
#   - fix_20261012/ 整个目录
#   - aborted_20261012/ 中的 .log 文件与 args.yaml(不含权重)
# 其余清单、排除项(不打包 logs/、权重、图片、runs_eval/)与 r22_package.sh 完全一致。
set -u
ROOT=/home/DSJ44/paper2_yolov11_steel
D=$ROOT/round22_rtdetr_20261010
TAR=$ROOT/round22_rtdetr.tar.gz
LIST=$(mktemp)

cd "$D" || exit 1

# --- 顶层交付文件(与 r22_package.sh 相同) ---
for f in START.txt END.txt launch_check.txt weights.txt templates.txt \
         probe_timing.txt decision.txt dryrun_check.txt train_results.csv \
         eval_kwargs_source.txt eval_selfcheck.txt \
         dose_val_yamls.txt gc10_val_yamls.txt neu_val_yamls.txt \
         neu_leak_rtdetr.csv dose_results_rtdetr.csv \
         gc10_label_results_rtdetr.csv gc10_per_class_rtdetr.csv \
         gc10_per_class_check.txt \
         precheck_settings_before.json precheck_nvidia_smi.txt precheck_gpu_apps.txt \
         precheck_disk.txt precheck_versions.txt \
         impact_project.txt impact_project_followlinks.txt \
         impact_ultralytics.txt impact_settings_diff.txt \
         deviation_note.txt ; do
  [ -f "$f" ] && echo "$f" >> "$LIST"
done

# --- 任何 STOPPED_*.txt ---
for f in STOPPED_*.txt; do [ -f "$f" ] && echo "$f" >> "$LIST"; done

# --- 目录:gates/ optimizer/ determinism/ log_tails/ ---
for dir in gates optimizer determinism log_tails; do
  [ -d "$dir" ] && find "$dir" -type f >> "$LIST"
done

# --- 每个 run 的 args.yaml 与 results.csv ---
if [ -f run_names.txt ]; then
  while IFS= read -r r; do
    [ -n "$r" ] || continue
    for f in args.yaml results.csv; do
      [ -f "$r/$f" ] && echo "$r/$f" >> "$LIST"
    done
  done < run_names.txt
else
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    t=$(basename "$(dirname "$line")")
    r=$(echo "$t" | sed 's/yolo11n/rtdetr-l/')
    for f in args.yaml results.csv; do
      [ -f "$r/$f" ] && echo "$r/$f" >> "$LIST"
    done
  done < templates_paths.txt
fi

# --- 补救追加 1:fix_20261012/ 整个目录 ---
[ -d fix_20261012 ] && find fix_20261012 -type f >> "$LIST"
# --- 补救追加 2:aborted_20261012/ 中的 .log 与 args.yaml(不含权重) ---
[ -d aborted_20261012 ] && find aborted_20261012 -type f \( -name '*.log' -o -name 'args.yaml' \) >> "$LIST"

sort -u "$LIST" -o "$LIST"
echo "[pkg] 打包条目数=$(wc -l < "$LIST")"
echo "[pkg] 条目清单:"
sed 's/^/    /' "$LIST"

tar -czf "$TAR" -C "$D" -T "$LIST"
RC=$?
rm -f "$LIST"

if [ "$RC" -eq 0 ]; then
  echo "[pkg] OK $TAR"
  ls -l "$TAR"
  stat -c 'size_bytes=%s' "$TAR"
  sha256sum "$TAR"
else
  echo "[pkg] FAILED rc=$RC"
fi
exit "$RC"
