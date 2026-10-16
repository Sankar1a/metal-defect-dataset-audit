#!/usr/bin/env bash
# Round 17 · eval-only · preparation (A1-A3, B1, C1-C2)
# Writes ONLY into round17_eval_20261005/. Trains nothing, modifies nothing else.

set -u
ROOT=/home/DSJ44/paper2_yolov11_steel
cd "$ROOT"
OUT=$ROOT/round17_eval_20261005

VAL359_YAML=audits/leakage_sensitivity/neu_det_val359.yaml
VAL359_LIST=audits/leakage_sensitivity/val359.txt
VAL360_YAML=exp_leakage_20260817/configs/neu_val360_controlled.yaml
VAL360_LIST=exp_leakage_20260817/configs/val360.txt

NEU_RUNS="neu_yolo11n_s0 neu_yolo11n_s1 neu_yolo11n_s42 neu_yolo11s_s0 neu_yolo11s_s1 neu_yolo11s_s42"

echo "########## A1 ##########"
ls runs_unified_20260803/ > "$OUT/A_runs_listing.txt"
: > "$OUT/A_weights.txt"
for r in $NEU_RUNS; do
  for p in runs_unified_20260803/$r/weights/best.pt; do
    printf '%s  %s\n' "$(sha256sum "$p" | cut -d' ' -f1)" "$p" >> "$OUT/A_weights.txt"
  done
done
cat "$OUT/A_weights.txt"

echo "########## A2 ##########"
{
  echo "A2 · located evaluation sets (nothing created)"
  echo "val359 yaml : $VAL359_YAML"
  echo "val359 list : $VAL359_LIST"
  echo "val360 yaml : $VAL360_YAML"
  echo "val360 list : $VAL360_LIST"
  echo
  echo "--- exists / size / sha256 ---"
  for f in "$VAL359_YAML" "$VAL359_LIST" "$VAL360_YAML" "$VAL360_LIST"; do
    if [ -f "$f" ]; then
      printf '%-70s OK  %8s bytes  %s\n' "$f" "$(stat -c%s "$f")" "$(sha256sum "$f" | cut -d' ' -f1)"
    else
      printf '%-70s MISSING\n' "$f"
    fi
  done
} > "$OUT/A_val_sets.txt"
cat "$OUT/A_val_sets.txt"

echo "########## A3 ##########"
wc -l "$VAL359_LIST" "$VAL360_LIST" > "$OUT/A_check_counts.txt"
diff "$VAL359_LIST" "$VAL360_LIST" > "$OUT/A_check_list_diff.txt"; echo "list_diff_exit=$?"
diff "$VAL359_YAML" "$VAL360_YAML" > "$OUT/A_check_yaml_diff.txt"; echo "yaml_diff_exit=$?"
grep -H '^data:' runs_unified_20260803/neu_yolo11n_s0/args.yaml \
                runs_unified_20260803/neu_yolo11n_s1/args.yaml \
                runs_unified_20260803/neu_yolo11n_s42/args.yaml \
                runs_unified_20260803/neu_yolo11s_s0/args.yaml \
                runs_unified_20260803/neu_yolo11s_s1/args.yaml \
                runs_unified_20260803/neu_yolo11s_s42/args.yaml > "$OUT/A_check_run_data.txt"

# --- extra evidence step requested for the `path:` line difference ---
{
  echo "\$ ls -la /home/DSJ44/paper2_yolov11_steel/datasets/NEU-DET"
  ls -la /home/DSJ44/paper2_yolov11_steel/datasets/NEU-DET
  echo
  echo "\$ realpath /home/DSJ44/paper2_yolov11_steel/datasets/NEU-DET"
  realpath /home/DSJ44/paper2_yolov11_steel/datasets/NEU-DET
  echo
  echo "\$ realpath /home/DSJ44/sr_project/yolo_dataset_original"
  realpath /home/DSJ44/sr_project/yolo_dataset_original
  echo
  echo "\$ grep -nE '^(path|train|val|names|nc):' $VAL359_YAML $VAL360_YAML"
  grep -nE '^(path|train|val|names|nc):' "$VAL359_YAML" "$VAL360_YAML"
  echo
  echo "--- resolved train directories (path + train, realpath) ---"
  echo "val359 path=$(grep -m1 '^path:' "$VAL359_YAML" | sed 's/^path:[[:space:]]*//')  train=$(grep -m1 '^train:' "$VAL359_YAML" | sed 's/^train:[[:space:]]*//')"
  echo "  -> realpath: $(realpath "$(grep -m1 '^path:' "$VAL359_YAML" | sed 's/^path:[[:space:]]*//')/$(grep -m1 '^train:' "$VAL359_YAML" | sed 's/^train:[[:space:]]*//')")"
  echo "val360 path=$(grep -m1 '^path:' "$VAL360_YAML" | sed 's/^path:[[:space:]]*//')  train=$(grep -m1 '^train:' "$VAL360_YAML" | sed 's/^train:[[:space:]]*//')"
  echo "  -> realpath: $(realpath "$(grep -m1 '^path:' "$VAL360_YAML" | sed 's/^path:[[:space:]]*//')/$(grep -m1 '^train:' "$VAL360_YAML" | sed 's/^train:[[:space:]]*//')")"
} > "$OUT/A_check_path_realpath.txt" 2>&1
cat "$OUT/A_check_path_realpath.txt"

echo "########## A3 verdict ##########"
RP359=$(realpath /home/DSJ44/paper2_yolov11_steel/datasets/NEU-DET)
RP360=$(realpath /home/DSJ44/sr_project/yolo_dataset_original)
T359=$(realpath "$(grep -m1 '^path:' "$VAL359_YAML" | sed 's/^path:[[:space:]]*//')/$(grep -m1 '^train:' "$VAL359_YAML" | sed 's/^train:[[:space:]]*//')")
T360=$(realpath "$(grep -m1 '^path:' "$VAL360_YAML" | sed 's/^path:[[:space:]]*//')/$(grep -m1 '^train:' "$VAL360_YAML" | sed 's/^train:[[:space:]]*//')")
echo "realpath(datasets/NEU-DET)          = $RP359"
echo "realpath(sr_project/yolo_dataset_original) = $RP360"
echo "train realpath val359 = $T359"
echo "train realpath val360 = $T360"
if [ "$RP359" = "$RP360" ] && [ "$T359" = "$T360" ]; then
  echo "A3_VERDICT=PASS  (path lines resolve to the same directory; train resolves identically)"
  echo "path 行差异经 realpath 核验为同一目录。" >> "$OUT/A_check_yaml_diff.txt"
else
  echo "A3_VERDICT=FAIL"
fi

echo "########## B1 ##########"
{
  echo "B1 · APSPC dose A-arm best.pt (3) and val-set yamls (9)"
  echo "source of truth: exp_apspc_dose_unified_20260915/scripts/common.py (ARM_RUNS, VAL_SETS, VAL_YAML)"
  echo
  echo "--- 3 x A_leaktrain best.pt ---"
  for s in 0 1 42; do
    p=exp_apspc_dose_unified_20260915/runs/apspc_yolo11n_leaktrain_unified_s$s/weights/best.pt
    printf '%s  %s\n' "$(sha256sum "$p" | cut -d' ' -f1)" "$p"
  done
  echo
  echo "--- 9 x val-set yaml ---"
  for v in val_d0 val_d20_same val_d28_same val_d41_same val_d82_same val_d20_conf val_d28_conf val_d41_conf val_d82_conf; do
    p=exp_apspc_dose_20260817/configs/apspc_${v#val_}.yaml
    printf '%s  %s\n' "$(sha256sum "$p" | cut -d' ' -f1)" "$p"
  done
} > "$OUT/B_inputs.txt" 2>&1
cat "$OUT/B_inputs.txt"

echo "########## C1 ##########"
wc -l exp_gc10_label_20260914/per_class_ap.csv            >  "$OUT/C_per_class_ap_raw.txt"
cat exp_gc10_label_20260914/per_class_ap.csv              >> "$OUT/C_per_class_ap_raw.txt"
grep -ln 'per_class_ap' exp_gc10_label_20260914/*.py      >  "$OUT/C_generator_script.txt" 2>&1
echo "--- literal grep on exp_gc10_label_20260914/*.py returned: ---"
cat "$OUT/C_generator_script.txt"
echo "--- (no *.py at that level; the generator lives in scripts/) ---"
{
  echo "NOTE: the literal command 'grep -ln per_class_ap exp_gc10_label_20260914/*.py' matched no file"
  echo "(there is no *.py directly under exp_gc10_label_20260914/). The generating script is:"
  echo "  exp_gc10_label_20260914/scripts/ed_eval.py"
  echo "  sha256: $(sha256sum exp_gc10_label_20260914/scripts/ed_eval.py | cut -d' ' -f1)"
  echo
  echo "==================== BEGIN VERBATIM CODE THAT GENERATES per_class_ap.csv ===================="
  echo "--- ed_eval.py lines 106-121 : per-class AP50 extraction from the val() return object ---"
  sed -n '106,121p' exp_gc10_label_20260914/scripts/ed_eval.py
  echo
  echo "--- ed_eval.py lines 141-148 : writing per_class_ap.csv ---"
  sed -n '141,148p' exp_gc10_label_20260914/scripts/ed_eval.py
  echo
  echo "--- ed_eval.py lines 27-31 : CLASS_NAMES and JOBS ---"
  sed -n '27,31p' exp_gc10_label_20260914/scripts/ed_eval.py
  echo "==================== END VERBATIM CODE ===================="
} >> "$OUT/C_generator_script.txt"
wc -l "$OUT/C_generator_script.txt"

echo "########## C2 ##########"
{
  echo "C2 · GC10 label experiment — 6 best.pt (xml x3, dir x3) + XML val yaml"
  for c in xml dir; do
    for s in 0 1 42; do
      p=exp_gc10_label_20260914/runs/gc10_yolo11n_${c}_s${s}/weights/best.pt
      printf '%s  %s\n' "$(sha256sum "$p" | cut -d' ' -f1)" "$p"
    done
  done
  echo
  echo "--- XML condition val yaml ---"
  p=exp_gc10_label_20260914/configs/gc10_xml.yaml
  printf '%s  %s\n' "$(sha256sum "$p" | cut -d' ' -f1)" "$p"
  echo
  echo "--- DIR condition val yaml (listed for completeness, not evaluated) ---"
  p=exp_gc10_label_20260914/configs/gc10_dir.yaml
  printf '%s  %s\n' "$(sha256sum "$p" | cut -d' ' -f1)" "$p"
} > "$OUT/C_inputs.txt" 2>&1
cat "$OUT/C_inputs.txt"

echo "########## PREP DONE ##########"
ls -la "$OUT"
