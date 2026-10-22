#!/usr/bin/env bash
# Re-run the cross-check of the three reference datasets against the manuscript's audit lists.
#
#   ./verify_all.sh --repo DIR --apspc DIR --gc10 DIR --neu DIR --out DIR [--tool DIR] [--python CMD]
#
# The datasets and the manuscript repository are only read. Everything this script writes goes into
# --out, which must not exist yet: no directory is deleted or overwritten.
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"

usage() {
  cat <<'USAGE'
usage: verify_all.sh --repo DIR --apspc DIR --gc10 DIR --neu DIR --out DIR [options]

  --repo    DIR   root of the repository that holds the manuscript's audit/ directory
  --apspc   DIR   root of the APSPC dataset
  --gc10    DIR   root of the GC10-DET dataset
  --neu     DIR   root of the NEU-DET dataset
  --out     DIR   output directory; must not exist yet
  --tool    DIR   dsqc checkout (default: the parent directory of this script)
  --python  CMD   Python interpreter (default: $PYTHON if set, else python3)
  -h, --help      show this help and exit

What it does
  1. scans the three datasets with dsqc (specs/apspc_raw.yaml, specs/gc10_raw.yaml,
     specs/neu_yolo.yaml; the dataset roots are passed as the positional argument)
  2. regenerates the A/B/C1-C3 reference from the manuscript's own
     audit/apspc_duplicate_groups.csv + audit/apspc_duplicate_boxes.csv with classify_groups.py,
     so that reference does not depend on dsqc
  3. runs compare_with_manuscript.py, which prints one line per checked item with the counts
     only-in-reference / only-in-tool / field-differs

Expected result: 13 items, 0 differing rows for every item.
USAGE
}

REPO=""; APSPC=""; GC10=""; NEU=""; OUT=""
TOOL="$(cd "$HERE/.." && pwd)"
PY="${PYTHON:-python3}"

need_value() {
  if [ "$#" -lt 2 ]; then
    echo "STOP: option $1 needs a value" >&2
    exit 2
  fi
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --repo)   need_value "$@"; REPO="$2";   shift 2 ;;
    --apspc)  need_value "$@"; APSPC="$2";  shift 2 ;;
    --gc10)   need_value "$@"; GC10="$2";   shift 2 ;;
    --neu)    need_value "$@"; NEU="$2";    shift 2 ;;
    --out)    need_value "$@"; OUT="$2";    shift 2 ;;
    --tool)   need_value "$@"; TOOL="$2";   shift 2 ;;
    --python) need_value "$@"; PY="$2";     shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "STOP: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for spec in "repo:$REPO" "apspc:$APSPC" "gc10:$GC10" "neu:$NEU" "out:$OUT"; do
  name="${spec%%:*}"; value="${spec#*:}"
  if [ -z "$value" ]; then
    echo "STOP: missing --$name" >&2
    usage >&2
    exit 2
  fi
done

for d in "$REPO" "$APSPC" "$GC10" "$NEU" "$TOOL"; do
  if [ ! -d "$d" ]; then
    echo "STOP: not a directory: $d" >&2
    exit 2
  fi
done
if [ ! -f "$TOOL/dsqc_run.py" ]; then
  echo "STOP: $TOOL does not look like a dsqc checkout (no dsqc_run.py); use --tool" >&2
  exit 2
fi

# No deletion, no overwrite.
if [ -e "$OUT" ]; then
  echo "STOP: output directory already exists: $OUT" >&2
  echo "      nothing was deleted or overwritten; pass a new --out" >&2
  exit 3
fi
mkdir -p "$OUT" || exit 3

step() { printf '\n[verify] %s\n' "$*"; }

step "scanning APSPC"
( cd "$TOOL" && "$PY" -I dsqc_run.py scan --spec specs/apspc_raw.yaml "$APSPC" -o "$OUT/apspc" ) || exit 4

step "scanning GC10-DET"
( cd "$TOOL" && "$PY" -I dsqc_run.py scan --spec specs/gc10_raw.yaml "$GC10" -o "$OUT/gc10" ) || exit 4

step "scanning NEU-DET"
( cd "$TOOL" && "$PY" -I dsqc_run.py scan --spec specs/neu_yolo.yaml "$NEU" -o "$OUT/neu" ) || exit 4

step "regenerating the A/B/C1-C3 reference from the manuscript's own CSVs"
mkdir -p "$OUT/ref" || exit 5
# repository layout first, then the copy shipped next to this script
CLASSIFIER="$REPO/audit/scripts/classify_groups.py"
[ -f "$CLASSIFIER" ] || CLASSIFIER="$HERE/classify_groups.py"
if [ ! -f "$CLASSIFIER" ]; then
  echo "STOP: classify_groups.py not found in the repository or next to this script" >&2
  exit 5
fi
echo "[verify] classifier: $CLASSIFIER"
cp "$REPO/audit/apspc_duplicate_groups.csv" "$OUT/ref/" || exit 5
cp "$REPO/audit/apspc_duplicate_boxes.csv" "$OUT/ref/" || exit 5
"$PY" -I "$CLASSIFIER" "$OUT/ref" || exit 5

step "comparing the scans with the audit lists"
"$PY" -I "$HERE/compare_with_manuscript.py" \
  --apspc "$OUT/apspc" --gc10 "$OUT/gc10" --neu "$OUT/neu" \
  --repo "$REPO" --ref "$OUT/ref" \
  --apspc-dataset "$APSPC" --gc10-dataset "$GC10" \
  --out-csv "$OUT/comparison.csv" --out-json "$OUT/comparison.json"
rc=$?
if [ "$rc" -eq 0 ]; then
  step "OK: every item has 0 differing rows"
else
  step "some items have differing rows (see the table above); exit status $rc"
fi
exit "$rc"
