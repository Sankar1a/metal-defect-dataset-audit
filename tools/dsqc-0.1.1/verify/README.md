# verify — reproducing the audit cross-check

This directory re-runs the check that compares dsqc's scan output with the audit lists published
with the accompanying manuscript. It is not needed to use dsqc itself; it only exists so that the
numbers in the manuscript can be reproduced end to end.

## What is checked

Thirteen items, five for APSPC, two for NEU-DET and six for GC10-DET:

| # | item |
|---|---|
| 1 | APSPC image files |
| 2 | APSPC pixel-identical duplicate groups (groups / files / size distribution) |
| 3 | APSPC groups whose annotations are identical |
| 4 | APSPC group types A / B / C1 / C2 / C3 and the per-group minimum IoU |
| 5 | SHA-256 of the three released APSPC archives |
| 6 | NEU-DET pixel-identical pair (files and pixel hash) |
| 7 | NEU-DET pair crosses the train/val split |
| 8 | GC10-DET images / distinct images / distinct images with XML |
| 9 | GC10-DET stems that occur in two class folders |
| 10 | GC10-DET folder class absent from the image's own XML |
| 11 | GC10-DET disagreements per class folder (1–10) |
| 12 | GC10-DET entries without XML |
| 13 | GC10-DET empty-annotation images |

For every item three counts are printed — `only_ref` (rows in the reference but not in the tool
output), `only_tool` (rows in the tool output but not in the reference) and `field_diff` (rows in
both whose value differs) — plus their sum in `diff_rows`.

## Prerequisites

1. **The three datasets** on disk, anywhere: APSPC, GC10-DET (as released: class folders `1`–`10`
   plus `lable/*.xml`), NEU-DET (YOLO conversion with `images/train`, `images/val`, `labels/…`).
2. **The repository that holds the manuscript's `audit/` directory**, with

   ```
   audit/apspc_duplicate_groups.csv
   audit/apspc_duplicate_boxes.csv
   audit/neu_duplicate_pair.txt
   audit/gc10_cross_folder_duplicates.csv
   audit/gc10_folder_xml_mismatch.csv
   audit/apspc_archive_sha256.txt
   audit/intermediate/gc10_missing_xml.txt      (older layout: audit/audit_scripts/)
   ```

   The scripts accept both layouts of that repository. They look for the files in this order and
   print the path they actually used:

   | file | first choice | fallback |
   |---|---|---|
   | GC10 per-image consistency, GC10 missing-XML list | `audit/intermediate/` | `audit/audit_scripts/` |
   | A/B/C classifier | `audit/scripts/classify_groups.py` | the copy shipped in this directory |

   Two file formats are also accepted: `gc10_folder_xml_mismatch.csv` may be a 10-column CSV with a
   header or a 4-column pipe-separated list without one, and `gc10_cross_folder_duplicates.csv` may
   contain only GC10 (one row per image) or all datasets (one row per group).
3. Python with the packages of `requirements.txt`.

## Usage

The tool is meant to live inside the manuscript repository, at `tools/dsqc-0.1.1/`:

```bash
cd tools/dsqc-0.1.1/verify
./verify_all.sh --repo ../../.. \
                --apspc /path/to/APSPC \
                --gc10  /path/to/gc10_raw \
                --neu   /path/to/NEU-DET \
                --out   /path/to/new_out
```

`--repo` is the repository root, i.e. the directory that contains `audit/`.

`--out` must not exist yet: the script never deletes or overwrites anything. `--tool` defaults to
the parent directory of this script (the dsqc checkout), `--python` to `$PYTHON` or `python3`.
`./verify_all.sh --help` prints the same summary.

Runtime is dominated by hashing: about 1900 APSPC images at 2560×1920, 2300 GC10-DET images at
2048×1000 and 1800 NEU-DET images at 200×200 — roughly two to three minutes on a 16-core machine.

## What the script does

1. scans the three datasets with `python dsqc_run.py scan --spec specs/<name>.yaml <dataset root>`
   into `--out/{apspc,gc10,neu}`;
2. copies the manuscript's `audit/apspc_duplicate_groups.csv` and
   `audit/apspc_duplicate_boxes.csv` into `--out/ref/` and runs the classifier
   (`audit/scripts/classify_groups.py` when present, otherwise the copy in this directory) to
   regenerate `apspc_group_types.csv`. That reference therefore comes from the manuscript's own
   inputs and script, not from dsqc;
   when the repository also ships `audit/apspc_group_types.csv`, the A/B/C item is checked against
   **both** references and the two references are compared byte for byte;
3. runs `compare_with_manuscript.py`, which prints the table and writes `comparison.csv` and
   `comparison.json` into `--out/`.

## Expected result

```
=== 13 items checked, total differing rows = 0 ===
```

i.e. `13 items checked, total differing rows = 0`: thirteen items, and every count column is 0.

with exit status 0. Every item must show `0` in all four count columns. Any non-zero count is a
real discrepancy between the tool output and the audited lists and should be reported, not patched.

## Files

| file | purpose |
|---|---|
| `verify_all.sh` | driver: scans, regenerates the reference, compares |
| `compare_with_manuscript.py` | the comparison itself; see `--help` |
| `classify_groups.py` | the manuscript's own A/B/C1–C3 classifier, used to regenerate the reference |
| `README.md` | this file |

None of the scripts contains a hard-coded path: dataset roots, the repository, the tool checkout and
the output directory are all command-line arguments.
