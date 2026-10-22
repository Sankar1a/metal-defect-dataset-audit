# dsqc 0.1.1 — dataset quality check for object-detection datasets

dsqc reads an object-detection dataset (images + labels) and writes a report about the
inconsistencies it can detect with deterministic rules. It never trains or evaluates a model, never
writes inside the dataset, and uses CPU only.

The output is meant to be read by a person: a `report.html` for browsing, plus CSV/JSON files that
can be diffed between two dataset versions.

## Install

Tested with **Python 3.10** (3.10.20). Only four third-party packages are needed:

```bash
pip install -r requirements.txt      # numpy, scipy, Pillow, PyYAML (versions pinned)
```

Entry point: **`dsqc_run.py`**.

```bash
python dsqc_run.py --help
python dsqc_run.py scan --help
```

`python -I -m dsqc ...` does **not** work: isolated mode removes the working directory from
`sys.path`, so the package is not found. Use `python dsqc_run.py ...` (with or without `-I`), or
`python -m dsqc ...` without `-I`.

## Supported formats

| Format | Layout understood out of the box | Reader |
|---|---|---|
| YOLO | `images/<split?>/x.jpg` + `labels/<split?>/x.txt`, one `cls cx cy w h` per line | `dsqc/io/yolo.py` |
| Pascal VOC | one `.xml` per image, matched to the image by file stem | `dsqc/io/voc.py` |
| COCO | one JSON with `images`, `annotations`, `categories` (`bbox = [x, y, w, h]`) | `dsqc/io/coco.py` |

Anything else (class folders, labels in an unusual directory, several label sources, no split, a
split encoded in a path component, …) is described by a small **spec YAML**; three complete examples
are in `specs/`. A spec is read relative to a dataset root, so the same spec can be reused on another
copy of the dataset:

```bash
# the positional argument overrides `root:` in the spec
python dsqc_run.py scan --spec specs/gc10_raw.yaml /data/GC10-DET -o out/gc10
# standard layout, no spec
python dsqc_run.py scan /data/my_yolo_dataset -o out/mine
# what changed between two scans
python dsqc_run.py diff out/v1 out/v2 -o diff.csv
```

Minimal spec (paths are relative to `root` unless absolute):

```yaml
name: my dataset
root: /path/to/dataset          # placeholder; the positional argument overrides it
images: ["*/*.jpg"]             # globs over image files; default: every image file
exclude: []                     # globs to drop
split: none                     # or {type: component, index: 1}
                                # or {type: lists, lists: {train: train.txt, val: val.txt}}
class_names: [hole, spot]       # optional; enables the unknown-class check
class_aliases: {10_yaozhed: 10_yaozhe}   # optional: variant spellings
primary: xml                    # the main box source
sources:
  - {name: xml,    type: voc, dirs: [annotations]}
  - {name: yolo,   type: yolo, labels_dir: labels, images_dir: images}
  - {name: coco,   type: coco, json: annotations/train.json, image_prefix: images/train}
  - {name: folder, type: folder_class, level: -1, map: {"1": hole, "2": spot}}
```

`type: voc` also accepts `folder_tag_source: xml_folder` (read the `<folder>` element as an
image-level class), `folder_tag_map` (map its raw values to class names), and
`folder_tag_findings: false` (cross-tab only — use it when no class mapping is known). The same holds
for `folder_class` with `findings: false`.

## Checks

Seven checks run on every scan. `--skip c04a_transform_dup,c06_box_sanity` skips some of them;
`--config my.yaml` overrides the thresholds below. Other options: `--workers N` (default
`min(16, CPU count)`), `--no-transform` (skip `c04a`), `--thumbs N` (thumbnails per check, default 0
— images then never leave the dataset), `--cache-dir DIR` (hash cache; safe to delete).

| id | What it looks at | Rules / defaults | Tables written |
|---|---|---|---|
| `c00_inventory` | counts at every level: image files, decode errors, distinct stems, stems in several files, distinct byte/pixel hashes, distinct after rotation+flip, images with/without labels, empty labels, orphan annotations, images per split | counting only, no threshold; each row states its own definition | `c00_inventory__inventory.csv` |
| `c01_exact_dup` | images whose decoded RGB pixels are identical, even when the bytes differ | `pixel_sha256 = sha256(PIL Image.convert('RGB').tobytes())`; a group spanning several splits is reported as leakage (high); with no split, a group is reported as a risk (medium) | `c01_exact_dup__pixel_duplicate_groups.csv` |
| `c02_dup_label_types` | how the annotations inside one duplicate group differ | `identical` (same multiset of boxes) · `A` class sets differ · `B` box count of a class differs · `C1/C2/C3` same classes and counts, minimum IoU after one-to-one matching (`c_iou_low = 0.5`, `c_iou_high = 0.9`) · `shared_file` (two copies share one annotation file) | `c02_dup_label_types__duplicate_group_types.csv` |
| `c03_source_consistency` | agreement between label sources of the same image: image-level class (folder, XML `<folder>`) vs box classes, and box source vs box source | an image-level class counts as present only if it occurs among the box classes (an empty annotation counts as not present); unmapped `<folder>` values are counted separately and not compared; box vs box uses the `c02` comparator; never decides which source is right | `c03_source_consistency__per_image_*.csv`, `…__by_class_*.csv`, `…__contingency_*.csv` |
| `c04a_transform_dup` | images that become pixel-identical after a rotation (90/180/270) or a flip | `canon_sha256` = minimum hash over the 8 dihedral transforms; no threshold; groups already found by `c01` are not repeated | `c04a_transform_dup__rot_flip_groups.csv` |
| `c06_box_sanity` | annotation files and boxes of every box source | `missing_annotation` (medium), `empty_annotation` (low), `unreadable_annotation` / `parse_issue` / `unknown_class` / `degenerate_box` (high), `size_mismatch` / `out_of_bounds` / `duplicate_box` (medium), `alias_class` (info), `tiny_box` (`tiny_min_side_px = 4.0` px, low), `full_image_box` (`full_image_ratio = 0.95`, low), `duplicate_box` (`dup_box_iou = 0.95`), `oob_tol_px = 0.0` | `c06_box_sanity__box_issues.csv`, `…__class_counts_*.csv` |
| `c07_split_version` | split composition and a fingerprint of the whole dataset | per split and class: images containing the class, boxes, share of the split; `manifest.csv` with size and SHA-256 of every image and annotation file; `fingerprint_sha256` over the sorted `path<TAB>sha256` lines, so two byte-identical copies share one fingerprint | `manifest.csv`, `c07_split_version__split_class_counts.csv` |

`max_rows_html` (default 200) only limits how many findings appear per check in `report.html`; every
finding is always written to `findings.csv`.

## Outputs

| File | Content |
|---|---|
| `summary.json` | dataset name, root, layout, split mode, sources; per check: title, statistics, notes, findings by code and by severity; totals |
| `run_info.json` | version, full command, start/end/seconds, Python version and executable, platform, package versions, workers, thresholds, spec path and its SHA-256, thumbnail count |
| `findings.csv` | one row per finding: id, check, code, severity, number of images, image list, message, extra data as JSON |
| `manifest.csv` | one row per image and annotation file: kind, path, size, SHA-256 |
| `report.html` | self-contained HTML report, one section per check with tables and findings; images are embedded only if `--thumbs N` is given |
| `tables/<check>__<table>.csv` | the per-check tables listed above |

The output directory must not exist or must be empty; nothing is ever written inside the dataset.

## Reproducing the audit reported in the accompanying manuscript

`verify/` re-runs the cross-check of the three reference datasets (APSPC, GC10-DET, NEU-DET) against
the audit lists of the accompanying manuscript.

Requirements:

1. the three datasets on disk, any location;
2. a copy of the repository that holds the manuscript's `audit/` directory
   (`audit/apspc_duplicate_groups.csv`, `audit/apspc_duplicate_boxes.csv`,
   `audit/neu_duplicate_pair.txt`, `audit/gc10_cross_folder_duplicates.csv`,
   `audit/gc10_folder_xml_mismatch.csv`, `audit/audit_scripts/gc10_missing_xml.txt`,
   `audit/audit_scripts/gc10_folder_xml_consistency.csv`, `audit/apspc_archive_sha256.txt`);
3. Python with the pinned packages.

```bash
cd verify
./verify_all.sh --tool .. --repo /path/to/manuscript/repo \
                --apspc /path/to/APSPC --gc10 /path/to/gc10_raw --neu /path/to/NEU-DET \
                --out /tmp/dsqc_verify_out
```

The script scans the three datasets once, regenerates the A/B/C1–C3 reference from the manuscript's
own two CSVs with `verify/classify_groups.py` (so that reference does not depend on dsqc), and prints
one line per checked item with three difference counts (only in the reference / only in the tool /
field differs). Expected result: **13 items checked, 0 differing rows for every item**. See
`verify/README.md`.

## Limitations

- Duplicate detection only recognises **pixel-identical** images (and, in `c04a`, images that become
  identical after a rotation or a flip). Cropped, resized or re-compressed near-duplicates are not
  detected.
- The tool does not train or evaluate models, does not cover multi-seed evaluation, and does not read
  training logs.
- The label-source consistency check needs two sources declared in the spec; with one source it
  reports nothing.
- Checks are rule based: they point at inconsistencies and anomalies, they do not decide which label
  is correct.
