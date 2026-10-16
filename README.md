# metal-defect-dataset-audit

Audit lists, configurations, logs and scripts for our study of three public metal-surface
defect-detection datasets (NEU-DET, GC10-DET and APSPC). The accompanying manuscript is under
peer review; this repository is cited from its Data Availability Statement.

The study asks how much a reported mAP gain can be moved by data-handling decisions,
compared with changing the model size and with run-to-run seed variation.

## What we found (short version)

| Dataset | Audit finding | Measured effect on mAP50-95 (YOLO11n) |
|---|---|---|
| APSPC | 82 groups of pixel-identical images (174 files, 9.23% of 1885); none of the 82 groups has identical annotations | +1.27 ± 0.21 pp when 7.6% of the validation set duplicates training images (same labels); +0.38 ± 0.13 pp with each copy's own labels |
| GC10-DET | 81 of 2306 annotated images (3.51%) whose folder class does not appear in their XML annotation; 12 file stems stored in two class folders | −1.75 ± 0.87 pp when the folder class is used for those images |
| NEU-DET | 1 pixel-identical pair (`patches_101` / `patches_105`) across our train/val split | +0.092 ± 0.011 pp (paired, same weights) |

For reference, enlarging YOLO11n to YOLO11s (3.64× parameters) gains +0.48 / +0.39 / +0.29 pp
on APSPC / GC10-DET / NEU-DET; YOLO11n seed SD is 0.30–0.78 pp.

## Datasets (not redistributed)

No images or original annotation files are included. Obtain the datasets from their sources:

- **NEU-DET**: Northeastern University surface defect database (Song & Yan, 2013; He et al., 2020).
- **GC10-DET**: Lv et al., *Sensors* 2020, 20, 1562.
- **APSPC**: Alibaba Cloud Tianchi dataset No. 148297 (listed as APDDD; files `APSPC*.zip`).
  Our copies match these SHA-256 checksums (`audit/apspc_archive_sha256.txt`):

  | File | SHA-256 |
  |---|---|
  | APSPC1.zip | `5e0329656371883f07186d8b2835cfce16a97c35eac7a361efd8442fa0d5eee8` |
  | APSPC2.zip | `aae3ba56181dae93a58c8a6845ae42e220505879c44a8c1b65803359a9bccecb` |
  | APSPC-Annotations.zip | `4be5912d63558c8f326410b12e6ce3413644bd644dbdb4d58fdb579d66a04f22` |

`audit/apspc_duplicate_boxes.csv` contains only the box coordinates and class names of the 174
duplicated APSPC files, which are needed to check the annotation comparison.

## Repository layout

| Folder | Contents | Manuscript |
|---|---|---|
| `audit/` | Duplicate groups (`apspc_duplicate_groups.csv`), their boxes, the annotation-difference type of every group (`apspc_group_types.csv`: A 30, B 5, C1 11, C2 27, C3 9), the GC10-DET folder–XML mismatch list (81 rows) and cross-folder duplicates (12 stems), the NEU-DET pair record, archive checksums | Sections 2.2, 3.1 |
| `audit/scripts/` | Byte- and pixel-hash duplicate search, group building, label comparison, group-type classification, GC10-DET list derivation | Section 2.2 |
| `audit/intermediate/` | Raw outputs of the audit steps (per-image GC10-DET consistency table, hash tables, checks) | Section 2.2 |
| `splits/` | Train/val/test file lists for all three datasets, the APSPC dose-response validation sets (d0, d20, d28, d41, d82; `same` and `conf`), the GC10-DET label-source lists, and the scripts that built them | Sections 2.1, 2.4 |
| `configs/` | The `args.yaml` saved by Ultralytics for every one of the 78 training runs, renamed `<run>.yaml`; data YAMLs in `configs/data/` | Sections 2.3, 2.4 |
| `results/` | Evaluation summaries for every experiment, `training_curves/<run>.csv` (78 runs), `optimizer_lines.txt` (the optimizer line from every training log), parameter-gate diffs, and bitwise re-run checks | Sections 2.3–2.6, 3 |
| `scripts/` | Training, evaluation and checking scripts, grouped by experiment round | Sections 2.4–2.6 |
| `figures/` | Scripts for the manuscript figures | Figures 1–3 |
| `docs/` | Deviation notes written during the runs | Supplementary Table S6 |

## Environment

All runs used one machine: NVIDIA GeForce RTX 4080 (16 GB), driver 550.54.14, Python 3.10.20,
PyTorch 2.3.1 (CUDA 12.1), Ultralytics 8.3.40, one training job at a time.

Unified protocol (YOLO11 and YOLOv8): SGD, lr 0.01, momentum 0.937, 200 epochs, patience 50,
image size 640, batch 16, seeds 0 / 1 / 42, `deterministic=True`, `optimizer` set explicitly
(not `auto`). RT-DETR-L: AdamW, lr 1e-4, weight decay 1e-4, 100 epochs. The full parameter
set of each run is in its `configs/<run>.yaml`; `results/optimizer_lines.txt` shows that 66 runs
used `SGD(lr=0.01, momentum=0.937)` and the 12 RT-DETR-L runs used `AdamW(lr=0.0001, momentum=0.9)`.

## Reproducing

1. **Audit.** Run the scripts in `audit/scripts/` on the downloaded datasets. Two steps can be
   re-run directly from files in this repository:
   ```bash
   python audit/scripts/classify_groups.py            # needs numpy, scipy -> audit/apspc_group_types.csv
   python audit/scripts/gc10_lists_from_consistency.py # -> GC10-DET mismatch and cross-folder lists
   ```
2. **Splits.** Use the lists in `splits/`. The original scripts for the APSPC and NEU-DET
   splits were not kept; the lists (with SHA-256 for APSPC) define the splits exactly.
   GC10-DET was split by `splits/gc10/prepare_gc10.py` (seed 0, 80/20, not stratified).
3. **Training.** `yolo train cfg=configs/<run>.yaml` reproduces a run once the paths inside the
   YAML point to your copies of the data (they still contain our server paths).
4. **Evaluation.** All reported numbers come from re-evaluating `best.pt` on the target
   validation set (image size 640, batch 16). The scripts are in `scripts/`; outputs are in `results/`.
5. **Figures.** `figures/fig3_dose_response.py` contains its data and runs on its own.
   `figures/fig2_duplicate_examples.py` needs the APSPC images; `figures/fig1_study_design.py`
   uses thumbnails cut from Figure 2.

Bitwise reproducibility was checked only on the machine, driver and software versions above.

## Notes

- Run names: `<dataset>_<model>_<condition>_s<seed>`; `leaktrain` = APSPC dose-response training set
  (1280 images), `clean_repro` = re-run of the APSPC baseline, `xml` / `dir` = GC10-DET label-source conditions.
- Configuration files and some scripts contain absolute paths from our server (`/home/DSJ44/...`).
- The GC10-DET folder–XML check was first run interactively; `gc10_lists_from_consistency.py`
  re-derives both lists from the saved per-image table.

## License

Code: MIT License (see `LICENSE`). Lists, tables and logs: CC BY 4.0. The datasets themselves
remain under their original terms.

## Citation

To be added after publication.
