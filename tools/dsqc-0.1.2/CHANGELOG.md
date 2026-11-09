# Changelog

## 0.1.2

Two problems found when dsqc was run on datasets outside the accompanying manuscript
(a PCB defect dataset and the BCCD blood-cell dataset). Every statistic, table column and finding
that 0.1.1 writes is unchanged for datasets without split lists, except the severity of C1 groups
flagged as small-box jitter.

- **Split lists given as file names without extension were not read.** Pascal VOC
  `ImageSets/Main/*.txt` and many other releases list images by stem (`BloodImage_00001`); 0.1.1
  matched only relative paths and file names with extension, so every image stayed without a split.
  Lines are now matched by the longest trailing part that names an image, with or without extension;
  Windows paths, `./` prefixes, VOC per-class flags, comments, blank lines and a byte-order mark are
  handled. A line that fits several images is still skipped. The matching is a dictionary lookup
  instead of a scan over all images per line.
  - `c07` reports `split_list.<split>.entries / matched / not_found / ambiguous`, and a
    `split_list_entries_unmatched` finding (medium) when some lines did not match.
  - Unmatched lines are summarised in one note per list (count and the first five) instead of one
    note per line.
- **Small boxes with a few pixels of offset were reported as large annotation differences.** The
  C1/C2/C3 types use fixed IoU thresholds; on boxes of a few dozen pixels, an offset of a few pixels
  gives IoU < 0.5. `c02` keeps the types and adds, for C-type groups:
  `max_coord_diff_px`, `max_center_offset_px`, `low_iou_max_area_px2` and `small_box_jitter`
  (`yes` when every matched pair with IoU < `c_iou_high` is two boxes with area < `small_box_area_px2`
  = 1024 px² whose centres lie inside each other). Flagged C1/C2 groups are reported with severity
  low (`jitter_downgrade: false` keeps the 0.1.1 severity). New statistics:
  `type_C1_small_box_jitter`, `type_C2_small_box_jitter`.
- Tests: 4 new unit tests (split lists written in several styles; the jitter flag and its
  configuration; the geometry helper). `verify/compare_versions.py <dsqc-0.1.1> <dsqc-0.1.2> <work dir>` scans the
  synthetic datasets with both versions and lists every difference in the outputs that 0.1.1 writes.

## 0.1.1

First public version, used for the accompanying manuscript.
