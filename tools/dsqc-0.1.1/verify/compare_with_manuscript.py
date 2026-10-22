#!/usr/bin/env python3
"""Compare dsqc scan outputs with the audit lists published with the accompanying manuscript.

Thirteen items are checked (five for APSPC, two for NEU-DET, six for GC10-DET). For each item the
script counts

    only_ref    rows present in the reference but not in the tool output
    only_tool   rows present in the tool output but not in the reference
    field_diff  rows present in both but with a different field value

and prints the sum in the last column. Rows are matched by a key (a set of file names, a relative
path, a folder id, ...), not by row order, so the two sides may be sorted differently.

Reading is read-only; the only files written are the two result files given by --out-csv/--out-json.

Usage
  python -I compare_with_manuscript.py --apspc OUT/apspc --gc10 OUT/gc10 --neu OUT/neu \
         --repo /path/to/manuscript/repo [--ref DIR] [--apspc-dataset DIR]

Exit status is 0 when all thirteen items have zero differing rows, 1 otherwise.
"""
import argparse
import csv
import hashlib
import json
import os
import sys

RESULTS = []


def add(item, ref_value, tool_value, ref_file, only_ref, only_tool, field_diff, note=""):
    RESULTS.append({
        "item": item, "reference": ref_value, "tool": tool_value, "ref_file": ref_file,
        "only_ref": only_ref, "only_tool": only_tool, "field_diff": field_diff,
        "diff_rows": only_ref + only_tool + field_diff, "note": note,
    })


def csv_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_checks(out, name=""):
    path = os.path.join(out, name, "summary.json") if name else os.path.join(out, "summary.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)["checks"]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(base, path):
    """Path shown in the output: relative to the repository when possible."""
    try:
        r = os.path.relpath(path, base)
        return r if not r.startswith("..") else path
    except ValueError:
        return path


def resolve(*candidates):
    """First existing candidate (used to accept both repository layouts)."""
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def need(path, what):
    if not os.path.exists(path):
        sys.exit(f"STOP: {what} not found: {path}")
    return path


# --------------------------------------------------------------------------- APSPC
def check_apspc(a, rows):
    out, repo, ref = a.apspc, a.repo, a.ref
    c = load_checks(out)

    # 1. image count
    n_img = c["c00_inventory"]["stats"]["image_files"]
    add("APSPC image files", 1885, n_img, "(published value)", 0, 0, 0 if n_img == 1885 else 1)

    # 2. pixel-identical duplicate groups, matched by the set of (part, file name)
    ref_groups, ref_sha = {}, {}
    for r in rows(os.path.join(repo, "audit", "apspc_duplicate_groups.csv")):
        ref_groups.setdefault(r["group_id"], set()).add((r["part"], r["filename"]))
        ref_sha[r["group_id"]] = r["pixel_sha256"]
    ref_map = {frozenset(v): k for k, v in ref_groups.items()}
    tool_groups, tool_sha = {}, {}
    for r in rows(os.path.join(out, "tables", "c01_exact_dup__pixel_duplicate_groups.csv")):
        parts = r["file"].split("/")
        tool_groups.setdefault(r["group_id"], set()).add((parts[-2], parts[-1]))
        tool_sha[r["group_id"]] = r["pixel_sha256"]
    tool_map = {frozenset(v): k for k, v in tool_groups.items()}
    sha_diff = sum(1 for k in set(ref_map) & set(tool_map)
                   if ref_sha[ref_map[k]] != tool_sha[tool_map[k]])
    sizes_ref = {n: sum(1 for v in ref_groups.values() if len(v) == n) for n in (2, 3)}
    sizes_tool = {n: sum(1 for v in tool_groups.values() if len(v) == n) for n in (2, 3)}
    add("APSPC pixel-identical duplicate groups",
        f"{len(ref_groups)} groups / {sum(len(v) for v in ref_groups.values())} files "
        f"({sizes_ref[2]}x2 + {sizes_ref[3]}x3)",
        f"{len(tool_groups)} groups / {sum(len(v) for v in tool_groups.values())} files "
        f"({sizes_tool[2]}x2 + {sizes_tool[3]}x3)",
        "audit/apspc_duplicate_groups.csv",
        len(set(ref_map) - set(tool_map)), len(set(tool_map) - set(ref_map)), sha_diff,
        "field_diff = groups whose pixel_sha256 differs")

    # 3. groups whose annotations are identical
    ident = c["c02_dup_label_types"]["stats"].get("type_identical")
    add("APSPC identical-annotation groups", 0, ident, "(published value)", 0, 0,
        0 if ident == 0 else 1)

    # 4. A / B / C1 / C2 / C3, per group.  Two references are used when both exist:
    #    A = the repository's own audit/apspc_group_types.csv (present in the GitHub layout),
    #    B = the file regenerated from audit/apspc_duplicate_groups.csv + ..._boxes.csv by the
    #        manuscript's classifier (verify_all.sh writes it into --ref).
    gt_repo = os.path.join(repo, "audit", "apspc_group_types.csv")
    gt_regen = os.path.join(ref, "apspc_group_types.csv") if ref else None
    tool_rows = rows(os.path.join(out, "tables", "c02_dup_label_types__duplicate_group_types.csv"))
    tool = {frozenset(os.path.basename(x) for x in r["files"].split("|")): r for r in tool_rows}
    stats = c["c02_dup_label_types"]["stats"]
    counts = "/".join(str(stats.get(f"type_{t}", 0)) for t in ("A", "B", "C1", "C2", "C3"))

    def abc_diff(path):
        gt = {frozenset(r["files"].split("|")): r for r in rows(path)}
        mis = sum(1 for k in set(gt) & set(tool)
                  if gt[k]["type"] != tool[k]["type"] or gt[k].get("min_iou") != tool[k].get("min_iou"))
        return len(set(gt) - set(tool)), len(set(tool) - set(gt)), mis

    refs = []
    if os.path.isfile(gt_repo):
        refs.append(("A", gt_repo, "audit/apspc_group_types.csv"))
    if gt_regen and os.path.isfile(gt_regen):
        refs.append(("B", gt_regen,
                     f"{os.path.basename(os.path.normpath(ref))}/apspc_group_types.csv"))
    if not refs:
        add("APSPC types A/B/C1/C2/C3", "30/5/11/27/9", f"{counts} (reference missing: not compared)",
            "audit/apspc_group_types.csv", 0, 0, 0,
            "no A/B/C reference found; use a repository that has one, or pass --ref DIR")
    else:
        res = {}
        for tag, path, label in refs:
            r = abc_diff(path)
            res[tag] = (r, label)
            print(f"[ref] A/B/C reference {tag}: {label}  sha256={sha256_file(path)[:12]}...  "
                  f"only_ref={r[0]} only_tool={r[1]} field_diff={r[2]} -> {sum(r)}")
        if len(refs) == 2:
            # The repository CSV and the regenerated CSV may use different line endings (LF vs the
            # CRLF that the csv module writes by default), so compare after normalising \r\n -> \n.
            def _normalised(path):
                with open(path, "rb") as f:
                    return f.read().replace(b"\r\n", b"\n")

            for tag, path, label in refs:
                raw = sha256_file(path)
                norm = hashlib.sha256(_normalised(path)).hexdigest()
                print(f"[ref] A/B/C reference {tag} ({label}): sha256={raw} "
                      f"(LF-normalised: {norm})")
            same = _normalised(refs[0][1]) == _normalised(refs[1][1])
            print(f"[ref] A/B/C references identical after normalizing line endings: "
                  f"{'yes' if same else 'NO'}")
        worst_tag = max(res, key=lambda t: sum(res[t][0]))
        only_ref, only_tool, mis = res[worst_tag][0]
        add("APSPC types A/B/C1/C2/C3", "30/5/11/27/9", counts,
            " + ".join(f"{t}:{res[t][1]}" for t in sorted(res)),
            only_ref, only_tool, mis,
            "field_diff = groups whose type or min_iou differs; counts are the worse of the references")

    # 5. SHA-256 of the released archives
    arch = os.path.join(repo, "audit", "apspc_archive_sha256.txt")
    if a.apspc_dataset and os.path.isfile(arch):
        only_ref = mis = 0
        pairs = 0
        with open(arch, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                digest, name = line.split(None, 1)
                pairs += 1
                p = os.path.join(a.apspc_dataset, name.strip())
                if not os.path.isfile(p):
                    only_ref += 1
                elif sha256_file(p) != digest:
                    mis += 1
        add("APSPC archive SHA-256", f"{pairs} archives", f"{pairs - only_ref - mis}/{pairs} match",
            "audit/apspc_archive_sha256.txt", only_ref, 0, mis,
            "recomputed here; dsqc itself does not hash archives")
    else:
        add("APSPC archive SHA-256", "see reference file", "not checked",
            "audit/apspc_archive_sha256.txt", 0, 0, 0, "pass --apspc-dataset DIR to enable")


# --------------------------------------------------------------------------- NEU
def check_neu(a, rows):
    out, repo = a.neu, a.repo

    def rel(p):
        p = p.strip().strip("'\"[] ")
        i = p.find("/images/")
        return "images/" + p[i + len("/images/"):] if i >= 0 else p

    ref_sha, ref_files = None, set()
    with open(os.path.join(repo, "audit", "neu_duplicate_pair.txt"), encoding="utf-8") as f:
        for line in f:
            if "pixel_sha256:" in line:
                ref_sha = line.split("pixel_sha256:")[1].strip()
            if "excluded file:" in line:
                ref_files.add(rel(line.split("excluded file:", 1)[1]))
            if "matched train files" in line:
                for x in line.split(":", 1)[1].replace("[", "").replace("]", "").split(","):
                    if x.strip():
                        ref_files.add(rel(x))
    ref_files.discard("")

    tool_rows = rows(os.path.join(out, "tables", "c01_exact_dup__pixel_duplicate_groups.csv"))
    tool_files = {r["file"] for r in tool_rows}
    tool_sha = tool_rows[0]["pixel_sha256"] if tool_rows else None
    tool_cross = tool_rows[0]["cross_split"] if tool_rows else None

    add("NEU-DET pixel-identical pair",
        f"{len(ref_files)} files, sha {str(ref_sha)[:12]}...",
        f"{len(tool_files)} files, sha {str(tool_sha)[:12]}...",
        "audit/neu_duplicate_pair.txt",
        len(ref_files - tool_files) + (0 if ref_sha == tool_sha else 1),
        len(tool_files - ref_files), 0,
        "reference paths are absolute in the source file; compared as paths under images/")
    add("NEU-DET pair crosses train/val", "yes", "True" if tool_cross == "True" else str(tool_cross),
        "audit/neu_duplicate_pair.txt", 0, 0, 0 if tool_cross == "True" else 1,
        "field read from cross_split")


# --------------------------------------------------------------------------- GC10
def check_gc10(a, rows):
    out, repo = a.gc10, a.repo
    c = load_checks(out)
    inv = c["c00_inventory"]["stats"]
    per = rows(os.path.join(out, "tables", "c03_source_consistency__per_image_folder_vs_xml.csv"))

    got = (inv["image_files"], inv["distinct_pixel_sha256"], inv["xml.annotation_files"])
    add("GC10-DET images / distinct / with XML", "2312 / 2300 / 2294",
        f"{got[0]} / {got[1]} / {got[2]}", "(published values)", 0, 0,
        0 if got == (2312, 2300, 2294) else 1,
        "with-XML uses xml.annotation_files (number of XML files)")

    xfolder = need(os.path.join(repo, "audit", "gc10_cross_folder_duplicates.csv"),
                   "audit/gc10_cross_folder_duplicates.csv")
    ref_stems = set()
    with open(xfolder, encoding="utf-8") as f:
        rd = csv.DictReader(f)
        cols, xrows = rd.fieldnames or [], list(rd)
    if "dataset" in cols:               # server layout: three datasets in one file
        for r in xrows:
            if r["dataset"] == "GC10_RAW":
                for p in r["file_paths"].split(";"):
                    ref_stems.add(os.path.splitext(os.path.basename(p))[0])
        layout = "three-dataset layout, GC10_RAW rows only"
    else:                               # GitHub layout: one row per image, GC10 only
        for r in xrows:
            ref_stems.add(r.get("stem") or os.path.splitext(os.path.basename(r["relative_path"]))[0])
        layout = "GC10-only layout, one row per image"
    print(f"[ref] GC10 cross-folder duplicates: audit/gc10_cross_folder_duplicates.csv "
          f"({layout}, {len(xrows)} rows)")
    tool_stems = {r["stem"] for r in per if int(r["stem_file_count"]) > 1}
    add("GC10-DET stems in two folders", f"{len(ref_stems)}", f"{len(tool_stems)}",
        "audit/gc10_cross_folder_duplicates.csv",
        len(ref_stems - tool_stems), len(tool_stems - ref_stems), 0,
        "the reference file lists several datasets; only its GC10_RAW rows are used")

    mm_path = need(os.path.join(repo, "audit", "gc10_folder_xml_mismatch.csv"),
                   "audit/gc10_folder_xml_mismatch.csv")
    ref_mism = set()
    with open(mm_path, encoding="utf-8") as f:
        first = f.readline()
        f.seek(0)
        if first.startswith("relative_path,"):      # 10-column CSV with a header
            for r in csv.DictReader(f):
                ref_mism.add(r["relative_path"])
            mm_layout = "10-column CSV"
        else:                                        # 4 columns, pipe separated, no header
            for line in f:
                line = line.strip()
                if line:
                    ref_mism.add(line.split("|")[0].strip())
            mm_layout = "4-column pipe-separated, no header"
    print(f"[ref] GC10 folder-vs-XML mismatches: audit/gc10_folder_xml_mismatch.csv "
          f"({mm_layout}, {len(ref_mism)} rows)")
    tool_mism = {r["relative_path"] for r in per if r["class_in_boxes"] == "no"}
    denom = sum(1 for r in per if r["box_label_status"] in ("ok", "empty"))
    add("GC10-DET folder class absent from own XML", f"{len(ref_mism)} (denominator 2306)",
        f"{len(tool_mism)} (denominator {denom})", "audit/gc10_folder_xml_mismatch.csv",
        len(ref_mism - tool_mism), len(tool_mism - ref_mism), 0,
        "the reference file may be a 10-column CSV or a 4-column pipe-separated list")

    def per_folder(paths):
        d = {str(i): 0 for i in range(1, 11)}
        for p in paths:
            d[p.split("/")[0]] = d.get(p.split("/")[0], 0) + 1
        return d

    rf, tf = per_folder(ref_mism), per_folder(tool_mism)
    keys = [str(i) for i in range(1, 11)]
    add("GC10-DET disagreements per folder (1-10)",
        "/".join(str(rf[k]) for k in keys), "/".join(str(tf[k]) for k in keys),
        "audit/gc10_folder_xml_mismatch.csv", 0, 0, sum(1 for k in keys if rf[k] != tf[k]),
        "field_diff = number of folders whose count differs")

    miss_path = need(resolve(
        os.path.join(repo, "audit", "intermediate", "gc10_missing_xml.txt"),
        os.path.join(repo, "audit", "audit_scripts", "gc10_missing_xml.txt")),
        "audit/intermediate|audit_scripts/gc10_missing_xml.txt")
    print(f"[ref] GC10 missing-XML list: {rel(repo, miss_path)}")
    ref_missing = set()
    with open(miss_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ref_missing.add(os.path.relpath(line, a.gc10_dataset))
    tool_missing = {r["relative_path"] for r in per if r["box_label_status"] == "missing"}
    add("GC10-DET entries without XML", f"{len(ref_missing)}", f"{len(tool_missing)}",
        rel(repo, miss_path),
        len(ref_missing - tool_missing), len(tool_missing - ref_missing), 0,
        "reference paths are absolute; made relative to the GC10 dataset root")

    add("GC10-DET empty-annotation images", 2, inv["xml.images_label_empty"], "(published value)",
        0, 0, 0 if inv["xml.images_label_empty"] == 2 else 1)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(
        description="Compare dsqc scan outputs with the manuscript audit lists (13 items).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="exit status: 0 if every item has 0 differing rows, 1 otherwise")
    ap.add_argument("--apspc", required=True, metavar="DIR", help="dsqc scan output of the APSPC dataset")
    ap.add_argument("--gc10", required=True, metavar="DIR", help="dsqc scan output of the GC10-DET dataset")
    ap.add_argument("--neu", required=True, metavar="DIR", help="dsqc scan output of the NEU-DET dataset")
    ap.add_argument("--repo", required=True, metavar="DIR",
                    help="root of the repository that contains the manuscript's audit/ directory")
    ap.add_argument("--ref", metavar="DIR", default=None,
                    help="directory with apspc_group_types.csv (regenerated by classify_groups.py); "
                         "without it the A/B/C item is not compared")
    ap.add_argument("--gc10-dataset", metavar="DIR", default=None,
                    help="root of the GC10-DET dataset (used to resolve the absolute paths in the "
                         "missing-XML list); defaults to the parent of the scan output")
    ap.add_argument("--apspc-dataset", metavar="DIR", default=None,
                    help="root of the APSPC dataset holding the released archives; enables the "
                         "archive SHA-256 item")
    ap.add_argument("--out-csv", metavar="FILE", default=None, help="write the result table as CSV")
    ap.add_argument("--out-json", metavar="FILE", default=None, help="write the result table as JSON")
    a = ap.parse_args()
    a.gc10_dataset = a.gc10_dataset or a.gc10

    for d, what in ((a.apspc, "--apspc"), (a.gc10, "--gc10"), (a.neu, "--neu"), (a.repo, "--repo")):
        need(d, what)
    for d in (a.apspc, a.gc10, a.neu):
        need(os.path.join(d, "summary.json"), "scan output (summary.json)")

    for fn in (check_apspc, check_neu, check_gc10):
        try:
            fn(a, csv_rows)
        except Exception as e:
            add(fn.__name__, "", f"ERROR {type(e).__name__}: {e}", "", 0, 0, 1, "checker failed")

    header = ["#", "item", "reference", "dsqc", "only_ref", "only_tool", "field_diff", "diff_rows"]
    print(f"{header[0]:>3}  {header[1]:<44}{header[2]:<30}{header[3]:<30}"
          f"{header[4]:>9}{header[5]:>10}{header[6]:>11}{header[7]:>10}")
    for i, r in enumerate(RESULTS, 1):
        print(f"{i:>3}  {r['item'][:44]:<44}{str(r['reference'])[:29]:<30}{str(r['tool'])[:29]:<30}"
              f"{r['only_ref']:>9}{r['only_tool']:>10}{r['field_diff']:>11}{r['diff_rows']:>10}")
    total = sum(r["diff_rows"] for r in RESULTS)
    print(f"\n=== {len(RESULTS)} items checked, total differing rows = {total} ===")

    if a.out_csv:
        with open(a.out_csv, "w", newline="", encoding="utf-8") as f:
            fields = ["item", "reference", "tool", "ref_file", "only_ref", "only_tool",
                      "field_diff", "diff_rows", "note"]
            wtr = csv.DictWriter(f, fieldnames=fields)
            wtr.writeheader()
            wtr.writerows(RESULTS)
        print(f"written: {a.out_csv}")
    if a.out_json:
        with open(a.out_json, "w", encoding="utf-8") as f:
            json.dump(RESULTS, f, ensure_ascii=False, indent=1)
        print(f"written: {a.out_json}")
    return 0 if total == 0 and len(RESULTS) == 13 else 1


if __name__ == "__main__":
    sys.exit(main())
