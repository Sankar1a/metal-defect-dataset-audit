"""Check 00 - inventory: count the dataset at every level, each with its definition.

Motivation (GC10-DET in the accompanying manuscript): GC10-DET can be read as 2312 / 2306 / 2300 / 2294 / 2292 images depending on
what is counted. The report lists each level so any reported size can be traced to a definition.
"""
from collections import Counter, defaultdict

from ..model import EMPTY, ERROR, MISSING, OK, UNMAPPED, IMAGE_CLASS, Finding
from . import CheckResult

NAME = "c00_inventory"
TITLE = "Inventory (counts at every level)"


def run(ds, cfg, ctx):
    res = CheckResult(NAME, TITLE)
    rows = []

    def add(layer, n, definition):
        rows.append({"layer": layer, "count": n, "definition": definition})
        res.stats[layer] = n

    ims = ds.images
    add("image_files", len(ims), "image files matched on disk (each folder entry counted)")
    add("image_decode_errors", sum(1 for im in ims if im.error), "image files PIL could not decode")
    stems = defaultdict(list)
    for im in ims:
        stems[im.stem].append(im)
    multi = {s: v for s, v in stems.items() if len(v) > 1}
    add("distinct_stems", len(stems), "distinct file names without extension")
    add("stems_in_several_files", len(multi), "file names that occur more than once (e.g. same file in two folders)")
    add("distinct_byte_sha256", len({im.byte_sha for im in ims if im.byte_sha}), "distinct file contents")
    add("distinct_pixel_sha256", len({im.pixel_sha for im in ims if im.pixel_sha}),
        "distinct decoded images (RGB pixels)")
    if any(im.canon_sha for im in ims):
        add("distinct_after_rot_flip", len({im.canon_sha for im in ims if im.canon_sha}),
            "distinct images when the 8 rotations/flips are treated as the same image")
    for src, kind in ds.sources.items():
        st = Counter(im.labels[src].status if src in im.labels else MISSING for im in ims)
        if src in ds.annotation_files:
            add(f"{src}.annotation_files", len(ds.annotation_files[src]), f"annotation files of source '{src}' on disk")
        add(f"{src}.images_with_label", st[OK] + st[EMPTY] + st[ERROR] + st[UNMAPPED],
            f"image files that have a '{src}' label (file or folder)")
        add(f"{src}.images_label_nonempty", st[OK], f"image files whose '{src}' label has at least one object/class")
        add(f"{src}.images_label_empty", st[EMPTY], f"image files whose '{src}' annotation has no object")
        add(f"{src}.images_label_missing", st[MISSING], f"image files without a '{src}' label")
        add(f"{src}.images_label_unreadable", st[ERROR], f"'{src}' annotation could not be parsed")
        if kind == IMAGE_CLASS and st[UNMAPPED]:
            vals = Counter(im.labels[src].raw_image_class for im in ims
                           if src in im.labels and im.labels[src].status == UNMAPPED)
            add(f"{src}.images_label_unmapped", st[UNMAPPED],
                f"'{src}' value not in the class mapping (not compared): "
                + ", ".join(f"{v} ({n})" for v, n in vals.most_common()))
        if src in ds.orphans:
            add(f"{src}.orphan_annotations", len(ds.orphans[src]), f"'{src}' annotation files with no image")
    splits = Counter(im.split for im in ims)
    if ds.split_mode != "none":
        for sp, n in sorted(splits.items(), key=lambda x: str(x[0])):
            add(f"split.{sp}", n, f"image files in split '{sp}' ({ds.split_mode})")
    res.tables["inventory"] = (["layer", "count", "definition"], rows)

    for im in ims:
        if im.error:
            res.findings.append(Finding(NAME, "decode_error", "high", [im.rel], im.error))
    for s, v in sorted(multi.items()):
        res.findings.append(Finding(NAME, "same_stem_several_files", "info", [im.rel for im in v],
                                    f"file name '{s}' occurs {len(v)} times",
                                    {"stem": s, "n": len(v),
                                     "same_bytes": len({im.byte_sha for im in v}) == 1}))
    for src, files in ds.orphans.items():
        for f in files:
            res.findings.append(Finding(NAME, "orphan_annotation", "medium", [], f"'{src}': {f}",
                                        {"source": src, "file": f}))
    res.notes.extend(ds.notes)
    return res
