"""Build a Dataset from a spec YAML (explicit layout) or by auto-detecting a standard layout.

Spec YAML (all paths relative to `root`; see specs/*.yaml for real examples):

    name: GC10-DET (raw)
    root: /path/to/gc10_raw
    images: ["*/*.jpg"]              # globs over image files; default: every image file
    exclude: []                      # globs to drop
    split: {type: component, index: 1}  # or {type: lists, lists: {train: a.txt, val: b.txt}} or none
    class_names: [...]               # optional; enables the unknown-class check
    class_aliases: {10_yaozhed: 10_yaozhe}
    primary: xml                     # main box source
    sources:
      - {name: xml, type: voc, dirs: [lable], folder_tag_source: xml_folder, folder_tag_map: {...}}
      - {name: folder, type: folder_class, level: -1, map: {"1": 1_chongkong}}
        # findings: false / folder_tag_findings: false -> cross-tab only (use when no class mapping is known)
      - {name: yolo, type: yolo, labels_dir: labels, images_dir: images}
      - {name: coco, type: coco, json: annotations/train.json, image_prefix: images/train}
"""
import json
import os

from ..model import Dataset, ImageRecord
from .coco import attach_coco
from .common import IMG_EXTS, match_globs, posix, stem_of, walk_files
from .folder import attach_folder_class
from .voc import attach_voc
from .yolo import attach_yolo, read_names_from_yaml


def _make_images(root, globs=None, exclude=None):
    rels = match_globs(walk_files(root, IMG_EXTS), globs)
    if exclude:
        drop = set(match_globs(rels, exclude))
        rels = [r for r in rels if r not in drop]
    out = []
    for r in rels:
        p = os.path.join(root, r)
        out.append(ImageRecord(rel=r, path=p, stem=stem_of(r), size_bytes=os.path.getsize(p)))
    return out


def _apply_split(ds, split):
    if not split or split == "none" or split.get("type", "none") == "none":
        ds.split_mode = "none"
        return
    t = split["type"]
    if t == "component":
        idx = int(split.get("index", 1))
        for im in ds.images:
            parts = im.rel.split("/")
            im.split = parts[idx] if len(parts) - 1 > idx else None
        ds.split_mode = f"component[{idx}]"
    elif t == "lists":
        by_rel = ds.by_rel()
        by_base = {}
        for im in ds.images:
            by_base.setdefault(os.path.basename(im.rel), []).append(im)
        for name, lst in split["lists"].items():
            p = lst if os.path.isabs(lst) else os.path.join(ds.root, lst)
            with open(p, encoding="utf-8") as f:
                for line in f:
                    s = posix(line.strip())
                    if not s:
                        continue
                    hit = [im for r, im in by_rel.items() if s.endswith("/" + r) or s == r]
                    if not hit:
                        cands = by_base.get(os.path.basename(s), [])
                        hit = cands if len(cands) == 1 else []
                        if len(cands) > 1:
                            ds.notes.append(f"split list {name}: '{s}' matches {len(cands)} images; skipped")
                    if not hit:
                        ds.notes.append(f"split list {name}: '{s}' not found")
                    for im in hit:
                        if im.split not in (None, name):
                            ds.notes.append(f"{im.rel} listed in both {im.split} and {name}")
                        im.split = name
        ds.split_mode = "lists"
    else:
        raise ValueError(f"unknown split type {t}")


def load_spec(spec_path, root_override=None):
    import yaml
    with open(spec_path, encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    root = os.path.abspath(root_override or spec["root"])
    aliases = {str(k): str(v) for k, v in (spec.get("class_aliases") or {}).items()}
    ds = Dataset(name=spec.get("name", os.path.basename(root)), root=root,
                 images=_make_images(root, spec.get("images"), spec.get("exclude")), sources={},
                 class_names=[str(c) for c in spec["class_names"]] if spec.get("class_names") else None,
                 class_aliases=aliases, layout=f"spec:{os.path.basename(spec_path)}")
    for s in spec.get("sources", []):
        t, name = s["type"], s["name"]
        if t == "voc":
            attach_voc(ds, name, s.get("dirs", ["."]), aliases, s.get("folder_tag_source"),
                       {str(k): str(v) for k, v in (s.get("folder_tag_map") or {}).items()})
            if s.get("folder_tag_source") and s.get("folder_tag_findings", True) is False:
                ds.contingency_only.add(s["folder_tag_source"])
        elif t == "folder_class":
            attach_folder_class(ds, name, int(s.get("level", -1)),
                                {str(k): str(v) for k, v in (s.get("map") or {}).items()})
            if s.get("findings", True) is False:
                ds.contingency_only.add(name)
        elif t == "yolo":
            names = ds.class_names or read_names_from_yaml(root)
            attach_yolo(ds, name, aliases, names, s.get("labels_dir", "labels"), s.get("images_dir", "images"))
        elif t == "coco":
            attach_coco(ds, name, s["json"], s.get("image_prefix", ""), aliases)
        else:
            raise ValueError(f"unknown source type {t}")
    ds.primary = spec.get("primary") or (ds.box_sources()[0] if ds.box_sources() else None)
    _apply_split(ds, spec.get("split"))
    return ds


def _find_coco(root):
    for d in ("", "annotations"):
        base = os.path.join(root, d)
        if not os.path.isdir(base):
            continue
        for fn in sorted(os.listdir(base)):
            if fn.lower().endswith(".json"):
                p = os.path.join(base, fn)
                try:
                    with open(p, encoding="utf-8") as f:
                        head = json.load(f)
                except Exception:
                    continue
                if isinstance(head, dict) and "images" in head and "annotations" in head:
                    return posix(os.path.relpath(p, root)), head
    return None, None


def load_auto(root, fmt="auto", names=None):
    """Standard layouts: yolo (images/ + labels/), coco (json with images+annotations), voc (xml files)."""
    root = os.path.abspath(root)
    has_yolo = os.path.isdir(os.path.join(root, "images")) and os.path.isdir(os.path.join(root, "labels"))
    coco_rel, coco = (None, None) if fmt not in ("auto", "coco") else _find_coco(root)
    if fmt == "auto":
        fmt = "yolo" if has_yolo else "coco" if coco_rel else "voc"
    ds = Dataset(name=os.path.basename(root.rstrip("/")), root=root, images=[], sources={},
                 class_names=names, layout=f"auto:{fmt}")
    if fmt == "yolo":
        ds.images = _make_images(root, ["images/*"])
        ds.class_names = names or read_names_from_yaml(root)
        attach_yolo(ds, "yolo", {}, ds.class_names)
        if any(im.rel.count("/") >= 2 for im in ds.images):
            _apply_split(ds, {"type": "component", "index": 1})
    elif fmt == "coco":
        if coco_rel is None:
            raise SystemExit("no COCO json (with 'images' and 'annotations') found")
        ds.images = _make_images(root)
        first = coco["images"][0]["file_name"] if coco["images"] else ""
        prefix = ""
        for cand in ("", "images", os.path.dirname(coco_rel), os.path.join("images", os.path.splitext(
                os.path.basename(coco_rel))[0])):
            if os.path.isfile(os.path.join(root, cand, first)):
                prefix = cand
                break
        attach_coco(ds, "coco", coco_rel, prefix, {})
    elif fmt == "voc":
        ds.images = _make_images(root)
        attach_voc(ds, "xml", ["."], {})
    else:
        raise SystemExit(f"unknown format {fmt}")
    ds.primary = ds.box_sources()[0] if ds.box_sources() else None
    return ds
