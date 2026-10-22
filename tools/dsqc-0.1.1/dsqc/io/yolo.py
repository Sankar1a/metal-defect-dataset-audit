"""YOLO txt: images/<split?>/x.jpg  <->  labels/<split?>/x.txt (one line: cls cx cy w h)."""
import glob
import os

from ..model import BOXES, EMPTY, ERROR, MISSING, OK, Box, LabelSet
from .common import apply_alias, posix, walk_files


def read_names_from_yaml(root):
    """Class names from a data YAML next to the dataset (first one found), else None."""
    try:
        import yaml
    except ImportError:
        return None
    for p in sorted(glob.glob(os.path.join(root, "*.yaml")) + glob.glob(os.path.join(root, "*.yml"))):
        try:
            with open(p, encoding="utf-8") as f:
                d = yaml.safe_load(f) or {}
        except Exception:
            continue
        names = d.get("names")
        if isinstance(names, dict):
            return [str(names[k]) for k in sorted(names, key=lambda x: int(x))]
        if isinstance(names, list):
            return [str(n) for n in names]
    return None


def label_rel_for(img_rel, images_dir="images", labels_dir="labels"):
    parts = img_rel.split("/")
    if images_dir in parts:
        i = len(parts) - 1 - parts[::-1].index(images_dir)
        parts[i] = labels_dir
    else:
        parts = [labels_dir] + parts
    return "/".join(parts[:-1] + [os.path.splitext(parts[-1])[0] + ".txt"])


def parse_yolo(path, names, aliases):
    boxes, issues = [], []
    with open(path, encoding="utf-8", errors="replace") as f:
        for n, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) != 5:
                issues.append(("malformed_line", f"line {n}: '{s[:60]}'"))
                continue
            try:
                cid = int(float(parts[0]))
                cx, cy, w, h = (float(x) for x in parts[1:])
            except ValueError:
                issues.append(("malformed_line", f"line {n}: '{s[:60]}'"))
                continue
            raw = str(cid)
            if names is not None:
                if 0 <= cid < len(names):
                    raw = names[cid]
                else:
                    issues.append(("class_id_out_of_range", f"line {n}: id {cid}"))
            boxes.append(Box(apply_alias(raw, aliases), cx, cy, w, h, raw_cls=raw, normalized=True))
    return boxes, issues


def attach_yolo(ds, name, aliases, names=None, labels_dir="labels", images_dir="images"):
    ds.sources[name] = BOXES
    label_files = walk_files(os.path.join(ds.root, labels_dir), (".txt",)) if os.path.isdir(
        os.path.join(ds.root, labels_dir)) else []
    label_files = [posix(os.path.join(labels_dir, r)) for r in label_files]
    ds.annotation_files[name] = label_files
    used = set()
    for im in ds.images:
        rel = label_rel_for(im.rel, images_dir, labels_dir)
        p = os.path.join(ds.root, rel)
        if not os.path.isfile(p):
            im.labels[name] = LabelSet(name, BOXES, MISSING)
            continue
        used.add(rel)
        ls = LabelSet(name, BOXES, OK, file=rel)
        try:
            ls.boxes, issues = parse_yolo(p, names, aliases)
            ls.issues.extend(issues)
            ls.status = OK if ls.boxes else EMPTY
        except Exception as e:
            ls.status = ERROR
            ls.issues.append(("parse_error", f"{type(e).__name__}: {e}"))
        im.labels[name] = ls
    ds.orphans[name] = [f for f in label_files if f not in used and not f.endswith("classes.txt")]
