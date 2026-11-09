"""Pascal VOC XML: one XML per image, matched to images by file stem."""
import os
import xml.etree.ElementTree as ET
from collections import defaultdict

from ..model import BOXES, EMPTY, ERROR, IMAGE_CLASS, MISSING, OK, UNMAPPED, Box, LabelSet
from .common import apply_alias, posix, stem_of, walk_files


def parse_voc(path, aliases):
    """Returns (boxes, declared_size, folder_tag, issues). Raises on unreadable XML."""
    root = ET.parse(path).getroot()
    issues, boxes = [], []
    size, declared = root.find("size"), None
    if size is not None:
        try:
            declared = (int(float(size.findtext("width"))), int(float(size.findtext("height"))))
        except (TypeError, ValueError):
            issues.append(("bad_size_field", "size/width or height not numeric"))
    folder_tag = root.findtext("folder")
    folder_tag = folder_tag.strip() if folder_tag is not None else None
    for k, obj in enumerate(root.iter("object")):
        name = obj.findtext("name")
        name = "" if name is None else name.strip()
        bb = obj.find("bndbox")
        if bb is None:
            issues.append(("object_without_bndbox", f"object {k} '{name}'"))
            continue
        try:
            x1, y1, x2, y2 = (float(bb.findtext(t)) for t in ("xmin", "ymin", "xmax", "ymax"))
        except (TypeError, ValueError):
            issues.append(("bad_bndbox", f"object {k} '{name}'"))
            continue
        boxes.append(Box(apply_alias(name, aliases), x1, y1, x2, y2, raw_cls=name))
    return boxes, declared, folder_tag, issues


def index_xml(root, dirs):
    """stem -> list of xml rel paths (several = conflict, reported as an issue)."""
    idx = defaultdict(list)
    files, seen = [], set()
    for d in dirs:
        base = os.path.join(root, d) if d not in ("", ".") else root
        for rel in walk_files(base, (".xml",)):
            full = posix(os.path.relpath(os.path.join(base, rel), root))
            if full in seen:
                continue
            seen.add(full)
            files.append(full)
            idx[stem_of(full)].append(full)
    return idx, sorted(set(files))


def attach_voc(ds, name, dirs, aliases, folder_tag_source=None, folder_tag_map=None):
    """Attach a VOC box source (and optionally the XML <folder> tag as an image-class source)."""
    idx, files = index_xml(ds.root, dirs)
    ds.sources[name] = BOXES
    ds.annotation_files[name] = files
    if folder_tag_source:
        ds.sources[folder_tag_source] = IMAGE_CLASS
    used = set()
    for im in ds.images:
        cands = idx.get(im.stem, [])
        if not cands:
            im.labels[name] = LabelSet(name, BOXES, MISSING)
            if folder_tag_source:
                im.labels[folder_tag_source] = LabelSet(folder_tag_source, IMAGE_CLASS, MISSING)
            continue
        rel = cands[0]
        used.add(rel)
        ls = LabelSet(name, BOXES, OK, file=rel)
        if len(cands) > 1:
            ls.issues.append(("several_annotation_files", " | ".join(cands)))
        tag = None
        try:
            boxes, declared, tag, issues = parse_voc(os.path.join(ds.root, rel), aliases)
            ls.boxes, ls.declared_size = boxes, declared
            ls.issues.extend(issues)
            ls.status = OK if boxes else EMPTY
        except Exception as e:  # unreadable XML
            ls.status = ERROR
            ls.issues.append(("parse_error", f"{type(e).__name__}: {e}"))
        im.labels[name] = ls
        if folder_tag_source:
            if tag is None:
                im.labels[folder_tag_source] = LabelSet(folder_tag_source, IMAGE_CLASS, MISSING, file=rel)
            else:
                if folder_tag_map and tag not in folder_tag_map:
                    im.labels[folder_tag_source] = LabelSet(folder_tag_source, IMAGE_CLASS, UNMAPPED, file=rel,
                                                            raw_image_class=tag)
                else:
                    im.labels[folder_tag_source] = LabelSet(folder_tag_source, IMAGE_CLASS, OK, file=rel,
                                                            image_class=(folder_tag_map or {}).get(tag, tag),
                                                            raw_image_class=tag)
    ds.orphans[name] = [f for f in files if f not in used]
