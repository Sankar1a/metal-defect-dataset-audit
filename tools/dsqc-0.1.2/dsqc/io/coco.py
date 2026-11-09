"""COCO json: images[].file_name relative to an image directory, bbox = [x, y, w, h]."""
import json
import os
from collections import defaultdict

from ..model import BOXES, EMPTY, MISSING, OK, Box, LabelSet
from .common import apply_alias, posix


def attach_coco(ds, name, json_rel, image_prefix, aliases):
    """image_prefix: directory of the COCO file_name entries, relative to dataset root."""
    with open(os.path.join(ds.root, json_rel), encoding="utf-8") as f:
        d = json.load(f)
    ds.sources[name] = BOXES
    ds.annotation_files[name] = [json_rel]
    cats = {c["id"]: str(c.get("name", c["id"])) for c in d.get("categories", [])}
    img_by_id = {}
    for e in d.get("images", []):
        rel = posix(os.path.normpath(os.path.join(image_prefix, e["file_name"])))
        img_by_id[e["id"]] = (rel, e.get("width"), e.get("height"))
    anns = defaultdict(list)
    bad = []
    for a in d.get("annotations", []):
        if a.get("image_id") not in img_by_id:
            bad.append(str(a.get("id")))
            continue
        anns[a["image_id"]].append(a)
    by_rel = {v[0]: (k, v[1], v[2]) for k, v in img_by_id.items()}
    for im in ds.images:
        if im.rel not in by_rel:
            im.labels[name] = LabelSet(name, BOXES, MISSING)
            continue
        iid, w, h = by_rel[im.rel]
        ls = LabelSet(name, BOXES, OK, file=json_rel,
                      declared_size=(int(w), int(h)) if w and h else None)
        for a in anns.get(iid, []):
            raw = cats.get(a.get("category_id"), str(a.get("category_id")))
            bb = a.get("bbox") or []
            if len(bb) != 4:
                ls.issues.append(("bad_bbox", f"annotation {a.get('id')}"))
                continue
            x, y, w_, h_ = (float(v) for v in bb)
            ls.boxes.append(Box(apply_alias(raw, aliases), x, y, x + w_, y + h_, raw_cls=raw))
        ls.status = OK if ls.boxes else EMPTY
        im.labels[name] = ls
    on_disk = {im.rel for im in ds.images}
    ds.orphans[name] = sorted(f"image entry without file: {r}" for r in by_rel if r not in on_disk) + \
        [f"annotation {i} -> unknown image_id" for i in bad]
