"""Build small synthetic datasets with known, planted problems (used by test_synthetic.py)."""
import json
import os
import shutil

import numpy as np
from PIL import Image


def _img(path, seed, size=(64, 48), fmt=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(arr).save(path, format=fmt)
    return arr


def _copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy(src, dst)


def _txt(path, s):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(s)


def make_yolo(root):
    """YOLO layout with planted problems. Returns the expected facts."""
    I, L = os.path.join(root, "images"), os.path.join(root, "labels")
    # cross-split pixel duplicate (file copy) with different class sets -> type A, cross_split_leak
    _img(f"{I}/train/a.png", 1)
    _copy(f"{I}/train/a.png", f"{I}/val/a_copy.png")
    _txt(f"{L}/train/a.txt", "0 0.5 0.5 0.2 0.2\n")
    _txt(f"{L}/val/a_copy.txt", "1 0.5 0.5 0.2 0.2\n")
    # same pixels, different bytes (PNG vs BMP), same split, slightly shifted box -> C-type, reencoding
    arr = _img(f"{I}/train/b.png", 2)
    Image.fromarray(arr).save(f"{I}/train/b_bmp.bmp")
    _txt(f"{L}/train/b.txt", "0 0.5 0.5 0.5 0.5\n")
    _txt(f"{L}/train/b_bmp.txt", "0 0.52 0.5 0.5 0.5\n")
    # horizontal flip in the other split -> check 04a cross split
    arr = _img(f"{I}/train/c.png", 3)
    os.makedirs(f"{I}/val", exist_ok=True)
    Image.fromarray(arr[:, ::-1]).save(f"{I}/val/c_flip.png")
    _txt(f"{L}/train/c.txt", "0 0.5 0.5 0.1 0.1\n")
    _txt(f"{L}/val/c_flip.txt", "0 0.5 0.5 0.1 0.1\n")
    # box problems: out of bounds, tiny, full image, duplicate boxes, malformed line, class id out of range
    _img(f"{I}/train/d.png", 4)
    _txt(f"{L}/train/d.txt", "0 0.95 0.5 0.3 0.2\n"      # out of bounds
                              "0 0.3 0.3 0.01 0.01\n"     # tiny (0.64 x 0.48 px)
                              "1 0.5 0.5 1.0 1.0\n"       # full image
                              "2 0.2 0.7 0.2 0.2\n2 0.2 0.7 0.2 0.2\n"  # duplicate boxes
                              "bad line\n"
                              "9 0.5 0.5 0.1 0.1\n")      # id 9 with 3 names
    _img(f"{I}/train/e.png", 5)                          # no label file
    _img(f"{I}/val/f.png", 6)
    _txt(f"{L}/val/f.txt", "")                            # empty label file
    _txt(f"{L}/train/ghost.txt", "0 0.5 0.5 0.1 0.1\n")   # orphan label
    _txt(os.path.join(root, "data.yaml"), "names: [scratch, hole, stain]\n")
    return {"images": 9}


def _xml(path, folder, objs, size=(64, 48)):
    o = "".join(f"<object><name>{n}</name><bndbox><xmin>{a}</xmin><ymin>{b}</ymin><xmax>{c}</xmax>"
                f"<ymax>{d}</ymax></bndbox></object>" for n, a, b, c, d in objs)
    _txt(path, f"<annotation><folder>{folder}</folder><size><width>{size[0]}</width><height>{size[1]}</height>"
               f"<depth>3</depth></size>{o}</annotation>")


def make_gc10_like(root):
    """Class folders + flat XML dir, GC10-DET style."""
    _img(f"{root}/1/g1.jpg", 11)
    _xml(f"{root}/lable/g1.xml", "1", [("1_hole", 1, 1, 20, 20)])                 # consistent
    _img(f"{root}/1/g2.jpg", 12)
    _xml(f"{root}/lable/g2.xml", "1", [("2_spot", 1, 1, 20, 20)])                 # folder class absent
    _img(f"{root}/2/g3.jpg", 13)
    _xml(f"{root}/lable/g3.xml", "2", [])                                          # empty xml -> counts as absent
    _img(f"{root}/2/g4.jpg", 14)                                                   # no xml
    _img(f"{root}/1/g5.jpg", 15)
    _copy(f"{root}/1/g5.jpg", f"{root}/2/g5.jpg")                            # same file in two folders
    _xml(f"{root}/lable/g5.xml", "1", [("2_spotd", 2, 2, 30, 30), ("1_hole", 1, 1, 10, 10)])  # alias spelling
    _img(f"{root}/2/g6.jpg", 16)
    _xml(f"{root}/lable/g6.xml", "1", [("2_spot", 1, 1, 20, 20)], size=(100, 100))  # tag 1 vs folder 2; size mismatch
    _img(f"{root}/1/g7.jpg", 17)
    _xml(f"{root}/lable/g7.xml", "all", [("1_hole", 1, 1, 20, 20)])               # tag 'all' -> unmapped
    _xml(f"{root}/lable/orphan.xml", "1", [("1_hole", 1, 1, 5, 5)])
    spec = {
        "name": "gc10-like", "root": root, "images": ["*/*.jpg"], "split": "none", "primary": "xml",
        "class_names": ["1_hole", "2_spot"], "class_aliases": {"2_spotd": "2_spot"},
        "sources": [
            {"name": "xml", "type": "voc", "dirs": ["lable"], "folder_tag_source": "xml_folder",
             "folder_tag_map": {"1": "1_hole", "2": "2_spot"}},
            {"name": "folder", "type": "folder_class", "level": -1, "map": {"1": "1_hole", "2": "2_spot"}},
        ],
    }
    return spec


def make_coco(root):
    _img(f"{root}/images/x1.png", 21)
    _img(f"{root}/images/x2.png", 22)
    d = {"images": [{"id": 1, "file_name": "x1.png", "width": 64, "height": 48},
                    {"id": 2, "file_name": "x2.png", "width": 64, "height": 48},
                    {"id": 3, "file_name": "x3.png", "width": 64, "height": 48}],
         "annotations": [{"id": 10, "image_id": 1, "category_id": 1, "bbox": [5, 5, 10, 10]},
                         {"id": 11, "image_id": 1, "category_id": 1, "bbox": [60, 40, 10, 10]},
                         {"id": 12, "image_id": 7, "category_id": 1, "bbox": [1, 1, 2, 2]}],
         "categories": [{"id": 1, "name": "dent"}]}
    _txt(f"{root}/annotations/instances.json", json.dumps(d))
