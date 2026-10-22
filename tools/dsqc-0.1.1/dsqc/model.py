"""Unified in-memory representation shared by all readers and checks.

Every reader turns its format into a Dataset of ImageRecords. Each image can carry several
label sources (e.g. VOC XML boxes and the class implied by its folder). Boxes are stored in
absolute pixel xyxy; YOLO boxes are kept normalised until image sizes are known
(Dataset.finalize).
"""
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

BOXES = "boxes"
IMAGE_CLASS = "image_class"

OK, EMPTY, MISSING, ERROR = "ok", "empty", "missing", "error"
UNMAPPED = "unmapped"   # image-class value not in the given class mapping (e.g. "all", "no class hint")


@dataclass
class Box:
    cls: str                 # canonical class name (after aliases)
    x1: float
    y1: float
    x2: float
    y2: float
    raw_cls: str = ""        # class string as written in the annotation file
    normalized: bool = False  # True while still YOLO cx,cy,w,h in [0,1] (x1..y2 hold cx,cy,w,h)

    def key(self):
        return (self.cls, self.x1, self.y1, self.x2, self.y2)

    def xyxy(self):
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass
class LabelSet:
    source: str
    kind: str                       # BOXES or IMAGE_CLASS
    status: str                     # OK / EMPTY / MISSING / ERROR / UNMAPPED
    boxes: List[Box] = field(default_factory=list)
    image_class: Optional[str] = None
    raw_image_class: Optional[str] = None
    file: Optional[str] = None      # annotation file, relative to dataset root
    declared_size: Optional[Tuple[int, int]] = None   # (w, h) written in the annotation
    issues: List[Tuple[str, str]] = field(default_factory=list)  # (code, detail) found while parsing

    def classes(self):
        return {b.cls for b in self.boxes}


@dataclass
class ImageRecord:
    rel: str                        # path relative to dataset root, '/' separators
    path: str                       # absolute path
    stem: str
    split: Optional[str] = None
    labels: Dict[str, LabelSet] = field(default_factory=dict)
    size_bytes: int = 0
    byte_sha: Optional[str] = None
    pixel_sha: Optional[str] = None  # sha256 of PIL convert('RGB').tobytes(), as in the accompanying manuscript
    canon_sha: Optional[str] = None  # min sha over the 8 rotations/flips (shape included)
    width: Optional[int] = None
    height: Optional[int] = None
    mode: Optional[str] = None
    fmt: Optional[str] = None
    error: Optional[str] = None


@dataclass
class Dataset:
    name: str
    root: str
    images: List[ImageRecord]
    sources: Dict[str, str]                       # source name -> kind
    primary: Optional[str] = None                 # main box source
    class_names: Optional[List[str]] = None
    class_aliases: Dict[str, str] = field(default_factory=dict)
    annotation_files: Dict[str, List[str]] = field(default_factory=dict)  # source -> annotation files (rel)
    orphans: Dict[str, List[str]] = field(default_factory=dict)          # source -> annotation files w/o image
    contingency_only: set = field(default_factory=set)   # class sources compared by cross-tab only
    split_mode: str = "none"
    layout: str = ""
    notes: List[str] = field(default_factory=list)

    def box_sources(self):
        return [s for s, k in self.sources.items() if k == BOXES]

    def class_sources(self):
        return [s for s, k in self.sources.items() if k == IMAGE_CLASS]

    def by_rel(self):
        return {im.rel: im for im in self.images}

    def finalize(self):
        """Convert normalised YOLO boxes to absolute pixels once image sizes are known."""
        for im in self.images:
            for ls in im.labels.values():
                for b in ls.boxes:
                    if b.normalized:
                        if im.width is None or im.height is None:
                            continue
                        cx, cy, w, h = b.x1, b.y1, b.x2, b.y2
                        W, H = im.width, im.height
                        b.x1, b.y1 = (cx - w / 2) * W, (cy - h / 2) * H
                        b.x2, b.y2 = (cx + w / 2) * W, (cy + h / 2) * H
                        b.normalized = False


SEVERITIES = ("high", "medium", "low", "info")


@dataclass
class Finding:
    check: str
    code: str
    severity: str
    images: List[str]
    message: str
    data: dict = field(default_factory=dict)

    @property
    def id(self):
        h = hashlib.sha1(("|".join([self.check, self.code] + sorted(self.images))).encode("utf-8"))
        return f"{self.check}-{h.hexdigest()[:10]}"
