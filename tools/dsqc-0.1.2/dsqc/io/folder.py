"""Folder-as-class: the class implied by the directory an image is stored in (e.g. GC10-DET)."""
from ..model import IMAGE_CLASS, MISSING, OK, UNMAPPED, LabelSet


def attach_folder_class(ds, name, level=-1, mapping=None):
    """level: index into the directory components of the image's relative path (-1 = parent)."""
    ds.sources[name] = IMAGE_CLASS
    for im in ds.images:
        dirs = im.rel.split("/")[:-1]
        try:
            raw = dirs[level]
        except IndexError:
            im.labels[name] = LabelSet(name, IMAGE_CLASS, MISSING)
            continue
        if mapping and raw not in mapping:
            im.labels[name] = LabelSet(name, IMAGE_CLASS, UNMAPPED, raw_image_class=raw)
        else:
            im.labels[name] = LabelSet(name, IMAGE_CLASS, OK, image_class=(mapping or {}).get(raw, raw),
                                       raw_image_class=raw)
