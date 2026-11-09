import fnmatch
import os

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def posix(rel):
    return rel.replace(os.sep, "/")


def walk_files(root, exts=None):
    """All files under root (relative posix paths, sorted), skipping hidden directories."""
    out = []
    for dp, dns, fns in os.walk(root):
        dns[:] = sorted(d for d in dns if not d.startswith("."))
        for fn in fns:
            if fn.startswith("."):
                continue
            if exts is None or fn.lower().endswith(exts):
                out.append(posix(os.path.relpath(os.path.join(dp, fn), root)))
    return sorted(out)


def match_globs(rels, globs):
    if not globs:
        return rels
    return [r for r in rels if any(fnmatch.fnmatchcase(r, g) for g in globs)]


def stem_of(rel):
    return os.path.splitext(os.path.basename(rel))[0]


def apply_alias(name, aliases):
    return aliases.get(name, name)
