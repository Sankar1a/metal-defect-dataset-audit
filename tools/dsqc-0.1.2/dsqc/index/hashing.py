"""Per-image hashing, run in parallel and cached.

byte_sha  : sha256 of the file bytes
pixel_sha : sha256 of PIL Image.convert('RGB').tobytes()  (same definition as the accompanying manuscript; no EXIF rotation,
            no shape in the hash)
canon_sha : min over the 8 rotations/flips of sha256(shape + RGB bytes); equal canon_sha with
            different pixel_sha = same pixels after a rotation or flip (check 04a)
"""
import hashlib
import io
import json
import os
import sqlite3
from multiprocessing import Pool

import numpy as np
from PIL import Image

CACHE_VERSION = 1


def _dihedral_min_sha(arr):
    best = None
    for k in range(4):
        r = np.rot90(arr, k)
        for v in (r, r[:, ::-1]):
            v = np.ascontiguousarray(v)
            h = hashlib.sha256(str(v.shape).encode("ascii"))
            h.update(v.tobytes())
            d = h.hexdigest()
            if best is None or d < best:
                best = d
    return best


def hash_one(args):
    path, want_canon = args
    out = {"byte_sha": None, "pixel_sha": None, "canon_sha": None, "width": None, "height": None,
           "mode": None, "fmt": None, "error": None}
    try:
        with open(path, "rb") as f:
            data = f.read()
        out["byte_sha"] = hashlib.sha256(data).hexdigest()
        with Image.open(io.BytesIO(data)) as im:
            out["mode"], out["fmt"] = im.mode, im.format
            out["width"], out["height"] = im.size
            rgb = im.convert("RGB")
            out["pixel_sha"] = hashlib.sha256(rgb.tobytes()).hexdigest()
            if want_canon:
                out["canon_sha"] = _dihedral_min_sha(np.asarray(rgb))
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


class HashCache:
    """sqlite cache keyed by (absolute path, size, mtime_ns); safe to delete at any time."""

    def __init__(self, cache_dir):
        self.db = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            self.db = sqlite3.connect(os.path.join(cache_dir, "hashes.sqlite"))
            self.db.execute("CREATE TABLE IF NOT EXISTS h (path TEXT, size INTEGER, mtime INTEGER, "
                            "ver INTEGER, canon INTEGER, val TEXT, PRIMARY KEY(path, size, mtime, ver, canon))")

    def get(self, path, st, canon):
        if not self.db:
            return None
        row = self.db.execute("SELECT val FROM h WHERE path=? AND size=? AND mtime=? AND ver=? AND canon=?",
                              (path, st.st_size, st.st_mtime_ns, CACHE_VERSION, int(canon))).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, path, st, canon, val):
        if self.db and not val.get("error"):
            self.db.execute("INSERT OR REPLACE INTO h VALUES (?,?,?,?,?,?)",
                            (path, st.st_size, st.st_mtime_ns, CACHE_VERSION, int(canon), json.dumps(val)))

    def close(self):
        if self.db:
            self.db.commit()
            self.db.close()


def hash_dataset(ds, workers=8, want_canon=True, cache_dir=None, progress=print):
    cache = HashCache(cache_dir)
    todo, stats = [], {}
    for im in ds.images:
        st = os.stat(im.path)
        stats[im.path] = st
        hit = cache.get(im.path, st, want_canon)
        if hit is not None:
            _assign(im, hit)
        else:
            todo.append(im)
    progress(f"hashing {len(todo)} images ({len(ds.images) - len(todo)} from cache), workers={workers}")
    jobs = [(im.path, want_canon) for im in todo]
    if workers > 1 and len(jobs) > 1:
        with Pool(workers) as pool:
            results = pool.map(hash_one, jobs, chunksize=max(1, len(jobs) // (workers * 8)))
    else:
        results = [hash_one(j) for j in jobs]
    for im, r in zip(todo, results):
        _assign(im, r)
        cache.put(im.path, stats[im.path], want_canon, r)
    cache.close()


def _assign(im, r):
    for k in ("byte_sha", "pixel_sha", "canon_sha", "width", "height", "mode", "fmt", "error"):
        setattr(im, k, r.get(k))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
