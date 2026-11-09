"""Side-by-side thumbnails of the images of a finding, with the boxes of one source drawn."""
import os

from PIL import Image, ImageDraw

COLORS = [(230, 75, 53), (60, 84, 136), (0, 160, 135), (243, 155, 127), (132, 145, 180),
          (145, 209, 194), (220, 0, 0), (126, 97, 72), (176, 156, 133), (77, 187, 213)]


def panel(ds, rels, source, out_path, height=220, max_images=4):
    by = ds.by_rel()
    tiles = []
    classes = sorted({b.cls for r in rels for b in (by[r].labels.get(source).boxes if by.get(r) and by[r].labels.get(source) else [])})
    color = {c: COLORS[i % len(COLORS)] for i, c in enumerate(classes)}
    for r in rels[:max_images]:
        im = by.get(r)
        if im is None:
            continue
        try:
            with Image.open(im.path) as src:
                img = src.convert("RGB")
        except Exception:
            continue
        s = height / img.height
        img = img.resize((max(1, int(img.width * s)), height))
        d = ImageDraw.Draw(img)
        ls = im.labels.get(source)
        for b in (ls.boxes if ls else []):
            if b.normalized:
                continue
            xy = [b.x1 * s, b.y1 * s, b.x2 * s, b.y2 * s]
            d.rectangle(xy, outline=color.get(b.cls, (255, 0, 0)), width=2)
            d.text((xy[0] + 2, xy[1] + 1), b.cls, fill=color.get(b.cls, (255, 0, 0)))
        cap = Image.new("RGB", (img.width, 16), (255, 255, 255))
        ImageDraw.Draw(cap).text((2, 2), os.path.basename(r)[:40], fill=(0, 0, 0))
        tile = Image.new("RGB", (img.width, height + 16), (255, 255, 255))
        tile.paste(img, (0, 0))
        tile.paste(cap, (0, height))
        tiles.append(tile)
    if not tiles:
        return False
    W = sum(t.width for t in tiles) + 6 * (len(tiles) - 1)
    out = Image.new("RGB", (W, height + 16), (255, 255, 255))
    x = 0
    for t in tiles:
        out.paste(t, (x, 0))
        x += t.width + 6
    out.save(out_path, quality=85)
    return True
