import csv, os, collections, hashlib
import xml.etree.ElementTree as ET

D = "round18_groups_20261006"
IMG = {"APSPC1": "APSPC/extracted/part1/APSPC1", "APSPC2": "APSPC/extracted/part2/APSPC2"}
ANNOT = "APSPC/extracted/annotations/APSPC-Annotations/Annotations"

meta = {r['filename']: r for r in csv.DictReader(open(os.path.join(D, ".all_hashes.csv")))}
assert len(meta) == 1885

# ---- grouping source: verify9_20260816/fresh_dup_groups.csv (pixel_sha256) ----
# cross-checked identical (membership-wise) with:
#   APSPC/dedup_audit/conflicting_duplicate_groups.csv
#   APSPC/conflict_exclusion/excluded_conflict_groups.csv
#   verify3_20260805/raw_dup_groups.csv (byte & pixel, label_status=conflicting)
groups = []           # (pixel_sha256, [filenames sorted])
key_check = []
for r in csv.DictReader(open("verify9_20260816/fresh_dup_groups.csv")):
    mem = sorted(os.path.basename(p) for p in r['file_paths'].split(';'))
    calc = {meta[f]['pixel_sha256'] for f in mem}
    key_check.append((r['pixel_sha256'], calc, mem))
    assert len(calc) == 1, mem
    groups.append((r['pixel_sha256'], mem))
assert all(k == list(c)[0] for k, c, _ in key_check), "pixel_sha256 recompute != group file key"

# ---- verification numbers ----
ngroups = len(groups)
nfiles = sum(len(m) for _, m in groups)
dist = collections.Counter(len(m) for _, m in groups)
allfiles = [f for _, m in groups for f in m]
assert len(set(allfiles)) == nfiles, "a file appears in more than one group"

# ---- group ids by lexicographic order of pixel_sha256 ----
groups.sort(key=lambda t: t[0])
gid = {h: "G%02d" % (i + 1) for i, (h, _) in enumerate(groups)}

# ---- parse XML ----
def parse_voc(p):
    root = ET.parse(p).getroot()
    out = []
    for obj in root.findall('object'):
        name = (obj.findtext('name') or '').strip()
        bb = obj.find('bndbox')
        def raw(tag):
            t = bb.findtext(tag)
            return t.strip() if t is not None else ''
        out.append((name, raw('xmin'), raw('ymin'), raw('xmax'), raw('ymax')))
    return out

def num(s):
    """raw XML value -> int/float without rounding (fallback: original string)"""
    try:
        f = float(s)
    except ValueError:
        return s
    return int(f) if f.is_integer() else f

groups_rows, boxes_rows = [], []
for h, mem in groups:
    g = gid[h]
    for f in mem:
        stem = os.path.splitext(f)[0]
        part = meta[f]['part']
        xml_name = stem + ".xml"
        xml_path = os.path.join(ANNOT, xml_name)
        objs = parse_voc(xml_path)
        classes = sorted(o[0] for o in objs)
        groups_rows.append({
            'group_id': g, 'pixel_sha256': h, 'n_copies': len(mem), 'filename': f,
            'part': part, 'xml_file': xml_name, 'n_objects': len(objs),
            'classes_sorted': '|'.join(classes),
        })
        for i, (name, xmin, ymin, xmax, ymax) in enumerate(objs):
            boxes_rows.append({
                'group_id': g, 'filename': f, 'obj_idx': i, 'class_name': name,
                'xmin': num(xmin), 'ymin': num(ymin), 'xmax': num(xmax), 'ymax': num(ymax),
                'width': num(xmax) - num(xmin), 'height': num(ymax) - num(ymin),
            })

# ---- png_map ----
pngs = [l.strip() for l in open(os.path.join(D, "step1_png_list.txt")) if l.strip()]
# byte sha256 of each conflict group (all members share one byte hash; verified below)
byte_of_group = {}
for h, mem in groups:
    bs = {meta[f]['byte_sha256'] for f in mem}
    assert len(bs) == 1, mem
    byte_of_group[h] = bs.pop()
by_prefix = {}
for h, b in byte_of_group.items():
    by_prefix.setdefault(b[:12], []).append(h)

png_rows = []
for p in pngs:
    ds, _, h12 = p[:-4].rpartition('_')
    hit = by_prefix.get(h12, [])
    if ds == 'APSPC_RAW' and len(hit) == 1:
        h = hit[0]
        mem = dict(groups)[h]
        png_rows.append({'png_filename': p, 'group_id': gid[h], 'member_filenames': '|'.join(mem)})
    else:
        png_rows.append({'png_filename': p, 'group_id': 'UNMATCHED', 'member_filenames': ''})

# ---- write ----
with open(os.path.join(D, "groups.csv"), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=['group_id','pixel_sha256','n_copies','filename','part','xml_file','n_objects','classes_sorted'])
    w.writeheader(); w.writerows(groups_rows)
with open(os.path.join(D, "boxes.csv"), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=['group_id','filename','obj_idx','class_name','xmin','ymin','xmax','ymax','width','height'])
    w.writeheader(); w.writerows(boxes_rows)
with open(os.path.join(D, "png_map.csv"), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=['png_filename','group_id','member_filenames'])
    w.writeheader(); w.writerows(png_rows)

print("groups:", ngroups, "files:", nfiles, "dist:", dict(sorted(dist.items())))
print("groups.csv rows:", len(groups_rows), "boxes.csv rows:", len(boxes_rows), "png_map.csv rows:", len(png_rows))
print("gid range:", min(gid.values()), max(gid.values()))
print("classes:", collections.Counter(c for r in groups_rows for c in r['classes_sorted'].split('|') if c))
print("UNMATCHED:", sum(1 for r in png_rows if r['group_id']=='UNMATCHED'))
