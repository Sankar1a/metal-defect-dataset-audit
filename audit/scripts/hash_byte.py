import os, csv, hashlib
from PIL import Image
ROOT="."
DIRS=[("APSPC1","APSPC/extracted/part1/APSPC1"),("APSPC2","APSPC/extracted/part2/APSPC2")]
rows=[]
def sha_file(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for c in iter(lambda: f.read(1<<16), b''): h.update(c)
    return h.hexdigest()
for part,d in DIRS:
    for f in sorted(os.listdir(d)):
        if not f.lower().endswith('.jpg'): continue
        p=os.path.join(d,f)
        b=sha_file(p)
        with Image.open(p) as im:
            w,ht=im.size
            px=hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()
        rows.append({'part':part,'filename':f,'path':p,'byte_sha256':b,'pixel_sha256':px,'width':w,'height':ht,'size':os.path.getsize(p)})
with open("round18_groups_20261006/.all_hashes.csv","w",newline='') as f:
    w=csv.DictWriter(f,fieldnames=['part','filename','path','byte_sha256','pixel_sha256','width','height','size'])
    w.writeheader(); w.writerows(rows)
print("hashed",len(rows))
