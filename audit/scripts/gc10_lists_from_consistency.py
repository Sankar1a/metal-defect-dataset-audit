"""Derive the two GC10-DET audit lists from the per-image consistency table.

Input : ../intermediate/gc10_folder_xml_consistency.csv  (one row per image file in each class folder)
Output: ../gc10_folder_xml_mismatch.csv   images with an XML file whose folder class is absent from the XML
        ../gc10_cross_folder_duplicates.csv  file stems that appear in more than one class folder
Usage: python gc10_lists_from_consistency.py
"""
import csv, os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "intermediate", "gc10_folder_xml_consistency.csv")
OUT = os.path.join(HERE, "..")

rows = list(csv.DictReader(open(SRC, newline="")))
fields = list(rows[0])
mismatch = [r for r in rows if r["xml_exists"] == "yes" and r["folder_in_xml"] == "no"]
dups = sorted((r for r in rows if int(r["stem_folder_count"]) > 1), key=lambda r: (r["stem"], r["folder_id"]))
for name, data in [("gc10_folder_xml_mismatch.csv", mismatch), ("gc10_cross_folder_duplicates.csv", dups)]:
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(data)
print(f"rows={len(rows)} with_xml={sum(r['xml_exists'] == 'yes' for r in rows)} "
      f"mismatch={len(mismatch)} cross_folder_rows={len(dups)} stems={len({r['stem'] for r in dups})}")
