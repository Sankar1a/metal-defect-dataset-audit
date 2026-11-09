"""Unit tests on synthetic data with planted problems.  Run:  python -m unittest discover -s tests -v"""
import csv
import json
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from dsqc.cli import main  # noqa: E402
import make_synthetic as ms  # noqa: E402


def read_csv(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def scan(args):
    main(["scan"] + args + ["--workers", "1"])


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def load(self, out):
        with open(os.path.join(out, "summary.json"), encoding="utf-8") as f:
            s = json.load(f)
        return s, read_csv(os.path.join(out, "findings.csv"))


class TestYolo(Base):
    def test_planted_problems(self):
        root, out = os.path.join(self.tmp, "ds"), os.path.join(self.tmp, "out")
        ms.make_yolo(root)
        scan([root, "-o", out, "--thumbs", "3"])
        s, f = self.load(out)
        codes = [x["code"] for x in f]
        st = s["checks"]
        self.assertEqual(st["c00_inventory"]["stats"]["image_files"], 9)
        self.assertEqual(st["c00_inventory"]["stats"]["yolo.orphan_annotations"], 1)
        self.assertEqual(st["c01_exact_dup"]["stats"]["groups"], 2)
        self.assertEqual(st["c01_exact_dup"]["stats"]["cross_split_groups"], 1)
        self.assertEqual(st["c01_exact_dup"]["stats"]["groups_with_reencoding"], 1)
        self.assertEqual(st["c02_dup_label_types"]["stats"]["type_A"], 1)
        self.assertEqual(st["c02_dup_label_types"]["stats"]["type_C3"], 1)   # 0.02 shift on 0.5 box -> IoU ~0.92
        self.assertEqual(st["c04a_transform_dup"]["stats"]["groups"], 1)
        self.assertEqual(st["c04a_transform_dup"]["stats"]["cross_split_groups"], 1)
        for code in ("missing_annotation", "empty_annotation", "out_of_bounds", "tiny_box", "full_image_box",
                     "duplicate_box", "parse_issue", "cross_split_leak", "rot_flip_cross_split"):
            self.assertIn(code, codes, code)
        self.assertTrue(any("class_id_out_of_range" in x["message"] for x in f))
        self.assertTrue(os.path.isfile(os.path.join(out, "report.html")))
        self.assertTrue(os.listdir(os.path.join(out, "thumbs")))

    def test_refuses_nonempty_out(self):
        root, out = os.path.join(self.tmp, "ds"), os.path.join(self.tmp, "out")
        ms.make_yolo(root)
        os.makedirs(out)
        open(os.path.join(out, "x"), "w").close()
        with self.assertRaises(SystemExit):
            scan([root, "-o", out])

    def test_diff(self):
        root = os.path.join(self.tmp, "ds")
        ms.make_yolo(root)
        a, b = os.path.join(self.tmp, "a"), os.path.join(self.tmp, "b")
        scan([root, "-o", a, "--no-transform"])
        os.remove(os.path.join(root, "images", "val", "f.png"))
        with open(os.path.join(root, "labels", "train", "a.txt"), "a") as fh:
            fh.write("1 0.1 0.1 0.1 0.1\n")
        scan([root, "-o", b, "--no-transform"])
        s_a, _ = self.load(a)
        s_b, _ = self.load(b)
        self.assertNotEqual(s_a["checks"]["c07_split_version"]["stats"]["fingerprint_sha256"],
                            s_b["checks"]["c07_split_version"]["stats"]["fingerprint_sha256"])
        diff_csv = os.path.join(self.tmp, "d.csv")
        main(["diff", a, b, "-o", diff_csv])
        rows = read_csv(diff_csv)
        self.assertIn(("removed", "images/val/f.png"), [(r["status"], r["path"]) for r in rows])
        self.assertIn(("changed", "labels/train/a.txt"), [(r["status"], r["path"]) for r in rows])


class TestGC10Like(Base):
    def test_folder_vs_xml(self):
        import yaml
        root, out = os.path.join(self.tmp, "gc"), os.path.join(self.tmp, "out")
        spec = ms.make_gc10_like(root)
        sp = os.path.join(self.tmp, "spec.yaml")
        with open(sp, "w", encoding="utf-8") as fh:
            yaml.safe_dump(spec, fh, allow_unicode=True)
        scan(["--spec", sp, "-o", out])
        s, f = self.load(out)
        inv = s["checks"]["c00_inventory"]["stats"]
        self.assertEqual(inv["image_files"], 8)
        self.assertEqual(inv["distinct_stems"], 7)
        self.assertEqual(inv["xml.annotation_files"], 7)
        self.assertEqual(inv["xml.images_with_label"], 7)       # g4 has no xml
        self.assertEqual(inv["xml_folder.images_label_unmapped"], 1)   # g7 tag 'all'
        self.assertEqual(s["checks"]["c02_dup_label_types"]["stats"]["type_shared_file"], 1)  # g5 in two folders
        self.assertEqual(inv["xml.orphan_annotations"], 1)
        c3 = s["checks"]["c03_source_consistency"]["stats"]
        # disagreements: g2 (class absent), g3 (empty), 2/g5 (folder 2 present? yes via alias -> no), g6
        rows = read_csv(os.path.join(out, "tables", "c03_source_consistency__per_image_folder_vs_xml.csv"))
        no = sorted(r["relative_path"] for r in rows if r["class_in_boxes"] == "no")
        self.assertEqual(no, ["1/g2.jpg", "2/g3.jpg"])
        self.assertEqual(c3["folder_vs_xml.images_with_both"], 7)
        self.assertEqual(c3["xml_folder_vs_xml.image_class_unmapped"], 1)
        self.assertEqual(c3["xml_folder_vs_folder.disagree"], 2)  # 2/g5 (tag 1) and 2/g6 (tag 1)
        codes = [x["code"] for x in f]
        for code in ("alias_class", "size_mismatch", "same_stem_several_files", "orphan_annotation",
                     "image_class_vs_empty_annotation", "image_class_absent_from_boxes", "image_class_sources_differ"):
            self.assertIn(code, codes, code)


class TestCoco(Base):
    def test_coco(self):
        root, out = os.path.join(self.tmp, "co"), os.path.join(self.tmp, "out")
        ms.make_coco(root)
        scan([root, "-o", out])
        s, f = self.load(out)
        inv = s["checks"]["c00_inventory"]["stats"]
        self.assertEqual(inv["image_files"], 2)
        self.assertEqual(inv["coco.orphan_annotations"], 2)     # x3 entry without file, annotation to image 7
        self.assertIn("out_of_bounds", [x["code"] for x in f])
        self.assertIn("empty_annotation", [x["code"] for x in f])


class TestComparator(unittest.TestCase):
    def test_types(self):
        from dsqc.geometry import group_type, pair_type
        a = [("x", (0, 0, 10, 10))]
        self.assertEqual(pair_type(a, a)[0], "identical")
        self.assertEqual(pair_type(a, [("y", (0, 0, 10, 10))])[0], "A")
        self.assertEqual(pair_type(a, a + [("x", (20, 20, 30, 30))])[0], "B")
        self.assertEqual(pair_type(a, [("x", (5, 0, 15, 10))])[0], "C1")       # IoU 1/3
        self.assertEqual(pair_type(a, [("x", (1, 0, 11, 10))])[0], "C2")       # IoU 0.818
        self.assertEqual(pair_type(a, [("x", (0, 0, 10, 10.5))])[0], "C3")     # IoU 0.952
        self.assertEqual(group_type([a, a, [("y", (0, 0, 1, 1))]])[0], "A")


class TestSplitLists(Base):
    """0.1.2: split lists that name images by stem, file name, relative or foreign absolute path."""

    def test_lists(self):
        import yaml
        root, out = os.path.join(self.tmp, "voc"), os.path.join(self.tmp, "out")
        spec = ms.make_voc_lists(root)
        sp = os.path.join(self.tmp, "spec.yaml")
        with open(sp, "w", encoding="utf-8") as fh:
            yaml.safe_dump(spec, fh, allow_unicode=True)
        scan(["--spec", sp, "-o", out, "--no-transform"])
        s, f = self.load(out)
        inv = s["checks"]["c00_inventory"]["stats"]
        self.assertEqual(inv["image_files"], 11)
        # lists are read in key order (safe_dump sorts: test, train, val); im0 is named in test (as
        # Annotations/im0.xml) and in train, the later list wins and a note records the conflict
        self.assertEqual(inv["split.train"], 4)          # im0 im1 im2 img.v2
        self.assertEqual(inv["split.val"], 3)            # im3 im4 im5
        self.assertEqual(inv["split.test"], 2)           # im6 im7
        self.assertEqual(inv["split.None"], 2)           # the two 'twin' images (ambiguous entry)
        c7 = s["checks"]["c07_split_version"]["stats"]
        self.assertEqual((c7["split_list.train.entries"], c7["split_list.train.matched"]), (4, 4))
        self.assertEqual((c7["split_list.val.entries"], c7["split_list.val.matched"]), (3, 3))
        self.assertEqual((c7["split_list.test.entries"], c7["split_list.test.matched"],
                          c7["split_list.test.not_found"], c7["split_list.test.ambiguous"]), (5, 3, 1, 1))
        codes = [x["code"] for x in f]
        self.assertIn("split_list_entries_unmatched", codes)
        notes = " ".join(s["checks"]["c00_inventory"]["notes"])
        self.assertIn("JPEGImages/im0.jpg listed in both test and train", notes)
        self.assertIn("'nothere'", notes)
        self.assertIn("'twin' (2 images)", notes)


class TestSmallBoxJitter(Base):
    """0.1.2: geometry columns and the small-box jitter flag of check c02 (types unchanged)."""

    def run_case(self, extra_cfg=None):
        root, out = os.path.join(self.tmp, "jit"), os.path.join(self.tmp, "out")
        expected = ms.make_jitter(root)
        args = [root, "-o", out, "--no-transform"]
        if extra_cfg:
            import yaml
            cp = os.path.join(self.tmp, "cfg.yaml")
            with open(cp, "w", encoding="utf-8") as fh:
                yaml.safe_dump(extra_cfg, fh)
            args += ["--config", cp]
        scan(args)
        s, f = self.load(out)
        rows = read_csv(os.path.join(out, "tables", "c02_dup_label_types__duplicate_group_types.csv"))
        by = {r["files"].split("|")[0].split("/")[-1].rsplit("_", 1)[0]: r for r in rows}
        sev = {x["images"].split("|")[0].split("/")[-1].rsplit("_", 1)[0]: x["severity"]
               for x in f if x["check"] == "c02_dup_label_types"}
        return expected, s, by, sev

    def test_flag(self):
        expected, s, by, sev = self.run_case()
        for name, (t, j) in expected.items():
            self.assertEqual((by[name]["type"], by[name]["small_box_jitter"]), (t, j), name)
        self.assertEqual(by["small"]["max_coord_diff_px"], "6.0")
        self.assertEqual(by["small"]["max_center_offset_px"], "8.5")
        st = s["checks"]["c02_dup_label_types"]["stats"]
        self.assertEqual((st["type_C1"], st["type_C2"]), (3, 1))
        self.assertEqual((st["type_C1_small_box_jitter"], st["type_C2_small_box_jitter"]), (1, 1))
        self.assertEqual((sev["small"], sev["large"], sev["apart"], sev["small_c2"]), ("low", "medium", "medium", "low"))

    def test_no_downgrade(self):
        _, _, _, sev = self.run_case({"jitter_downgrade": False})
        self.assertEqual(sev["small"], "medium")


class TestGeometry(unittest.TestCase):
    def test_group_geometry(self):
        from dsqc.geometry import group_geometry, group_type
        a = [("x", (100, 100, 120, 120))]
        b = [("x", (106, 106, 126, 126))]
        self.assertEqual(group_type([a, b])[0], "C1")
        g = group_geometry([a, b])
        self.assertTrue(g["small_box_jitter"])
        self.assertEqual(g["max_coord_diff_px"], 6)
        self.assertIsNone(group_geometry([a, a + a]))      # counts differ -> not a C-type group
        big = [("x", (0, 0, 100, 100))]
        self.assertFalse(group_geometry([big, [("x", (10, 0, 110, 100))]])["small_box_jitter"])   # IoU 0.82, large box
        self.assertIsNone(group_geometry([a, a])["low_iou_max_area_px2"])   # identical: no low-IoU pair


if __name__ == "__main__":
    unittest.main()
