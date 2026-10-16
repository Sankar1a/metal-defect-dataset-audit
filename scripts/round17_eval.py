#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Round 17 · eval-only · 45 evaluations, single process, serial, device=0.

  A4 : 6 NEU-DET best.pt x {val359, val360}          = 12
  B2 : 3 APSPC A-arm best.pt x 9 val-set yamls       = 27
  C3 : 6 GC10 best.pt x XML val-set yaml             =  6
                                                    ----
                                                      45

model.val(...) kwargs are the round-16 template, copied verbatim from
  exp_apspc_dose_unified_20260915/scripts/eval_dose.py  lines 88-90
Only `data`, `project` and `name` change (see eval_kwargs_source.txt).

Every row is written and flushed the moment its evaluation finishes.
"""
import csv
import os
import sys
import time

ROOT = "/home/DSJ44/paper2_yolov11_steel"
OUT = os.path.join(ROOT, "round17_eval_20261005")
VALRUNS = os.path.join(OUT, "val_runs")

# ---------------------------------------------------------------- inputs ----
NEU_RUNS = [("YOLO11n", 0, "neu_yolo11n_s0"), ("YOLO11n", 1, "neu_yolo11n_s1"),
            ("YOLO11n", 42, "neu_yolo11n_s42"), ("YOLO11s", 0, "neu_yolo11s_s0"),
            ("YOLO11s", 1, "neu_yolo11s_s1"), ("YOLO11s", 42, "neu_yolo11s_s42")]
NEU_YAML = {
    "val359": os.path.join(ROOT, "audits/leakage_sensitivity/neu_det_val359.yaml"),
    "val360": os.path.join(ROOT, "exp_leakage_20260817/configs/neu_val360_controlled.yaml"),
}

SEEDS = [0, 1, 42]
VAL_SETS = ["val_d0",
            "val_d20_same", "val_d28_same", "val_d41_same", "val_d82_same",
            "val_d20_conf", "val_d28_conf", "val_d41_conf", "val_d82_conf"]
APSPC_CFG = os.path.join(ROOT, "exp_apspc_dose_20260817", "configs")

GC10_JOBS = [("xml", 0), ("xml", 1), ("xml", 42),
             ("dir", 0), ("dir", 1), ("dir", 42)]
GC10_XML_YAML = os.path.join(ROOT, "exp_gc10_label_20260914/configs/gc10_xml.yaml")

RECON = os.path.join(ROOT, "verify3_20260805/metric_reconciliation.csv")
DOSE = os.path.join(ROOT, "exp_apspc_dose_unified_20260915/dose_results_unified.csv")
GC10 = os.path.join(ROOT, "exp_gc10_label_20260914/gc10_label_results.csv")


def sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class Sink(object):
    """Append-and-flush CSV writer; the header is written once, up front."""

    def __init__(self, path, cols):
        self.path = path
        self.cols = cols
        new = not os.path.exists(path)
        self.f = open(path, "a", newline="", encoding="utf-8")
        self.w = csv.writer(self.f)
        if new:
            self.w.writerow(cols)
            self.f.flush()
        self.n = 0

    def row(self, values):
        assert len(values) == len(self.cols), (len(values), self.cols)
        self.w.writerow(values)
        self.f.flush()
        os.fsync(self.f.fileno())
        self.n += 1

    def close(self):
        self.f.close()


def per_class(res):
    """{class_name: (ap50, ap5095)} read from the returned object."""
    b = res.box
    ap50 = list(b.ap50)
    ap = list(b.ap)
    idx = list(b.ap_class_index)
    names = res.names
    if isinstance(names, (list, tuple)):
        names = {i: n for i, n in enumerate(names)}
    got = {}
    for i, c in enumerate(idx):
        got[int(c)] = (float(ap50[i]), float(ap[i]))
    return names, got


def evaluate(best_pt, data_yaml, name):
    """The round-16 call, verbatim except for data / project / name."""
    from ultralytics import YOLO
    return YOLO(best_pt).val(data=data_yaml, imgsz=640, split="val", batch=16,
                             device=0, verbose=False,
                             project=VALRUNS, name=name)


def main():
    os.makedirs(VALRUNS, exist_ok=True)
    t0 = time.time()
    print("R17 eval start=%s" % time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)

    # ------------------------------------------------------------ refs ----
    recon = {r["run_name"]: r for r in read_csv(RECON)}
    dose = {(r["seed"], r["val_set"]): r for r in read_csv(DOSE)}
    gc10 = {r["run_name"]: r for r in read_csv(GC10)}

    a_over = Sink(os.path.join(OUT, "A_neu_overall.csv"),
                  ["model", "seed", "run_dir", "best_pt_sha256", "eval_set",
                   "map50", "map5095", "ref_bestpt_val_map50", "ref_bestpt_val_map5095"])
    a_pc = Sink(os.path.join(OUT, "A_neu_per_class.csv"),
                ["model", "seed", "eval_set", "class_name", "ap50", "ap5095"])
    b_over = Sink(os.path.join(OUT, "B_dose_overall.csv"),
                  ["seed", "valset", "map50", "map5095", "ref_map50", "ref_map5095"])
    b_pc = Sink(os.path.join(OUT, "B_dose_per_class.csv"),
                ["seed", "valset", "class_name", "ap50", "ap5095"])
    c_over = Sink(os.path.join(OUT, "C_gc10_overall.csv"),
                  ["condition", "seed", "map50", "map5095", "ref_map50", "ref_map5095"])
    c_pc = Sink(os.path.join(OUT, "C_gc10_per_class.csv"),
                ["condition", "seed", "class_name", "ap50", "ap5095"])

    done = 0
    failures = []

    # ================================================== A4 : 12 evals ====
    print("\n" + "=" * 70 + "\nA4  NEU-DET leakage: 6 best.pt x {val359,val360} = 12\n" + "=" * 70, flush=True)
    for model, seed, run in NEU_RUNS:
        run_dir = os.path.join("runs_unified_20260803", run)
        best = os.path.join(ROOT, run_dir, "weights", "best.pt")
        best_sha = sha256(best)
        ref = recon[run]
        for setname in ("val359", "val360"):
            name = "A_%s__on_%s" % (run, setname)
            try:
                r = evaluate(best, NEU_YAML[setname], name)
                m50, m5095 = float(r.box.map50), float(r.box.map)
                names, got = per_class(r)
                ref50 = ref["bestpt_val_map50"] if setname == "val359" else ""
                ref5095 = ref["bestpt_val_map5095"] if setname == "val359" else ""
                a_over.row([model, seed, run_dir, best_sha, setname,
                            repr(m50), repr(m5095), ref50, ref5095])
                for ci in sorted(names):
                    v = got.get(ci)
                    a_pc.row([model, seed, setname, names[ci],
                              repr(v[0]) if v else "", repr(v[1]) if v else ""])
                done += 1
                print("  A4 %-14s %-7s map50=%s map5095=%s  [%d/45]"
                      % (run, setname, repr(m50), repr(m5095), done), flush=True)
            except Exception as e:
                failures.append("A4 %s %s: %r" % (run, setname, e))
                print("  A4 FAIL %s %s: %r" % (run, setname, e), flush=True)

    # ================================================== B2 : 27 evals ====
    print("\n" + "=" * 70 + "\nB2  APSPC dose: 3 best.pt x 9 val sets = 27\n" + "=" * 70, flush=True)
    for seed in SEEDS:
        best = os.path.join(ROOT, "exp_apspc_dose_unified_20260915/runs",
                            "apspc_yolo11n_leaktrain_unified_s%d" % seed, "weights", "best.pt")
        for v in VAL_SETS:
            vyaml = os.path.join(APSPC_CFG, "apspc_%s.yaml" % v[len("val_"):])
            name = "B_s%d__on_%s" % (seed, v)
            try:
                r = evaluate(best, vyaml, name)
                m50, m5095 = float(r.box.map50), float(r.box.map)
                names, got = per_class(r)
                ref = dose[(str(seed), v)]
                b_over.row([seed, v, repr(m50), repr(m5095),
                            ref["map50"], ref["map5095"]])
                for ci in sorted(names):
                    x = got.get(ci)
                    b_pc.row([seed, v, names[ci],
                              repr(x[0]) if x else "", repr(x[1]) if x else ""])
                done += 1
                print("  B2 s%-3d %-13s map50=%s map5095=%s  [%d/45]"
                      % (seed, v, repr(m50), repr(m5095), done), flush=True)
            except Exception as e:
                failures.append("B2 s%d %s: %r" % (seed, v, e))
                print("  B2 FAIL s%d %s: %r" % (seed, v, e), flush=True)

    # =================================================== C3 : 6 evals ====
    print("\n" + "=" * 70 + "\nC3  GC10: 6 best.pt x XML val set = 6\n" + "=" * 70, flush=True)
    for cond, seed in GC10_JOBS:
        run = "gc10_yolo11n_%s_s%d" % (cond, seed)
        best = os.path.join(ROOT, "exp_gc10_label_20260914/runs", run, "weights", "best.pt")
        name = "C_%s__on_xml" % run
        try:
            r = evaluate(best, GC10_XML_YAML, name)
            m50, m5095 = float(r.box.map50), float(r.box.map)
            names, got = per_class(r)
            ref = gc10[run]
            c_over.row([cond, seed, repr(m50), repr(m5095),
                        ref["eval_on_xml_map50"], ref["eval_on_xml_map5095"]])
            for ci in sorted(names):
                x = got.get(ci)
                c_pc.row([cond, seed, names[ci],
                          repr(x[0]) if x else "", repr(x[1]) if x else ""])
            done += 1
            print("  C3 %-20s map50=%s map5095=%s  [%d/45]"
                  % (run, repr(m50), repr(m5095), done), flush=True)
        except Exception as e:
            failures.append("C3 %s: %r" % (run, e))
            print("  C3 FAIL %s: %r" % (run, e), flush=True)

    for s in (a_over, a_pc, b_over, b_pc, c_over, c_pc):
        s.close()

    print("\nrow counts: A_over=%d A_pc=%d B_over=%d B_pc=%d C_over=%d C_pc=%d"
          % (a_over.n, a_pc.n, b_over.n, b_pc.n, c_over.n, c_pc.n), flush=True)
    if failures:
        print("\nFAILURES (%d):" % len(failures), flush=True)
        for x in failures:
            print("  " + x, flush=True)

    with open(os.path.join(OUT, "DONE.txt"), "w", encoding="utf-8") as f:
        f.write("ROUND17_EVAL_DONE n=%d\n" % done)
        f.flush()
    print("\nROUND17_EVAL_DONE n=%d" % done, flush=True)
    print("R17 eval end=%s elapsed=%.1fs" % (time.strftime("%Y-%m-%d %H:%M:%S"),
                                             time.time() - t0), flush=True)
    return 0 if done == 45 else 1


if __name__ == "__main__":
    sys.exit(main())
