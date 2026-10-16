"""Reproduce Fig. 2. Dependencies: matplotlib, numpy, pandas.

Run: python fig2_dose_response.py [--outdir DIRECTORY]
The embedded CSV is the sole source of plotted nonzero observations.
Means/SDs are plotted exactly as supplied; straight segments only join doses.
The 0% observations are the explicitly specified zero baseline.
"""
from pathlib import Path
from io import StringIO
import argparse
import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

CSV = """variant,dose,rate_pct,metric,delta_s0,delta_s1,delta_s42,mean,sd
same,d20,5.5,map5095,0.8460,1.0422,0.6232,0.8371,0.2096
same,d20,5.5,map50,0.7878,1.0902,1.4442,1.1074,0.3285
same,d28,7.6,map5095,1.2999,1.4625,1.0403,1.2676,0.2129
same,d28,7.6,map50,1.4766,1.8900,2.2748,1.8805,0.3992
same,d41,10.7,map5095,1.4299,1.8402,1.6218,1.6306,0.2053
same,d41,10.7,map50,1.4747,2.3758,3.0095,2.2867,0.7713
same,d82,19.3,map5095,0.2361,0.8315,0.7645,0.6107,0.3261
same,d82,19.3,map50,0.5396,1.2026,2.3916,1.3779,0.9384
conf,d20,5.5,map5095,0.1682,0.2129,-0.1654,0.0719,0.2067
conf,d20,5.5,map50,0.5969,0.5272,0.4673,0.5305,0.0649
conf,d28,7.6,map5095,0.3787,0.5056,0.2552,0.3798,0.1252
conf,d28,7.6,map50,1.1209,0.8536,1.0759,1.0168,0.1431
conf,d41,10.7,map5095,0.5065,0.8066,0.6210,0.6447,0.1514
conf,d41,10.7,map50,1.0359,1.1578,1.7025,1.2987,0.3549
conf,d82,19.3,map5095,0.1505,0.4911,0.2535,0.2984,0.1747
conf,d82,19.3,map50,0.0781,0.2326,1.4005,0.5704,0.7230
"""
DATA = pd.read_csv(StringIO(CSV))
ARROW_FONT = "DejaVu Sans"  # Liberation Sans draws a very small arrow
COLORS = {"same": "#2e5e9e", "conf": "#d9532c"}
RATES = np.array([0., 5.5, 7.6, 10.7, 19.3])
SEEDS = ["delta_s0", "delta_s1", "delta_s42"]
CHECKS = [("same", 7.6, "map5095", 1.27),
          ("same", 5.5, "map5095", 0.84),
          ("same", 10.7, "map5095", 1.63),
          ("same", 19.3, "map5095", 0.61),
          ("conf", 7.6, "map5095", 0.38),
          ("conf", 10.7, "map5095", 0.64),
          ("same", 7.6, "map50", 1.88)]


def select(variant, metric):
    return DATA[(DATA.variant == variant) & (DATA.metric == metric)].sort_values("rate_pct")


def check_data():
    assert len(DATA) == 16 and not DATA.isna().any().any()
    assert not DATA.duplicated(["variant", "rate_pct", "metric"]).any()
    for variant in COLORS:
        for metric in ["map5095", "map50"]:
            frame = select(variant, metric)
            assert np.array_equal(frame.rate_pct.to_numpy(), RATES[1:])
    for variant, rate, metric, expected in CHECKS:
        row = select(variant, metric).query("rate_pct == @rate").iloc[0]
        assert round(float(row["mean"]), 2) == expected, (variant, rate, metric)
        assert round(float(row[SEEDS].astype(float).mean()), 2) == expected


def configure(language):
    available = {f.name for f in font_manager.fontManager.ttflist}
    latin = next((f for f in ["Arial", "Liberation Sans", "DejaVu Sans"] if f in available))
    chinese = next((f for f in ["SimHei", "Microsoft YaHei", "Noto Sans CJK SC"]
                    if f in available), None)
    if language == "zh" and chinese is None:
        raise RuntimeError("Install SimHei, Microsoft YaHei, or Noto Sans CJK SC for Chinese output.")
    plt.rcParams.update({
        "font.family": [latin] + ([chinese] if language == "zh" else []),
        "font.sans-serif": [latin] + ([chinese] if language == "zh" else []),
        "font.size": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "axes.unicode_minus": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "text.usetex": False, "savefig.facecolor": "white",
    })
    return {"latin": latin, "chinese": chinese}


def reference_labels(ax, metric):
    """One grouped, directly connected label block per panel; no legend box."""
    if metric == "map5095":
        values = [0.81, 0.48]
        target_y = [0.87, 0.41]
        strings = ["n→m +0.81", "n→s +0.48"]
    else:
        values = [1.62, 1.53]
        target_y = [1.92, 1.24]
        strings = ["n→s +1.62", "n→m +1.53"]
    styles = [(0, (4, 2)), (0, (1.5, 2))]
    # Line style belongs to the model transition, and is consistent across panels.
    if metric == "map5095":
        styles.reverse()
    for value, dest, style in zip(values, target_y, styles):
        ax.axhline(value, color="0.51", lw=0.8, ls=style, zorder=1)
        ax.plot([21, 21.7, 22.1], [value, dest, dest], color="0.51",
                lw=0.6, clip_on=False, zorder=1)
    for dest, label in zip(target_y, strings):
        ax.text(22.25, dest, label, fontsize=6.5, color="0.36", family=ARROW_FONT,
                va="center", ha="left", clip_on=False)


def panel(ax, metric, language, single=False, letter=None):
    zh = language == "zh"
    primary = metric == "map5095"
    ylim = (-0.72, 2.60 if single else 2.40) if primary else (-0.90, 3.85)
    ax.set(xlim=(-0.5, 21), ylim=ylim)
    ax.axvspan(15, 21, facecolor="0.965", edgecolor="none", zorder=0)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="0.92", lw=0.45)
    ax.axhline(0, color="0.60", lw=0.65, zorder=1)
    # Keep the expected-dose guide in the data region, away from direct labels.
    ax.plot([7.6, 7.6], [0, 1.62 if primary else 2.65],
            color="0.45", lw=0.7, ls=(0, (2, 3)), zorder=1)
    frames = {v: select(v, metric) for v in COLORS}
    if single:
        xs = RATES[1:4]
        ax.fill_between(xs, frames["conf"]["mean"].to_numpy()[:3],
                        frames["same"]["mean"].to_numpy()[:3],
                        color=COLORS["same"], alpha=0.045, zorder=1)
    for variant, frame in frames.items():
        color = COLORS[variant]
        mean = np.r_[0., frame["mean"].to_numpy()]
        sd = np.r_[0., frame["sd"].to_numpy()]
        ax.fill_between(RATES, mean - sd, mean + sd, color=color,
                        alpha=0.13, lw=0, zorder=2)
        for col, offset in zip(SEEDS, [-0.17, 0., 0.17]):
            # Only display positions of seed dots are offset; means/bands use exact x.
            ax.scatter(RATES[1:] + offset, frame[col], s=7, c=color,
                       alpha=0.36, linewidths=0, zorder=3)
        ax.plot(RATES, mean, color=color, lw=1.5,
                marker="o" if variant == "same" else "s", ms=3.7,
                markeredgecolor="white", markeredgewidth=0.45, zorder=4)
        # Emphasize the real, already observed 7.6% point with a thin ring.
        ax.scatter([7.6], [mean[2]], s=39, marker="o" if variant == "same" else "s",
                   facecolors="none", edgecolors=color, linewidths=0.7, zorder=5)

    ax.set_xticks(RATES, ["0", "5.5", "7.6", "10.7", "19.3"])
    ax.set_yticks(np.arange(0, 2.01, 0.5) if primary else np.arange(0, 3.51, 0.5))
    for value, line in zip(ax.get_yticks(), ax.get_ygridlines()):
        if value > (1.91 if primary else 3.10):
            line.set_visible(False)
    ax.tick_params(axis="both", width=0.6, length=2.5, pad=2.5)
    for tick, value in zip(ax.get_xticklabels(), RATES):
        if value == 7.6:
            tick.set_fontweight("bold")
    ax.set_xlabel("验证集中泄漏图像的比例（%）" if zh else "Leaked images in validation set (%)", labelpad=6)
    ax.set_ylabel("相对 d0 的变化（百分点）" if zh else
                  ("Δ mAP50-95 vs. d0 (pp)" if primary else "Δ mAP50 vs. d0 (pp)"), labelpad=5)

    # Five annotation groups: expected level, extreme case, same, conf, references.
    ax.text(7.6, 1.035, "随机划分下的\n期望水平" if zh else "expected under\nrandom split",
            transform=ax.get_xaxis_transform(), ha="center", va="bottom",
            fontsize=6.5, linespacing=1.1, color="0.30")
    ax.text(18.0, 1.035, "82 组全部\n跨越划分" if zh else "all 82 groups\ncross the split",
            transform=ax.get_xaxis_transform(), ha="center", va="bottom",
            fontsize=6.5, linespacing=1.1, color="0.48")
    same_value = float(frames["same"].query("rate_pct == 7.6")["mean"].iloc[0])
    conf_value = float(frames["conf"].query("rate_pct == 7.6")["mean"].iloc[0])
    same_text = (f"same  +{same_value:.2f} pp (7.6%)\n" +
                 ("沿用训练副本标注" if zh else "same label as training copy"))
    if single:
        same_text += (f"\n标注冲突抵消 {same_value-conf_value:.2f} pp" if zh else
                      f"\nLabel conflict offsets {same_value-conf_value:.2f} pp")
    same_y = (2.45 if single else 2.23) if primary else 3.58
    ax.text(0.5, same_y, same_text, color=COLORS["same"], fontsize=6.5,
            va="top", linespacing=1.2)
    conf_text = f"conf  +{conf_value:.2f} pp (7.6%)\n" + ("保留自身标注" if zh else "own label")
    ax.text(5.4, -0.22, conf_text,
            color=COLORS["conf"], fontsize=6.5, va="top", linespacing=1.2)
    if single:
        # Bracket anchors at the supplied means: no inferred values.
        ax.annotate("", xy=(7.6, same_value), xytext=(7.6, conf_value),
                    arrowprops=dict(arrowstyle="|-|", lw=0.7,
                                    color="0.40", shrinkA=5, shrinkB=5), zorder=3)
    reference_labels(ax, metric)
    if letter:
        label = f"({letter})  " + ("mAP50-95" if primary else "mAP50")
        ax.text(-0.15, 1.14, label, transform=ax.transAxes,
                fontsize=8, fontweight="bold", va="bottom")


def check_text_bounds(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    for ax in fig.axes:
        texts = list(ax.texts) + [ax.xaxis.label, ax.yaxis.label]
        texts += ax.get_xticklabels() + ax.get_yticklabels()
        for text in texts:
            if not text.get_text():
                continue
            box = text.get_window_extent(renderer)
            assert box.x0 >= -1 and box.y0 >= -1 and box.x1 <= bounds.x1 + 1 and box.y1 <= bounds.y1 + 1, text.get_text()


def make_figure(version, language, outdir):
    fonts = configure(language)
    if version == "A":
        fig = plt.figure(figsize=(17.4 / 2.54, 7.0 / 2.54))
        left = fig.add_axes([0.077, 0.195, 0.312, 0.620])
        right = fig.add_axes([0.578, 0.195, 0.312, 0.620])
        panel(left, "map5095", language, letter="a")
        panel(right, "map50", language, letter="b")
    else:
        fig = plt.figure(figsize=(8.5 / 2.54, 7.0 / 2.54))
        ax = fig.add_axes([0.158, 0.195, 0.620, 0.620])
        panel(ax, "map5095", language, single=True)
    check_text_bounds(fig)
    name = f"fig2{version}_{language}"
    # Do not crop: exact journal dimensions are retained in both output formats.
    fig.savefig(outdir / f"{name}.pdf", metadata={"Title": name, "Creator": "matplotlib"})
    fig.savefig(outdir / f"{name}.png", dpi=600)
    plt.close(fig)
    return {"file": name, "width_cm": 17.4 if version == "A" else 8.5,
            "height_cm": 7.0, "dpi": 600, "fonts": fonts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    check_data()  # Fail before producing outputs if manuscript values disagree.
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Glyph .* missing from font")
        exports = [make_figure(v, lang, args.outdir)
                   for v in ["A", "B"] for lang in ["en", "zh"]]
    check_data()  # Required final manuscript self-check.
    (args.outdir / "fig2_validation.json").write_text(json.dumps({
        "assertions": "passed", "source_rows": len(DATA),
        "mean_and_sd": "supplied CSV columns, without recalculation or rounding",
        "variability": "+/- 1 SD across seeds 0, 1, 42; not a confidence interval",
        "x_positions": RATES.tolist(), "exports": exports,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("PASS: seven manuscript checks, source integrity, fonts and text bounds.")
    for item in exports:
        print(item["file"] + ": PDF + 600 dpi PNG")


if __name__ == "__main__":
    main()
