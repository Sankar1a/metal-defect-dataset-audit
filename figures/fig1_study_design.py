"""Study-design figure v2: a defect-detection pipeline with the factors that change the
reported score marked in orange where they enter (layout idea after Rosenblatt et al. 2024,
Nat. Commun., Fig. 1). Schematic only; thumbnails are real APSPC images from Figure 1.
Usage: python make_flowchart_v2.py --lang zh|en --out DIR
"""
import argparse, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, ConnectionPatch
from matplotlib import font_manager
from PIL import Image
import numpy as np

INK, MUTED, LINE = "#1f2328", "#5b6168", "#a3a9b0"
TRAIN, VAL = "#5a5f9e", "#1b9e77"          # split colours (purple-ish / green), as in many split schematics
ORANGE = "#d9532c"                         # factors that change the reported score
SOFT = "#f3f4f6"
HERE = os.path.dirname(os.path.abspath(__file__))

T = {
"zh": dict(font="Noto Sans CJK SC", thumbs="thumbs_py",
    stages=["公开数据集", "标注", "训练 / 验证划分", "训练", "评估"],
    train="训练集", val="验证集", same="像素相同\n标注不同",
    sizes="n / s / m", seeds="种子 0 / 1 / 42", eval1="mAP50", eval2="mAP50-95",
    score="报告的\n分数",
    f_label=("标签来源取舍", "实验四：GC10-DET，81 张"),
    f_split=("重复图像跨越划分", "实验二：NEU-DET，1 对", "实验三：APSPC，0–19.3%"),
    f_train=("模型尺寸（实验一，作参照）", "随机种子：每个条件 3 次", "其他检测器（实验五）"),
    g_data=("训练前先审计（2.2 节）", "重复图像、标注一致性、溯源"),
    g_eval=("配对评估（2.5 节）", "同一权重，只换验证集"),
    band1="统一协议：参数闸门与优化器解析行双重核验，训练逐位可复现（2.3、2.6 节）",
    band2="各因素的影响均与种子波动放在同一尺度上比较（3.9 节），据此给出检查清单（4.4 节）",
    legend="橙色：会改变报告分数、本文逐一测量的因素"),
"en": dict(font="Liberation Sans", thumbs="thumbs",
    stages=["Public dataset", "Labels", "Train / val split", "Training", "Evaluation"],
    train="train", val="val", same="identical pixels,\ndifferent labels",
    sizes="n / s / m", seeds="seeds 0 / 1 / 42", eval1="mAP50", eval2="mAP50-95",
    score="Reported\nscore",
    f_label=("Which label source", "Exp. 4 · GC10-DET, 81 images"),
    f_split=("Duplicates across the split", "Exp. 2 · NEU-DET, 1 pair", "Exp. 3 · APSPC, 0–19.3%"),
    f_train=("Model size · Exp. 1 (reference)", "Random seed · 3 runs each", "Other detectors · Exp. 5"),
    g_data=("Audit before training (2.2)", "duplicates · labels · provenance"),
    g_eval=("Paired evaluation (2.5)", "same weights, swap val set"),
    band1="Unified protocol: config gate + optimizer log; bitwise reproducible (2.3, 2.6)",
    band2="Every effect is compared with seed variation on one scale (3.9); checklist in 4.4",
    legend="Orange: factors that change the reported score, each measured here"),
}


def rbox(ax, x0, y0, x1, y1, fc=SOFT, ec="none", lw=0.6, r=0.8, z=1, ls="-"):
    p = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=fc, ec=ec, lw=lw, zorder=z, linestyle=ls)
    ax.add_patch(p)
    return p


def arrow(ax, x0, y0, x1, y1, color=MUTED, lw=0.9, ms=7, style="-|>", z=3, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=ms, lw=lw,
                                 color=color, shrinkA=0, shrinkB=0, zorder=z, linestyle=ls))


def thumb(ax, path, x0, y0, w, edge=None, lw=0.8, z=3):
    im = np.asarray(Image.open(path).convert("RGB").resize((256, 192), Image.LANCZOS))
    h = w * 0.75
    # pdfimages returns matplotlib-embedded rasters bottom-up; origin="lower" restores them
    ax.imshow(im, extent=(x0, x0 + w, y0, y0 + h), origin="lower", zorder=z,
              interpolation="none", aspect="auto")
    if edge:
        ax.add_patch(Rectangle((x0, y0), w, h, fill=False, ec=edge, lw=lw, zorder=z + 1))
    return h


def draw(lang, out):
    t = T[lang]
    avail = {f.name for f in font_manager.fontManager.ttflist}
    fam = [t["font"]] if t["font"] in avail else ["DejaVu Sans"]
    plt.rcParams.update({"font.family": fam + ["DejaVu Sans"], "pdf.fonttype": 42,
                         "axes.unicode_minus": False})
    fig = plt.figure(figsize=(17.4 / 2.54, 8.0 / 2.54))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 46.5); ax.axis("off")
    th = lambda i: os.path.join(HERE, t["thumbs"], f"t-00{i}.png")
    fs, fs2, fst = 6.2, 5.6, 6.8

    xs = [8.5, 26.5, 47.0, 68.0, 85.0]           # stage centres
    ytit, yicon = 43.4, 30.0
    for x, s in zip(xs, t["stages"]):
        ax.text(x, ytit, s, fontsize=fst, fontweight="bold", ha="center", va="center", color=INK)

    # 1 public dataset: stack of three real images
    for k, (i, dx, dy) in enumerate([(7, -2.2, 2.6), (2, -1.1, 1.3), (5, 0.0, 0.0)]):
        thumb(ax, th(i), xs[0] - 6.0 + dx, yicon - 6.0 + dy, 12.0, edge="white", lw=0.6, z=3 + k)
    # 2 labels: one image with its box
    thumb(ax, th(3), xs[1] - 6.0, yicon - 5.2, 12.0, edge=LINE, lw=0.5)

    # 3 split: train block and val block, with an identical pair across them
    tx0, vx0, bw = xs[2] - 12.0, xs[2] + 2.0, 10.0
    for x0, col, lab in [(tx0, TRAIN, t["train"]), (vx0, VAL, t["val"])]:
        rbox(ax, x0, yicon - 7.5, x0 + bw, yicon + 6.5, fc="white", ec=col, lw=1.0, r=0.6)
        ax.text(x0 + bw / 2, yicon + 8.0, lab, fontsize=fs2, ha="center", va="center", color=col,
                fontweight="bold")
        for r in range(2):                      # placeholder tiles = other images
            for c in range(3):
                ax.add_patch(Rectangle((x0 + 0.9 + c * 2.95, yicon - 6.6 + r * 2.4), 2.5, 1.9,
                                       fc=col, alpha=0.18, ec="none", zorder=2))
    hA = thumb(ax, th(0), tx0 + 0.8, yicon - 1.3, bw - 1.6, edge=ORANGE, lw=1.1)
    thumb(ax, th(1), vx0 + 0.8, yicon - 1.3, bw - 1.6, edge=ORANGE, lw=1.1)
    ya = yicon - 1.3 + hA / 2
    arrow(ax, tx0 + bw - 0.8, ya, vx0 + 0.8, ya, color=ORANGE, lw=0.9, ms=6, style="<|-|>", z=6)
    ax.text(xs[2], yicon + 8.3, t["same"], fontsize=fs2, ha="center", va="center",
            color=ORANGE, linespacing=1.05)
    arrow(ax, xs[2], yicon + 6.6, xs[2], ya + 0.9, color=ORANGE, lw=0.6, ms=4)

    # 4 training: three model sizes + seeds
    for k, s in enumerate([2.6, 3.6, 4.8]):
        x0 = xs[3] - 6.5 + [0, 3.4, 7.8][k]
        rbox(ax, x0, yicon - s / 2 + 1.5, x0 + s, yicon + s / 2 + 1.5, fc="#d9dde3", ec="#8b929b", lw=0.6, r=0.4)
    ax.text(xs[3], yicon - 3.2, t["sizes"], fontsize=fs2, ha="center", va="center", color=INK)
    ax.text(xs[3], yicon - 5.6, t["seeds"], fontsize=fs2, ha="center", va="center", color=MUTED)

    # 5 evaluation -> reported score
    rbox(ax, xs[4] - 4.6, yicon - 4.0, xs[4] + 4.6, yicon + 4.0, fc="white", ec="#8b929b", lw=0.7)
    ax.text(xs[4], yicon + 1.4, t["eval1"], fontsize=fs2, ha="center", va="center", color=INK)
    ax.text(xs[4], yicon - 1.4, t["eval2"], fontsize=fs2, ha="center", va="center", color=INK)
    rbox(ax, 92.6, yicon - 4.0, 99.4, yicon + 4.0, fc=INK, ec=INK)
    ax.text(96.0, yicon, t["score"], fontsize=fs2, ha="center", va="center", color="white",
            fontweight="bold", linespacing=1.05)
    arrow(ax, xs[4] + 4.8, yicon, 92.4, yicon)

    # pipeline arrows
    for a, b in [(xs[0] + 7.0, xs[1] - 6.6), (xs[1] + 6.6, tx0 - 0.6), (vx0 + bw + 0.6, xs[3] - 7.4),
                 (xs[3] + 7.6, xs[4] - 5.0)]:
        arrow(ax, a, yicon, b, yicon)

    # factor annotations (orange) and design notes (grey)
    def note(x, lines, color, head_color=None, y=19.6, ha="center"):
        yy = y
        for j, s in enumerate(lines):
            ax.text(x, yy, s, fontsize=fs if j == 0 else fs2, ha=ha, va="center",
                    color=(head_color or color) if j == 0 else color,
                    fontweight="bold" if j == 0 else "normal")
            yy -= 2.7
    for x, lines in [(xs[1], t["f_label"]), (xs[2], t["f_split"]), (xs[3], t["f_train"])]:
        arrow(ax, x, 23.6, x, 21.3, color=ORANGE, lw=0.7, ms=5, style="<|-")
        note(x, lines, ORANGE)
    for x, lines in [(xs[0], t["g_data"]), (xs[4] + 2.0, t["g_eval"])]:
        note(x, lines, MUTED, head_color=INK)

    # bottom bands
    rbox(ax, 0.6, 5.6, 99.4, 9.4, fc=SOFT)
    ax.text(50, 7.5, t["band1"], fontsize=fs2, ha="center", va="center", color=INK)
    rbox(ax, 0.6, 0.8, 99.4, 4.6, fc="#fdeee8")
    ax.text(50, 2.7, t["band2"], fontsize=fs2, ha="center", va="center", color=ORANGE, fontweight="bold")

    ax.set_xlim(0, 100); ax.set_ylim(0, 46.5)
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, f"fig_design_{lang}.pdf"), dpi=600)
    fig.savefig(os.path.join(out, f"fig_design_{lang}.png"), dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["zh", "en"], required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "out2"))
    a = ap.parse_args()
    draw(a.lang, a.out)
