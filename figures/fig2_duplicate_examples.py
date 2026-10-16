#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第24轮 任务书13 · 论文图 1 高清重出(v3,英文类名版 + 拼音版)。

底稿: round18_fig1v2_20261006/make_figure_v2.py(原目录不动)
在 v2 基础上只改三处,其余(行顺序、文件、颜色映射、文字颜色规则、框线宽、字号、
figsize/布局、子图 1280x960、boxes.csv 原值坐标、画前 pixel_sha256 核对)保持 v2 原样:
  1. PDF 高清: 所有 imshow(...) 加 interpolation='none'; 保存 PDF 加 dpi=600。
     保存后运行 pdfimages -list 自检, 每张照片 width 应 >= 1280, 否则停止并报告。
  2. 类别名两版: --labels {en,pinyin}。
     XML 原文(pinyin 版照用) -> en 版:
       qikeng->pit  budaodian->non-conductive  cahua->scratch
       tufen->powder bump  tucengkailie->coating crack
     en 版类名更长,若文字框会超出图片右边界,改为右对齐到框的右上角(文字框右边缘贴框右边,
     仍越界则贴图片右边界);上/下边界规则与 v2 相同。
  3. 画前重算 9 个文件的 pixel_sha256 并写 check_sha_v3b.txt, 不一致则停止(不画图)。
  4. (13b) 标签避让: 同图内新标签与已画标签相交时改画到框下方外侧;下方越界则贴框内左下角;
     移动后仍重叠只记入日志,不停止。统计 n_moved_below。
  5. (13b) PDF 字体: compose() 开头 matplotlib.rcParams['pdf.fonttype'] = 42 (TrueType 嵌入)。
  6. (13b) 输出全部加 _en_v3b 后缀, 不覆盖 v3 产物。

输入(只读):
  APSPC/extracted/part1/APSPC1/ 、 APSPC/extracted/part2/APSPC2/
  round18_groups_20261006/boxes.csv 、 round18_groups_20261006/groups.csv
产物: 只写 round24_fig1v3_20261016/
  figure1_en_v3b.pdf / .png(600 dpi) / _preview.png(150 dpi)
  panels_en_v3b/ 、 check_sha_v3b.txt 、 check_pdfimages_v3b.txt 、 check_pdffonts_v3b.txt
  run_log_en_v3b.txt
"""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from PIL import Image, ImageDraw, ImageFont

ROOT = "/home/DSJ44/paper2_yolov11_steel"
D = os.path.join(ROOT, "round24_fig1v3_20261016")
BOXES = os.path.join(ROOT, "round18_groups_20261006", "boxes.csv")
GROUPS = os.path.join(ROOT, "round18_groups_20261006", "groups.csv")
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

PANEL_W = 1280                      # 每张子图缩放后宽度
SRC_W = 2560                        # 原始宽度
SCALE = PANEL_W / SRC_W             # 0.5
LW = int(round(18 * SCALE))         # 线宽 18 像素(原分辨率)
FS = int(round(150 * SCALE))        # 字号 150 像素(原分辨率)
PAD = 10                            # 标签与框的间距(缩放后像素)

PDF_DPI = 600                       # 任务书 13 第 1 条: PDF 按 600 dpi 嵌入
PNG_DPI = 600
PREVIEW_DPI = 150

# Okabe-Ito 色盲友好调色板
PALETTE = ["#E69F00", "#56B4E9", "#009E73", "#F0E442",
           "#0072B2", "#D55E00", "#CC79A7", "#000000"]
# 浅色背景 -> 黑字(任务书 5b 第 3 条)
BLACK_TEXT_BG = {"#F0E442", "#E69F00", "#56B4E9"}

ROWS = [
    ("(a)", "G53", ["img283.jpg", "img3041.jpg"]),
    ("(b)", "G16", ["img135.jpg", "img658.jpg"]),
    ("(c)", "G50", ["img103.jpg", "img1220.jpg", "img524.jpg"]),
    ("(d)", "G36", ["img1047.jpg", "img468.jpg"]),
]

IMG_DIRS = ["APSPC/extracted/part1/APSPC1", "APSPC/extracted/part2/APSPC2"]

# 任务书 13 第 2 条: 类别名两版(XML 原文 = pinyin 版照用)
LABELS = {
    "pinyin": {
        "qikeng": "qikeng",
        "budaodian": "budaodian",
        "cahua": "cahua",
        "tufen": "tufen",
        "tucengkailie": "tucengkailie",
    },
    "en": {
        "qikeng": "pit",
        "budaodian": "non-conductive",
        "cahua": "scratch",
        "tufen": "powder bump",
        "tucengkailie": "coating crack",
    },
}


def find_image(fn):
    for d in IMG_DIRS:
        p = os.path.join(ROOT, d, fn)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(fn)


def pixel_sha256(path):
    """pixel_sha256 口径: PIL 打开 -> convert('RGB') -> tobytes() -> SHA256(不缩放)。"""
    with Image.open(path) as im:
        return hashlib.sha256(im.convert("RGB").tobytes()).hexdigest()


# ---------- 画前核对: 9 个文件的 pixel_sha256 ----------
def check_sha(step_lines):
    """重算 9 个文件的 pixel_sha256,写 check_sha_v3b.txt。返回是否全部一致。"""
    ref = {}
    for r in csv.DictReader(open(GROUPS, encoding="utf-8")):
        ref[r["filename"]] = (r["group_id"], r["pixel_sha256"])

    lines, ok = [], True
    A = lines.append
    A("check_sha_v3b.txt — 画前 pixel_sha256 核对")
    A("生成时间: %s (见 START.txt)" % open(os.path.join(D, "START.txt")).read().strip())
    A("口径: PIL 打开 -> convert('RGB') -> tobytes() -> SHA256(不缩放)")
    A("基准: round18_groups_20261006/groups.csv 的 pixel_sha256 列")
    A("=" * 78)

    for tag, gid, files in ROWS:
        A("")
        A("行 %s  组 %s  成员 %d 个" % (tag, gid, len(files)))
        calc = {}
        for f in files:
            try:
                path = find_image(f)
            except FileNotFoundError:
                A("  %-14s MISSING" % f)
                ok = False
                continue
            h = pixel_sha256(path)
            calc[f] = h
            g, want = ref[f]
            same_group = (g == gid)
            same_hash = (h == want)
            if not (same_group and same_hash):
                ok = False
            A("  %-14s %s" % (f, h))
            A("  %-14s   groups.csv group_id=%s(期望 %s) %s ; pixel_sha256 %s"
              % ("", g, gid, "一致" if same_group else "★不一致",
                 "一致" if same_hash else "★不一致"))
        uniq = sorted(set(calc.values()))
        A("  行内唯一 pixel_sha256 个数: %d  -> %s"
          % (len(uniq), "行内相同" if len(uniq) == 1 else "★行内不一致"))
        if len(uniq) != 1:
            ok = False

    A("")
    A("=" * 78)
    A("总结论: %s" % ("全部一致,可以继续画图" if ok else "★存在不一致,停止"))
    with open(os.path.join(D, "check_sha_v3b.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    step_lines.extend(lines)
    return ok


# ---------- 读框 ----------
boxes = {}
for r in csv.DictReader(open(BOXES, encoding="utf-8")):
    boxes.setdefault(r["filename"], []).append(r)


def rects_intersect(a, b):
    """两个矩形 (x0, y0, x1, y1) 是否相交(边上相接不算相交)。"""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return (ax0 < bx1) and (bx0 < ax1) and (ay0 < by1) and (by0 < ay1)


def make_panels(label_mode, panels_dir, step_lines):
    """逐张画框+类名,落盘 panels_*/ ; 返回 filename -> PIL.Image。"""
    drawn = {fn for _, _, files in ROWS for fn in files}
    classes = sorted({r["class_name"] for fn in drawn for r in boxes.get(fn, [])})
    assert len(classes) <= len(PALETTE), "类别数超过调色板色数,会出现同色"
    for c in classes:
        assert c in LABELS[label_mode], "未知类别名: %s" % c
    color_of = {c: PALETTE[i] for i, c in enumerate(classes)}
    text_color_of = {c: ("black" if color_of[c].upper() in BLACK_TEXT_BG else "white")
                     for c in classes}
    disp_of = {c: LABELS[label_mode][c] for c in classes}

    os.makedirs(panels_dir, exist_ok=True)
    font = ImageFont.truetype(FONT_PATH, FS)

    panels = {}
    n_right_aligned = 0
    n_moved_below = 0                       # 任务书 13b: 为避免重叠而改画到框下方的标签数
    still_overlapping = []                  # 任务书 13b: 移动后仍重叠的记录(只记录,不停止)
    for tag, gid, files in ROWS:
        for fn in files:
            im = Image.open(find_image(fn)).convert("RGB")
            orig_w, orig_h = im.size
            im = im.resize((PANEL_W, int(round(orig_h * SCALE))), Image.LANCZOS)
            dr = ImageDraw.Draw(im)
            placed = []                     # 任务书 13b: 本图已画标签的矩形 (x0,y0,x1,y1)
            for r in boxes.get(fn, []):
                x1, y1 = float(r["xmin"]), float(r["ymin"])
                x2, y2 = float(r["xmax"]), float(r["ymax"])
                # XML 原值可能存在 xmax<xmin / ymax<ymin,取 min/max 保证矩形合法
                l, rr = min(x1, x2), max(x1, x2)
                t, b = min(y1, y2), max(y1, y2)
                L, T = l * SCALE, t * SCALE
                R, B = rr * SCALE, b * SCALE
                cls = r["class_name"]
                col = color_of[cls]
                txt = disp_of[cls]
                dr.rectangle([L, T, R, B], outline=col, width=LW)
                # 类别名:框左上角外侧,背景为该类别颜色实心矩形;文字颜色按背景亮度
                tw = dr.textlength(txt, font=font)
                th = FS
                tx, ty = L, T - PAD - th          # 背景矩形左上角
                if ty < 0:                        # 上边界越界 -> 改画在框内左上角
                    ty = min(T + PAD, max(0, im.height - th))
                if tx + tw > im.width:
                    # 任务书 13 第 2 条(en 版类名更长): 右对齐到框的右上角,
                    # 即文字框右边缘贴 R(框右边); 仍越界则贴图片右边界
                    tx = min(R, im.width) - tw
                    n_right_aligned += 1
                tx = max(0, tx)                   # 左边界兜底
                if ty + th > im.height:           # 下边界越界则上移
                    ty = max(0, im.height - th)
                # ---- 任务书 13b 第 1 条: 标签避让 ----
                if any(rects_intersect((tx, ty, tx + tw, ty + th), p) for p in placed):
                    ty = B + PAD                  # 改画到框的下方外侧
                    if ty + th > im.height:       # 下方越界 -> 贴在框内左下角
                        ty = B - PAD - th
                    n_moved_below += 1
                    if any(rects_intersect((tx, ty, tx + tw, ty + th), p)
                           for p in placed):      # 移动后仍重叠: 只记录,不停止
                        still_overlapping.append(
                            "仍重叠: %s (obj_idx=%s, 类名=%s)" % (fn, r["obj_idx"], txt))
                dr.rectangle([tx, ty, tx + tw, ty + th], fill=col)
                dr.text((tx, ty), txt, font=font, fill=text_color_of[cls])
                placed.append((tx, ty, tx + tw, ty + th))
            panels[fn] = im
            im.save(os.path.join(panels_dir, "%s_%s.png" % (tag.strip("()"), fn)))
            print("panel %-14s %s -> %s ; 本图标签数 %d" % (fn, (orig_w, orig_h), im.size,
                                                        len(placed)))

    step_lines.append("")
    step_lines.append("labels=%s  右对齐修正的标签数: %d" % (label_mode, n_right_aligned))
    step_lines.append("labels=%s  为避免重叠改画到框下方的标签数 n_moved_below: %d"
                      % (label_mode, n_moved_below))
    if still_overlapping:
        step_lines.append("labels=%s  移动后仍重叠的记录 (%d 条):"
                          % (label_mode, len(still_overlapping)))
        step_lines.extend("  " + s for s in still_overlapping)
    else:
        step_lines.append("labels=%s  移动后仍重叠的记录: 无" % label_mode)
    step_lines.append("class->display: %s" % json.dumps(disp_of, ensure_ascii=False))
    print("\nlabels=%s  右对齐修正的标签数: %d" % (label_mode, n_right_aligned))
    print("labels=%s  改画到框下方的标签数 n_moved_below: %d" % (label_mode, n_moved_below))
    if still_overlapping:
        print("labels=%s  移动后仍重叠:" % label_mode)
        for s in still_overlapping:
            print("  " + s)
    else:
        print("labels=%s  移动后仍重叠: 无" % label_mode)
    print("class->display:", disp_of)
    return panels, color_of, text_color_of, disp_of


def compose(panels, out_pdf, out_png, out_preview):
    """拼图:4 行 x 4 列(第 0 列标签,第 1-3 列放图);PDF 按 PDF_DPI 嵌入。"""
    import matplotlib
    matplotlib.rcParams['pdf.fonttype'] = 42   # 任务书 13b 第 2 条: PDF 字体嵌入为 TrueType
    fig = plt.figure(figsize=(7.2, 7.4))
    plt.subplots_adjust(left=0.01, right=0.995, top=0.995, bottom=0.005,
                        wspace=0.03, hspace=0.04)
    gs = gridspec.GridSpec(4, 4, figure=fig, width_ratios=[0.06, 1, 1, 1])

    for i, (tag, gid, files) in enumerate(ROWS):
        ax_tag = fig.add_subplot(gs[i, 0])
        ax_tag.text(0.5, 0.5, tag, fontsize=11, ha="center", va="center")
        ax_tag.set_axis_off()

        for j in range(3):                       # 第 1-3 列
            ax = fig.add_subplot(gs[i, j + 1])
            if j < len(files):
                ax.imshow(panels[files[j]], interpolation="none")
            else:                                # (a)(b)(d) 第 3 列留空
                ax.set_axis_off()
                continue
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)

    fig.savefig(out_png, dpi=PNG_DPI)
    fig.savefig(out_pdf, dpi=PDF_DPI)
    fig.savefig(out_preview, dpi=PREVIEW_DPI)
    plt.close(fig)


def check_pdfimages(pdf, label_mode, expected_n, step_lines):
    """pdfimages -list 自检: 每张照片 width 应 >= 1280。结果写 check_pdfimages_v3b.txt。"""
    lines = []
    A = lines.append
    A("check_pdfimages_v3b.txt — PDF 内嵌照片分辨率自检 (任务书13 第1条 / 13b)")
    A("生成时间: %s (见 START_13b.txt)" % open(os.path.join(D, "START_13b.txt")).read().strip())
    A("命令: pdfimages -list <pdf>  (每张照片 width 应 >= 1280, 否则停止并报告)")
    A("=" * 78)
    A("pdfimages -list  %s   [labels=%s]" % (os.path.basename(pdf), label_mode))
    A("-" * 78)
    try:
        p = subprocess.run(["pdfimages", "-list", pdf], capture_output=True,
                           text=True, check=True)
    except Exception as e:                                    # noqa: BLE001
        A("★pdfimages 执行失败: %r" % (e,))
        A("总结论(labels=%s): ★失败" % label_mode)
        step_lines.extend(lines)
        return False
    raw = p.stdout.rstrip("\n")
    A(raw)

    rows = []
    for ln in raw.splitlines():
        f = ln.split()
        if len(f) >= 5 and f[0].isdigit() and f[1].isdigit() and f[2] == "image":
            rows.append((int(f[0]), int(f[1]), int(f[3]), int(f[4])))  # page,num,w,h
    widths = [w for _, _, w, _ in rows]
    ok = (len(rows) == expected_n) and all(w >= PANEL_W for w in widths)
    A("-" * 78)
    A("嵌入图片数: %d (期望 %d)" % (len(rows), expected_n))
    A("width 列表: %s" % widths)
    A("width 最小值: %s (要求 >= %d)" % (min(widths) if widths else "无", PANEL_W))
    A("总结论(labels=%s): %s" % (label_mode, "通过" if ok else "★不通过,停止并报告"))
    with open(os.path.join(D, "check_pdfimages_v3b.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    step_lines.extend(lines)
    return ok


def check_pdffonts(pdf, step_lines):
    """pdffonts <pdf> 输出写 check_pdffonts_v3b.txt。"""
    lines = []
    A = lines.append
    A("check_pdffonts_v3b.txt — pdffonts 输出 (任务书 13b 第 2 条: pdf.fonttype=42)")
    A("生成时间: %s (见 START_13b.txt)" % open(os.path.join(D, "START_13b.txt")).read().strip())
    A("命令: pdffonts %s" % os.path.basename(pdf))
    A("=" * 78)
    try:
        p = subprocess.run(["pdffonts", pdf], capture_output=True, text=True, check=True)
        A(p.stdout.rstrip("\n"))
    except Exception as e:                                    # noqa: BLE001
        A("★pdffonts 执行失败: %r" % (e,))
    with open(os.path.join(D, "check_pdffonts_v3b.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    step_lines.extend(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", choices=["en", "pinyin"], required=True,
                    help="类别名版本: en=英文类名, pinyin=XML 原文拼音(任务书 13 第 2 条)")
    args = ap.parse_args()
    label_mode = args.labels

    v3b = "_%s_v3b" % label_mode                # 任务书 13b: 输出加 _v3b 后缀, 不覆盖 v3 产物

    step_lines = []
    # ---- 画前核对: 不一致就停止 ----
    if not check_sha(step_lines):
        print("★pixel_sha256 存在不一致,停止,不画图。见 check_sha_v3b.txt")
        with open(os.path.join(D, "run_log%s.txt" % v3b), "w",
                  encoding="utf-8") as f:
            f.write("\n".join(step_lines) + "\n")
        return 2
    print("pixel_sha256 核对: 全部一致,继续画图")

    panels_dir = os.path.join(D, "panels%s" % v3b)
    panels, color_of, text_color_of, disp_of = make_panels(label_mode, panels_dir,
                                                           step_lines)

    out_pdf = os.path.join(D, "figure1%s.pdf" % v3b)
    out_png = os.path.join(D, "figure1%s.png" % v3b)
    out_prev = os.path.join(D, "figure1%s_preview.png" % v3b)
    compose(panels, out_pdf, out_png, out_prev)

    # ---- 保存后自检: pdfimages -list ----
    ok_pdf = check_pdfimages(out_pdf, label_mode, 9, step_lines)
    # ---- 保存后自检: pdffonts ----
    check_pdffonts(out_pdf, step_lines)

    with open(os.path.join(D, "figure1%s_size.json" % v3b), "w",
              encoding="utf-8") as f:
        json.dump({"figsize_in": [7.2, 7.4], "dpi_png": PNG_DPI,
                   "dpi_preview": PREVIEW_DPI, "dpi_pdf": PDF_DPI,
                   "panel_px": [PANEL_W, int(round(1920 * SCALE))],
                   "line_width_orig_px": 18, "font_orig_px": 150,
                   "line_width_panel_px": LW, "font_panel_px": FS,
                   "scale_from_original": SCALE, "labels": label_mode,
                   "imshow_interpolation": "none",
                   "pdf_fonttype": 42, "label_avoidance": "移框下方外侧",
                   "gridspec": "4x4 width_ratios=[0.06,1,1,1]",
                   "rows": [{"tag": t, "group": g, "files": fs} for t, g, fs in ROWS],
                   "class_color": color_of, "class_display": disp_of,
                   "class_text_color": text_color_of},
                  f, ensure_ascii=False, indent=2)

    with open(os.path.join(D, "run_log%s.txt" % v3b), "w",
              encoding="utf-8") as f:
        f.write("\n".join(step_lines) + "\n")

    print("\nfont=%s  FS(panel px)=%d  LW(panel px)=%d  scale=%s"
          % (FONT_PATH, FS, LW, SCALE))
    print("class->color:", color_of)
    print("class->text :", text_color_of)
    print("\n写出: %s" % out_pdf)
    print("写出: %s" % out_png)
    print("写出: %s" % out_prev)
    print("写出: %s" % panels_dir)
    if not ok_pdf:
        print("★pdfimages 自检不通过(有照片 width < %d),已停止并报告" % PANEL_W)
        return 3
    print("pdfimages 自检通过(9 张照片 width >= %d)" % PANEL_W)
    return 0


if __name__ == "__main__":
    sys.exit(main())
