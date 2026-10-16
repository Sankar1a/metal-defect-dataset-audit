"""

GC10-DET 数据集准备脚本

功能：VOC XML → YOLO txt 格式转换 + 8:2 train/val 划分

用法：python prepare_gc10.py

"""



import os

import xml.etree.ElementTree as ET

import shutil

import random

from pathlib import Path

from collections import defaultdict



# ======================== 配置 ========================

RAW_DIR = "/home/DSJ44/sr_project/gc10_raw"

OUTPUT_DIR = "/home/DSJ44/sr_project/gc10_yolo"

TRAIN_RATIO = 0.8

RANDOM_SEED = 0



# GC10-DET 10 类缺陷，类名 → YOLO 类别编号

CLASS_MAP = {

    "1_chongkong": 0,   # 冲孔

    "2_hanfeng": 1,      # 焊缝

    "3_yueyawan": 2,     # 月牙弯

    "4_shuiban": 3,      # 水斑

    "5_youban": 4,       # 油斑

    "6_siban": 5,        # 丝斑

    "7_yiwu": 6,         # 异物

    "8_yahen": 7,        # 压痕

    "9_zhehen": 8,       # 折痕

    "10_yaozhe": 9,      # 腰折

}

# 有些标注可能带 'd' 后缀，如 10_yaozhed

CLASS_MAP["10_yaozhed"] = 9



CLASS_NAMES = [

    "chongkong", "hanfeng", "yueyawan", "shuiban", "youban",

    "siban", "yiwu", "yahen", "zhehen", "yaozhe"

]

# ======================================================





def parse_voc_xml(xml_path):

    """解析 VOC XML，返回 (图片文件名, 图片宽高, bbox列表)"""

    tree = ET.parse(xml_path)

    root = tree.getroot()



    filename = root.find("filename").text

    size = root.find("size")

    w = int(size.find("width").text)

    h = int(size.find("height").text)



    boxes = []

    for obj in root.findall("object"):

        name = obj.find("name").text.strip()

        if name not in CLASS_MAP:

            print(f"  [WARNING] 未知类别 '{name}'，跳过 (文件: {xml_path})")

            continue

        cls_id = CLASS_MAP[name]



        bndbox = obj.find("bndbox")

        xmin = float(bndbox.find("xmin").text)

        ymin = float(bndbox.find("ymin").text)

        xmax = float(bndbox.find("xmax").text)

        ymax = float(bndbox.find("ymax").text)



        # 转 YOLO 格式：cx, cy, bw, bh (归一化)

        cx = (xmin + xmax) / 2.0 / w

        cy = (ymin + ymax) / 2.0 / h

        bw = (xmax - xmin) / w

        bh = (ymax - ymin) / h



        # 边界检查

        cx = max(0, min(1, cx))

        cy = max(0, min(1, cy))

        bw = max(0, min(1, bw))

        bh = max(0, min(1, bh))



        boxes.append((cls_id, cx, cy, bw, bh))



    return filename, w, h, boxes





def find_image(filename, image_dirs):

    """在所有子文件夹中查找图片"""

    for d in image_dirs:

        img_path = os.path.join(d, filename)

        if os.path.exists(img_path):

            return img_path

    return None





def main():

    random.seed(RANDOM_SEED)



    # 创建输出目录

    for split in ["train", "val"]:

        os.makedirs(os.path.join(OUTPUT_DIR, "images", split), exist_ok=True)

        os.makedirs(os.path.join(OUTPUT_DIR, "labels", split), exist_ok=True)



    # 收集所有图片子文件夹（1~10）

    image_dirs = []

    for item in os.listdir(RAW_DIR):

        item_path = os.path.join(RAW_DIR, item)

        if os.path.isdir(item_path) and item not in ["lable", "label"]:

            image_dirs.append(item_path)

    print(f"找到 {len(image_dirs)} 个图片子文件夹: {[os.path.basename(d) for d in sorted(image_dirs)]}")



    # 收集所有 XML 标注

    label_dir = os.path.join(RAW_DIR, "lable")  # 注意原数据集拼写

    if not os.path.exists(label_dir):

        label_dir = os.path.join(RAW_DIR, "label")

    if not os.path.exists(label_dir):

        print("[ERROR] 找不到标注文件夹 (lable 或 label)")

        return



    xml_files = sorted([f for f in os.listdir(label_dir) if f.endswith(".xml")])

    print(f"找到 {len(xml_files)} 个 XML 标注文件")



    # 解析所有标注，匹配图片

    valid_samples = []  # (xml_path, img_path, filename_stem)

    skipped = 0

    class_count = defaultdict(int)



    for xml_file in xml_files:

        xml_path = os.path.join(label_dir, xml_file)

        try:

            filename, w, h, boxes = parse_voc_xml(xml_path)

        except Exception as e:

            print(f"  [WARNING] 解析失败: {xml_file}, 错误: {e}")

            skipped += 1

            continue



        if not boxes:

            skipped += 1

            continue



        img_path = find_image(filename, image_dirs)

        if img_path is None:

            # 尝试从 XML 文件名推断

            stem = os.path.splitext(xml_file)[0]

            for ext in [".jpg", ".jpeg", ".png", ".bmp"]:

                img_path = find_image(stem + ext, image_dirs)

                if img_path:

                    break



        if img_path is None:

            print(f"  [WARNING] 找不到图片: {filename} (XML: {xml_file})")

            skipped += 1

            continue



        valid_samples.append((xml_path, img_path, boxes))

        for cls_id, *_ in boxes:

            class_count[cls_id] += 1



    print(f"\n有效样本: {len(valid_samples)}, 跳过: {skipped}")

    print("各类别标注框数量:")

    for cls_id in sorted(class_count.keys()):

        print(f"  {cls_id} ({CLASS_NAMES[cls_id]}): {class_count[cls_id]}")



    # 随机划分 train/val

    random.shuffle(valid_samples)

    split_idx = int(len(valid_samples) * TRAIN_RATIO)

    train_samples = valid_samples[:split_idx]

    val_samples = valid_samples[split_idx:]

    print(f"\n划分: train={len(train_samples)}, val={len(val_samples)}")



    # 写入文件

    for split, samples in [("train", train_samples), ("val", val_samples)]:

        for xml_path, img_path, boxes in samples:

            # 用图片原始文件名（避免重名加上哈希）

            img_name = os.path.basename(img_path)

            stem = os.path.splitext(img_name)[0]

            ext = os.path.splitext(img_name)[1]



            # 复制图片

            dst_img = os.path.join(OUTPUT_DIR, "images", split, img_name)

            # 处理重名

            if os.path.exists(dst_img):

                stem = stem + "_" + os.path.basename(os.path.dirname(xml_path))[:8]

                img_name = stem + ext

                dst_img = os.path.join(OUTPUT_DIR, "images", split, img_name)



            shutil.copy2(img_path, dst_img)



            # 写 YOLO 标签

            label_name = stem + ".txt"

            dst_label = os.path.join(OUTPUT_DIR, "labels", split, label_name)

            with open(dst_label, "w") as f:

                for cls_id, cx, cy, bw, bh in boxes:

                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")



    # 生成 YAML 配置文件

    yaml_content = f"""# GC10-DET dataset config for YOLOv8

path: {OUTPUT_DIR}

train: images/train

val: images/val



nc: 10

names:

  0: chongkong

  1: hanfeng

  2: yueyawan

  3: shuiban

  4: youban

  5: siban

  6: yiwu

  7: yahen

  8: zhehen

  9: yaozhe

"""

    yaml_path = os.path.join(OUTPUT_DIR, "gc10det.yaml")

    with open(yaml_path, "w") as f:

        f.write(yaml_content)



    # 同时在项目根目录放一份

    shutil.copy2(yaml_path, "/home/DSJ44/sr_project/gc10det.yaml")



    print(f"\n✅ 完成！")

    print(f"   数据集目录: {OUTPUT_DIR}")

    print(f"   YAML 配置:  /home/DSJ44/sr_project/gc10det.yaml")

    print(f"   train 图片:  {len(train_samples)}")

    print(f"   val 图片:    {len(val_samples)}")





if __name__ == "__main__":

    main()
