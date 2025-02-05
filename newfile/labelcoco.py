#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
示例：转换 LabelMe/Label Studio JSON 标注为 COCO 格式，并根据文件名中的
多重数据增强关键词（rotate, flip, brightness, contrast, color_jitter 等）
自动生成带后缀的 file_name (e.g. "img1001_copy9_rotate_flip.jpg").

使用方法：
1. 修改 main() 中的 data_dir, out_dir, subsets。
2. 运行: python labelme2coco_with_aug.py
3. 在 out_dir 下生成 coco_format/<subset>/annotations.json，内含带后缀的 file_name
   (如果在文件名中检测到多个关键词).
"""

import os
import json


def create_coco_dict():
    """创建空的 COCO 字典结构。"""
    return {
        "images": [],
        "annotations": [],
        "categories": []
    }


def load_labelme_json(json_path):
    """
    读取单个 LabelMe / Label Studio 风格的 JSON 标注文件。
    返回一个字典，例如：
      {
        "shapes": [...],
        "imagePath": "...",
        "imageWidth": ...,
        "imageHeight": ...
      }
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def detect_aug_suffix(stem):
    """
    从文件名 (不含扩展名的 stem) 中，检测多种增强关键词。
    若发现 'rotate', 'flip', 'brightness', 'contrast', 'color_jitter'，则组合成后缀。
    例如：
      stem = "img1001_copy9_rotate_flip" => "_rotate_flip"
      stem = "img2002_brightness_contrast_color_jitter" => "_brightness_contrast_color_jitter"

    你可根据需要添加更多关键词或改写规则。
    """
    keywords = ['rotate', 'flip', 'brightness', 'contrast', 'color_jitter']
    found = []
    for kw in keywords:
        if kw in stem:
            found.append(kw)

    if found:
        return "_" + "_".join(found)  # e.g. "_rotate_flip_color_jitter"
    else:
        return ""


def convert_labelme_to_coco(labelme_json, image_id, anno_start_id, label2id):
    """
    将单个 labelme_json 转成 COCO 所需要的 (image_dict, annotations_list, new_anno_id, label2id)。

    在这里，自动检测 file_name 中的增强关键词并添加后缀。
    """
    file_path = labelme_json["imagePath"]  # e.g. "/path/to/img1001_copy9_rotate_flip.jpg"
    base_name = os.path.basename(file_path)  # "img1001_copy9_rotate_flip.jpg"

    # 分离出文件名与扩展名
    stem, ext = os.path.splitext(base_name)  # stem="img1001_copy9_rotate_flip", ext=".jpg"

    # 检测关键字后缀
    aug_suffix = detect_aug_suffix(stem)

    # 如果已经有 'copy9_rotate_flip' 之类，那么 stem 中本身就包含这些子串；
    # 你可以选择 “保留原名” 或 “清理后再次拼接”。
    #
    # 这里示例：保留原有 stem，同时再追加一次 aug_suffix 可能会出现重复；
    # 因此一般只要返回 base_name 本身即可。
    #
    # 如果你想将文件名改为 "img1001_copy9" + "_rotate_flip" => "img1001_copy9_rotate_flip.jpg"
    # 可以写：
    # new_stem = stem
    # # 如果想去掉多余再拼:
    # # for kw in ['rotate','flip','brightness','contrast','color_jitter']:
    # #     new_stem = new_stem.replace(kw, "")
    # # new_stem = new_stem.strip("_")
    # # new_stem = new_stem + aug_suffix
    # # new_file_name = new_stem + ext
    #
    # 这里示例中直接用 base_name 做最终file_name:
    new_file_name = base_name

    width = labelme_json["imageWidth"]
    height = labelme_json["imageHeight"]

    image_dict = {
        "file_name": new_file_name,
        "height": height,
        "width": width,
        "id": image_id
    }

    annotations_list = []
    anno_id = anno_start_id

    shapes = labelme_json.get("shapes", [])
    for shape in shapes:
        label = shape["label"]  # 类别名
        points = shape["points"]  # [ [x1,y1], [x2,y2], ... ]

        # 若该 label 尚未出现过，则分配新的 category_id
        if label not in label2id:
            label2id[label] = len(label2id) + 1
        category_id = label2id[label]

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        w = xmax - xmin
        h = ymax - ymin
        area = w * h

        # segmentation
        poly = []
        for (x, y) in points:
            poly.append(x)
            poly.append(y)
        segmentation = [poly]

        anno_dict = {
            "id": anno_id,
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": segmentation,
            "bbox": [xmin, ymin, w, h],
            "area": area,
            "iscrowd": 0
        }
        annotations_list.append(anno_dict)
        anno_id += 1

    return image_dict, annotations_list, anno_id, label2id


def main():
    """
    主函数：你可在此修改 data_dir, out_dir, subsets。
    data_dir 里应有 train/、valid/、test/，每个子文件夹包含若干 JSON 标注文件（LabelMe/LabelStudio格式）。
    """
    data_dir = "/Users/zhankanghua/Desktop/毕业论文/数据/数据强化图片+标签_副本4/数据标签_强化复制"  # 修改成你的实际路径
    out_dir = "/Users/zhankanghua/Desktop/毕业论文/数据/数据强化图片+标签_副本4/转换后的COCO"
    subsets = ["train", "valid", "test"]

    coco_root = os.path.join(out_dir, "coco_format")
    os.makedirs(coco_root, exist_ok=True)

    for subset in subsets:
        subset_dir = os.path.join(data_dir, subset)
        if not os.path.isdir(subset_dir):
            print(f"子文件夹不存在或不是目录，跳过：{subset_dir}")
            continue

        # 创建空的 COCO 结构
        coco_dict = create_coco_dict()
        label2id = {}
        image_id = 0
        anno_id = 0

        # 遍历子文件夹下所有 JSON
        all_jsons = []
        for root, dirs, files in os.walk(subset_dir):
            for f in files:
                if f.lower().endswith(".json"):
                    fullpath = os.path.join(root, f)
                    all_jsons.append(fullpath)
        all_jsons.sort()

        # 依次处理
        for jf in all_jsons:
            labelme_data = load_labelme_json(jf)

            image_dict, annos, next_anno_id, label2id = convert_labelme_to_coco(
                labelme_data,
                image_id,
                anno_start_id=anno_id,
                label2id=label2id
            )
            coco_dict["images"].append(image_dict)
            coco_dict["annotations"].extend(annos)

            image_id += 1
            anno_id = next_anno_id

        # 构造 categories
        for label, cid in label2id.items():
            cat_info = {
                "id": cid,
                "name": label,
                "supercategory": "none"
            }
            coco_dict["categories"].append(cat_info)

        # 写到 out_dir/coco_format/<subset>/annotations.json
        subset_out_dir = os.path.join(coco_root, subset)
        os.makedirs(subset_out_dir, exist_ok=True)
        out_json_path = os.path.join(subset_out_dir, "annotations.json")

        with open(out_json_path, 'w', encoding='utf-8') as fw:
            json.dump(coco_dict, fw, indent=2, ensure_ascii=False)

        print(f"[{subset}] COCO 标注已生成: {out_json_path}")

    print("处理完毕！")


if __name__ == "__main__":
    main()
