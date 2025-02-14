import os
import json


def update_coco_filenames(ann_file: str, img_dir: str, out_ann_file: str):
    """
    批量修改 COCO 标注文件中 images[].file_name 使其与磁盘实际文件名匹配

    :param ann_file:     原始 COCO 标注文件路径 (e.g. ".../train/annotations.json")
    :param img_dir:      图片所在目录，COCO 格式下通常所有图片都放这里 (e.g. ".../train/images")
    :param out_ann_file: 输出修改后标注文件的路径 (e.g. ".../train/annotations_updated.json")
    """

    # 读取原始标注文件
    with open(ann_file, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    images = coco_data.get('images', [])
    all_files = os.listdir(img_dir)

    # 对每个 image，根据其原 file_name 的“前缀”来匹配可能带后缀的实际文件
    for img_info in images:
        old_name = img_info['file_name']  # e.g. "img1388_jpg.rf.1dcd20b70a87f256a868449197825930.jpg"
        stem, ext = os.path.splitext(old_name)  # stem = "img1388_jpg.rf.1dcd20b70a87f256a868449197825930", ext = ".jpg"

        # 在 img_dir 中寻找文件名以 stem 开头，并以 .jpg 结尾的文件
        candidates = []
        for f in all_files:
            if f.lower().endswith('.jpg') and f.startswith(stem):
                candidates.append(f)

        if len(candidates) == 1:
            # 如果找到且只有一个匹配文件，则更新 file_name
            new_name = candidates[0]
            img_info['file_name'] = new_name
        elif len(candidates) == 0:
            print(f"[Warning] No match found for {old_name} in {img_dir}, keep old name.")
        else:
            # 如果匹配到多个文件，则报警，但不修改 file_name
            print(
                f"[Warning] Multiple matches found for {old_name} in {img_dir}, keep old name.\nCandidates: {candidates}")

    # 将修改结果写回新的标注文件
    with open(out_ann_file, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, ensure_ascii=False, indent=2)

    print(f"Updated annotation file saved to {out_ann_file}")


if __name__ == '__main__':
    # 你只需修改下面三行，确保路径正确
    # 1) ann_file_path:  COCO 标注文件路径（需修改为你的路径）
    # 2) image_dir:      存放对应图片的文件夹路径
    # 3) out_ann_file_path: 修改后标注文件的输出路径

    ann_file_path = "/Users/zhankanghua/Desktop/毕业论文/数据/数据强化图片+标签_副本/转换后的COCO/coco_format/test/annotations.json"
    image_dir = "/Users/zhankanghua/Desktop/毕业论文/数据/数据强化图片+标签_副本/转换后的COCO/coco_format/test/image"
    out_ann_file_path = "/Users/zhankanghua/Desktop/毕业论文/数据/数据强化图片+标签_副本/转换后的COCO/coco_format/test/annotations.json"

    update_coco_filenames(ann_file_path, image_dir, out_ann_file_path)
