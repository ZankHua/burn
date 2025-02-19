import streamlit as st
from PIL import Image, ImageDraw
import json
import openai



openai.api_key = "看备忘录"

# ========== 你的 COCO 标注文件路径 ==============
COCO_ANN_PATH = "/Users/zhankanghua/Desktop/毕业论文/数据/新label_me/转换后的COCO/coco_format/train/annotations.json"

# ========== 映射 cat_id => 烧伤等级 ==========
BURN_LABEL_MAP = {
    0: "not_burn",
    1: "1st_degree",
    2: "2nd_degree",
    3: "3rd_degree"
}

@st.cache_data
def load_coco_annotations(json_path):
    """
    读取COCO格式 JSON，并返回其数据结构
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def get_bboxes_and_cats(coco_data, filename):

    images = coco_data["images"]
    annos = coco_data["annotations"]

    # 找 image_id
    image_id = None
    for img_info in images:
        if img_info["file_name"] == filename:
            image_id = img_info["id"]
            break
    if image_id is None:
        return []

    results = []
    for anno in annos:
        if anno["image_id"] == image_id:
            x, y, w, h = anno["bbox"]
            bbox = (x, y, x + w, y + h)
            cat_id = anno["category_id"]
            results.append((bbox, cat_id))
    return results

def draw_coco_bboxes(image_pil, bboxes_and_cats):
    """
    draw out bounding boxes，and written category id
    """
    draw = ImageDraw.Draw(image_pil)
    for (xmin, ymin, xmax, ymax), cat_id in bboxes_and_cats:
        # 映射 cat_id => label
        label = BURN_LABEL_MAP.get(cat_id, f"unknown_{cat_id}")
        draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=3)
        draw.text((xmin, ymin - 10), label, fill="red")
    return image_pil

def generate_medical_advice(severity_label):

    prompt = f"Patient has a {severity_label} burn. Provide medical advice."

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
             "content": "You are a helpful medical assistant. "
                        "Provide detailed, accurate medical advice，"
                        "remember to highlight key informations,"
                        "remember if the burn level is not burn "
                        "give some potential skin affection or skin problem list and solutions."},
            {"role": "user", "content": prompt}
        ]
    )
    advice = response["choices"][0]["message"]["content"].strip()
    return advice

def main():
    st.title("Burn Detection Demo (COCO BBoxes + LLM)")

    # 1) 加载 COCO 标注
    coco_data = load_coco_annotations(COCO_ANN_PATH)
    st.write("upload image， bounding boxes will be provided。")

    # 2) 上传图像
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # 转 PIL
        image_pil = Image.open(uploaded_file).convert("RGB")
        st.image(image_pil, caption="Original Image", use_container_width=True)

        # 3) 查找 bounding boxes
        filename = uploaded_file.name
        bboxes_and_cats = get_bboxes_and_cats(coco_data, filename)
        if len(bboxes_and_cats) == 0:
            st.write("No bounding boxes found in COCO for this filename.")
        else:
            st.write(f"Found {len(bboxes_and_cats)} bounding boxes from COCO.")
            drawn_img = image_pil.copy()
            drawn_img = draw_coco_bboxes(drawn_img, bboxes_and_cats)
            st.image(drawn_img, caption="COCO bounding boxes + burn label", use_container_width=True)

            # 4) 如果多个box, 这里仅取第一个 cat_id => burn severity
            _, cat_id_first = bboxes_and_cats[0]
            severity_label = BURN_LABEL_MAP.get(cat_id_first, "unknown")
            st.write(f"Detected burn severity (from first box): {severity_label}")

            # 5) 调用 LLM 生成建议
            try:
                advice = generate_medical_advice(severity_label)
                st.write("LLM-based Advice:")
                st.write(advice)
            except Exception as e:
                st.error(f"OpenAI API fail: {e}")

if __name__ == "__main__":
    main()
