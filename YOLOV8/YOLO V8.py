import os
# 在导入torch和ultralytics之前设置该环境变量
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
from ultralytics import YOLO

model = YOLO("yolo11n.yaml")

results = model.train(
    data="config.yaml",
    epochs=5,
    imgsz=640,
    device='mps'
)
