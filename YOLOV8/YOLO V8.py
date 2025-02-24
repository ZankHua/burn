import os

# 在导入torch和ultralytics之前设置该环境变量
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from ultralytics.nn.modules.block import SwinTransformer
from ultralytics.nn.modules.head import Detect
from ultralytics import YOLO



# model = YOLO("yolo11n.yaml")
model = YOLO("my_swin_model.yaml",task="detect")


results = model.train(
    data="config.yaml",
    epochs=5,
    imgsz=640,
    device='mps'
)

