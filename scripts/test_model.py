from pathlib import Path
from ultralytics import YOLO

project_root = Path(__file__).resolve().parent.parent
model_path = project_root / "models" / "yoloe-26x-seg.pt"

model = YOLO(model_path)

print("模型加载成功！")