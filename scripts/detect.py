from pathlib import Path
from ultralytics import YOLO

# 找到项目根目录
project_root = Path(__file__).resolve().parent.parent

# 加载模型，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，
model = YOLO(project_root / "models" / "yoloe-26x-seg.pt")

# 告诉 YOLOE 今天要识别哪些物体
model.set_classes([
    "person",
    "chair",
    "table",
    "laptop",
    "backpack",
    "bottle"
])

# 开始检测
results = model.predict(
    source=project_root / "images" / "office.jpg",
    save=True,
    conf=0.25
)
import export_json
import export_json
import json

print(export_json.__file__)

data = export_json.results_to_json(results)

print(json.dumps(data, indent=4))

print("检测完成！")