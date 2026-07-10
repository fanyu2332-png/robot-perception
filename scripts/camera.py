from pathlib import Path
import cv2
import json

from ultralytics import YOLO
from export_json import results_to_json

# ==========================
# 找到项目根目录
# ==========================
project_root = Path(__file__).resolve().parent.parent

# ==========================
# 加载模型
# ==========================
model = YOLO(project_root / "models" / "yoloe-26x-seg.pt")

# 告诉 YOLOE 要识别哪些类别
model.set_classes([
    "person",
    "bottle",
    "chair",
    "laptop"
])

# ==========================
# 打开摄像头
# ==========================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

print("摄像头已启动，按 q 退出。")

frame_id = 0

while True:

    # 读取一帧
    ret, frame = cap.read()

    if not ret:
        break

    # YOLO 检测
    results = model.predict(
        source=frame,
        conf=0.25,
        verbose=False
    )

    # 画检测框
    annotated_frame = results[0].plot()

    # 转换为 JSON
    data = {
        "frame_id": frame_id,
        "objects": results_to_json(results)
    }

    # 保存最新 JSON（每一帧都会覆盖）
    with open(project_root / "output.json", "w") as f:
        json.dump(data, f, indent=4)

    # 同时打印到终端（方便调试）
    print(json.dumps(data, indent=2))

    # 显示画面
    cv2.imshow("YOLOE Camera", annotated_frame)

    frame_id += 1

    # 按 q 退出
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==========================
# 释放资源
# ==========================
cap.release()
cv2.destroyAllWindows()

print("程序结束。")