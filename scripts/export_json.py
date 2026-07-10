import json


def results_to_json(results):
    output = []

    result = results[0]

    for box in result.boxes:

        cls_id = int(box.cls[0])
        class_name = result.names[cls_id]

        confidence = float(box.conf[0])

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        output.append({
            "class": class_name,
            "confidence": round(confidence, 3),
            "bbox": {
                "x1": round(x1, 1),
                "y1": round(y1, 1),
                "x2": round(x2, 1),
                "y2": round(y2, 1)
            }
        })

    return output