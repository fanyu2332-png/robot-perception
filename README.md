# Robot Perception

A robot vision perception pipeline built on [Ultralytics YOLOE](https://docs.ultralytics.com/models/yoloe/). It uses the **YOLOE-26x-seg** open-vocabulary instance segmentation model to detect objects via text prompts (`set_classes`). The project supports static image inference and real-time webcam detection, exporting structured JSON results for downstream robot systems.

## Features

- **Open-vocabulary detection** — Detect new object categories without retraining; just update the class name list
- **Instance segmentation** — YOLOE-26x-seg outputs bounding boxes and segmentation masks
- **Static image inference** — Run detection on a single image and save visualized results
- **Real-time webcam detection** — Process live camera frames and continuously update JSON output
- **Structured JSON export** — Convert detections (class, confidence, bounding box) into a machine-readable format

## Project Structure

```
robot-perception/
├── models/
│   └── yoloe-26x-seg.pt      # YOLOE segmentation model weights (gitignored)
├── scripts/
│   ├── test_model.py         # Verify model loading
│   ├── detect.py             # Static image detection
│   ├── camera.py             # Real-time webcam detection
│   └── export_json.py        # YOLO results → JSON converter
├── images/                   # Sample test images
│   ├── office.jpg
│   └── images.jpeg
├── runs/segment/             # YOLO inference output (auto-generated, gitignored)
├── output.json               # Latest detection results (written by camera.py, gitignored)
├── mobileclip2_b.ts          # MobileCLIP text encoder (YOLOE internal dependency)
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.11+
- Conda virtual environment (recommended)
- A working camera device for real-time detection

## Installation

### 1. Create and activate a Conda environment

```bash
conda create -n robot-perception python=3.11 -y
conda activate robot-perception
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> `ultralytics` installs PyTorch and related packages automatically. For GPU acceleration, install the appropriate CUDA build of PyTorch from the [official PyTorch guide](https://pytorch.org/get-started/locally/) before installing the remaining packages.

### 3. Verify model files

Ensure `models/yoloe-26x-seg.pt` is present. If missing, Ultralytics will download it on first run. You can also download the weights manually and place them in the `models/` directory.

## Usage

Run all scripts from the `scripts/` directory:

```bash
cd scripts
```

### Test model loading

Verify that the model weights load correctly:

```bash
python test_model.py
```

Expected output:

```
Model loaded successfully!
```

### Static image detection

Run object detection on `images/office.jpg`. Results are saved to `runs/segment/` and JSON is printed to the terminal:

```bash
python detect.py
```

Default detection classes: `person`, `chair`, `table`, `laptop`, `backpack`, `bottle`.

- **Change classes:** edit the list in `model.set_classes([...])` inside `detect.py`
- **Change input image:** update the `source` path in `detect.py`

### Real-time webcam detection

Start the camera, detect in real time, and write the latest results to `output.json` in the project root:

```bash
python camera.py
```

- Press **q** to quit
- Each frame overwrites `output.json` and prints JSON to the terminal
- Default detection classes: `person`, `bottle`, `chair`, `laptop`

## Output Format

`export_json.py` converts YOLO inference results into the following JSON structure:

```json
{
    "frame_id": 35,
    "objects": [
        {
            "class": "person",
            "confidence": 0.887,
            "bbox": {
                "x1": 1409.5,
                "y1": 837.5,
                "x2": 1735.1,
                "y2": 1077.9
            }
        }
    ]
}
```

| Field | Description |
|-------|-------------|
| `frame_id` | Frame index (only present in `camera.py` output) |
| `objects` | List of detected objects |
| `class` | Object class name |
| `confidence` | Confidence score (0–1) |
| `bbox` | Bounding box in pixels (`x1/y1` = top-left, `x2/y2` = bottom-right) |

## Custom Detection Classes

YOLOE supports open-vocabulary detection through text prompts—no retraining required. Update `set_classes` in the relevant script:

```python
model.set_classes([
    "person",
    "bottle",
    "chair",
    "laptop"
])
```

English class names work best because the model uses an English text encoder. Re-run the script after changing classes.

## Models

| File | Description |
|------|-------------|
| `yoloe-26x-seg.pt` | YOLOE-26X instance segmentation model with text/visual prompt support |
| `mobileclip2_b.ts` | MobileCLIP text encoder used by YOLOE to generate class embeddings |

YOLOE-26x is the highest-accuracy variant in the series, ideal when detection quality is the priority. For faster inference, consider smaller models such as `yoloe-26s-seg.pt` or `yoloe-26m-seg.pt`.

## FAQ

**Q: Camera won't open?**

Ensure the system has granted camera permissions and no other application is using the camera. `camera.py` defaults to device index `0`. If you have multiple cameras, change the index in `cv2.VideoCapture(0)`.

**Q: Target objects not detected?**

- Lower the confidence threshold (e.g. `conf=0.15`)
- Verify class names are spelled correctly and use common English terms
- Check lighting and camera angle

**Q: First run is slow?**

Ultralytics performs initialization and may download weights on first load. This is normal; subsequent runs are significantly faster.

**Q: GPU not being used?**

Install a CUDA-enabled PyTorch build and verify with:

```python
import torch
print(torch.cuda.is_available())
```

## References

- [Ultralytics YOLOE Documentation](https://docs.ultralytics.com/models/yoloe/)
- [Ultralytics YOLO26 Documentation](https://docs.ultralytics.com/models/yolo26/)
- [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)
