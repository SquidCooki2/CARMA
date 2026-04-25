# Gemini Context: Hand Localization Pipeline

This project implements a real-time 3D hand localization system using a multi-camera setup (typically 3 cameras). It leverages deep learning for 2D detection and geometric triangulation for 3D positioning.

## Project Overview
- **Purpose:** Locate the 3D center of a hand within a defined "box" volume using three Logitech C920 cameras.
- **Architecture:**
  - **Detection:** YOLOv8n (Nano) is fine-tuned on the EgoHands dataset to detect hands in 2D image frames.
  - **Geometry:** Direct Linear Transform (DLT) using Singular Value Decomposition (SVD) triangulates the 2D centers from multiple views into a single 3D point.
  - **Coordinate System:** World space is defined in mm, with the **Z-axis as UP**. Cameras are positioned according to CAD coordinates and use a "Look-At" algorithm to calculate orientation toward the origin `[0, 0, 0]`.

## Core Technologies
- **Python 3.x**
- **Ultralytics YOLOv8:** Hand detection model.
- **OpenCV:** Image acquisition and basic processing.
- **NumPy & SciPy:** Linear algebra for triangulation (SVD).
- **PyYAML:** Configuration for camera parameters and datasets.

## Key Files & Directories
- `configs/camera_params.yaml`: Contains camera intrinsics (fx, fy, cx, cy, distortion) and extrinsics (position in mm).
- `scripts/localization/localize_hand.py`: Main `HandLocalizer` class and 3D triangulation entry point.
- `scripts/localization/geometry_utils.py`: Mathematical foundation (projection matrices, DLT, bounding box utilities).
- `scripts/localization/train_hand_detector.py`: Training script for fine-tuning YOLOv8n.
- `data/egohands.yaml`: Dataset configuration for YOLO training.
- `models/`: Storage for fine-tuned `.pt` weights.

## Building and Running

### Environment Setup
1. Activate the environment: `conda activate hand-localization`
2. Or install dependencies: `pip install -r requirements.txt`

### Dataset Preparation
1. Download EgoHands dataset to `data/egohands/`.
2. (Optional) Run `python scripts/localization/relabel_data.py` to ensure all hands are mapped to class `0`.

### Training
Execute the training script to generate fine-tuned weights:
```bash
python scripts/localization/train_hand_detector.py
```

### 3D Localization
To run a simulation/math check:
```bash
python scripts/localization/localize_hand.py
```
To use in a real-time pipeline:
```python
from scripts.localization.localize_hand import HandLocalizer
localizer = HandLocalizer("configs/camera_params.yaml", "models/hand_detector/best.pt")
# frames = [cam1_img, cam2_img, cam3_img]
# pos_3d = localizer.localize_3d(frames)
```

## Development Conventions
- **Geometry:** Always assume Z-up world coordinates unless specified otherwise.
- **Camera Models:** Uses the standard OpenCV pinhole camera model.
- **Config First:** Modifications to camera positions or intrinsics should be done in `configs/camera_params.yaml`.
- **Modularity:** Keep geometric math in `geometry_utils.py` and detection logic in `localize_hand.py`.
