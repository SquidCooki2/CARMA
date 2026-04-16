# Hand Localization Pipeline (3 Cameras)

This project implements a real-time 3D hand localization system using three Logitech C920 cameras in a box setup. It uses YOLOv8 for hand detection and Direct Linear Transform (DLT) for 3D triangulation.

## Project Structure
- `configs/`: Camera intrinsic and CAD extrinsic parameters.
- `data/`: Dataset storage and YAML configuration for YOLO.
- `models/`: Fine-tuned YOLOv8 weights.
- `scripts/`: Implementation of detection and triangulation logic.
- `notebooks/`: For visualization and testing.

## Getting Started

### 1. Environment Setup
Activate the pre-configured conda environment:
```bash
conda activate hand-localization
```
*(If the environment is not found, install using `pip install -r requirements.txt`)*

### 2. Dataset Preparation
Download the **EgoHands** dataset in YOLO format (e.g., from Roboflow) and place it in `data/egohands/`.
The structure should be:
```
data/egohands/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

### 3. Training the Hand Detector
Run the training script to fine-tune YOLOv8n on the hand dataset:
```bash
cd scripts
python train_hand_detector.py
```
This will save the best model weights to `models/hand_detector/`.

### 4. Running Localization (Simulation/Math Check)
Verify the 3D triangulation math using the CAD coordinates (Z-up system):
```bash
cd scripts
python localize_hand.py
```
This should output a triangulated point near `[0, 0, 0]` for a target at the center of the box.

### 5. Running Real-Time Localization
Update `configs/camera_params.yaml` with any final adjustments. Use the `HandLocalizer` class:
```python
from scripts.localize_hand import HandLocalizer

# Use model_path=None for simulation mode
localizer = HandLocalizer("../configs/camera_params.yaml", "../models/hand_detector/best.pt")

# frames = [cam1_img, cam2_img, cam3_img]
# pos_3d = localizer.localize_3d(frames)
# print(f"Hand 3D Position (mm): {pos_3d}")
```

## Technical Details

### Hand Detection (ML)
We use **YOLOv8n** (Nano) for high-speed inference. It is fine-tuned to detect hands and find the center of the bounding box as the 2D point of interest.

### 3D Localization (Geometry)
We use the **Direct Linear Transform (DLT)** algorithm to triangulate the 3D position of the hand center from three camera views. 
- **World Space:** Defined by your CAD coordinates (Z-up).
- **Camera Orientation:** Automatically calculated using a "Look-At" algorithm, assuming all cameras point at the origin `[0, 0, 0]`.
- **Triangulation:** Uses Singular Value Decomposition (SVD) for a robust least-squares solution.
