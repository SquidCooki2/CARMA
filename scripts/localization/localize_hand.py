import os
import yaml
import numpy as np
from ultralytics import YOLO
from geometry_utils import get_look_at_projection_matrix, triangulate_n_views, get_hand_center

class HandLocalizer:
    def __init__(self, config_path, model_path):
        # 1. Load Camera Config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # 2. Load ML Model
        # If model_path is None, it runs in simulation mode (no detection)
        self.model = YOLO(model_path) if model_path else None
        
        # 3. Pre-calculate Projection Matrices for all cameras
        self.projection_matrices = []
        for cam_id in ['cam1', 'cam2', 'cam3']:
            cam_data = self.config['cameras'][cam_id]
            P = get_look_at_projection_matrix(
                self.config['intrinsics'],
                cam_data['position']
                # target defaults to [0, 0, 0]
            )
            self.projection_matrices.append(P)

    def detect_hands_in_frames(self, frames):
        """
        Detects hand centers in multiple frames.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Cannot detect hands in frames.")

        centers = []
        for i, frame in enumerate(frames):
            results = self.model.predict(frame, conf=0.5, verbose=False)
            
            if len(results[0].boxes) > 0:
                box = results[0].boxes[0].xyxy[0].cpu().numpy()
                center = get_hand_center(box)
                centers.append(center)
            else:
                print(f"Camera {i+1}: No hand detected.")
                return None
        
        return centers

    def localize_3d(self, frames):
        """
        Main pipeline: Detect -> Triangulate
        """
        centers_2d = self.detect_hands_in_frames(frames)
        if centers_2d is None:
            return None
        
        point_3d = triangulate_n_views(self.projection_matrices, centers_2d)
        return point_3d

if __name__ == "__main__":
    # Test with dummy coordinates
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, "../../configs/camera_params.yaml")
    # To run without a model, set model_path to None
    localizer = HandLocalizer(config_file, None)
    
    # Test Point: Hand is at [0, 0, 0]
    # In each camera, it should appear exactly at the principal point (cx, cy)
    cx, cy = localizer.config['intrinsics']['cx'], localizer.config['intrinsics']['cy']
    test_2d_points = [(cx, cy), (cx, cy), (cx, cy)]
    
    pos_3d = triangulate_n_views(localizer.projection_matrices, test_2d_points)
    print(f"Validation Test (Target at center): {pos_3d}")
    print("Should be near [0, 0, 0]")
