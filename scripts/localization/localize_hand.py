import os
import yaml
import time
import numpy as np
import torch
from ultralytics import YOLO
from .geometry_utils import (
    get_look_at_projection_matrix, 
    triangulate_n_views, 
    get_hand_center,
    OneEuroFilter,
    is_within_bounds
)

class HandLocalizer:
    def __init__(self, config_path, model_path, workspace_bounds=None, filter=None):
        # 1. Load Camera Config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # 2. Load ML Model with CUDA support
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"HandLocalizer: Using device '{self.device}'")
        
        if model_path:
            self.model = YOLO(model_path).to(self.device)
        else:
            self.model = None
        
        # 3. Pre-calculate Projection Matrices
        self.projection_matrices = []
        for cam_id in ['cam1', 'cam2', 'cam3']:
            cam_data = self.config['cameras'][cam_id]
            P = get_look_at_projection_matrix(
                self.config['intrinsics'],
                cam_data['position']
            )
            self.projection_matrices.append(P)

        # 4. Filters and State
        # Default to a 400mm cube centered at origin if not provided
        self.workspace_bounds = workspace_bounds or [[-177.8, 177.8], [-177.8, 177.8], [-177.8, 177.8]]
        self.error_threshold = 25.0 # Max reprojection error allowed (pixels)
        
        self.filter = filter
        self.last_pos = None

    def detect_hands_in_frames(self, frames):
        """
        Detects hand centers in multiple frames using BATCH inference for speed.
        Uses YOLOv8-Pose to extract Keypoint 0 (Wrist) for consistent 3D triangulation.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Cannot detect hands in frames.")

        # Batch Inference (Passing all frames at once is much faster on GPU)
        results = self.model.predict(frames, conf=0.5, verbose=False, device=self.device)

        centers = []
        for i, result in enumerate(results):
            # Check if a hand was detected AND if it has keypoints
            if len(result.boxes) > 0 and hasattr(result, 'keypoints') and result.keypoints is not None:
                # result.keypoints.xy is a tensor of shape (num_hands, num_keypoints, 2)
                # We take the first hand [0], and the first keypoint [0] which is the wrist
                wrist_xy = result.keypoints.xy[0][0].cpu().numpy()
                
                # Check if the keypoint is valid (YOLO outputs [0,0] if it can't see the keypoint but sees the box)
                if wrist_xy[0] == 0 and wrist_xy[1] == 0:
                    return None
                    
                centers.append((wrist_xy[0], wrist_xy[1]))
            else:
                return None # Pipeline requires all views to have a detection for robust DLT
        
        return centers

    def localize_3d(self, frames):
        """
        Main pipeline: Detect -> Triangulate -> Filter -> Smooth
        """
        centers_2d = self.detect_hands_in_frames(frames)
        if centers_2d is None:
            return None
        
        # 1. Triangulate
        point_3d, error = triangulate_n_views(self.projection_matrices, centers_2d)
        
        # 2. Reject if Cameras Disagree (Error is too high)
        if error > self.error_threshold:
            return self.last_pos
            
        # 3. Reject if Outside Workspace
        if not is_within_bounds(point_3d, self.workspace_bounds):
            return self.last_pos

        # 4. Smooth with One Euro Filter
        t = time.time()
        if self.filter is None:
            # self.filter = OneEuroFilter(t, point_3d, min_cutoff=1.5, beta=0.01)
            self.last_pos = point_3d
        else:
            self.last_pos = self.filter(t, point_3d)
            
        return self.last_pos

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, "../../configs/camera_params.yaml")
    localizer = HandLocalizer(config_file, None)
    print("HandLocalizer initialized successfully.")
