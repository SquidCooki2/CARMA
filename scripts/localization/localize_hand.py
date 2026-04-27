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
        self.workspace_bounds = workspace_bounds or [[-177.8, 177.8], [-177.8, 177.8], [-177.8, 177.8]]
        self.error_threshold = 35.0 # Max reprojection error allowed (pixels)
        
        # Initialize filter if not provided
        self.filter = filter if filter else OneEuroFilter(time.time(), np.array([0, 0, 0]), min_cutoff=1.5, beta=0.01)
        self.last_pos = None

    def detect_hands_in_frames(self, frames):
        """
        Detects hand centers in multiple frames using BATCH inference.
        Returns a list of (u, v) or None for each camera.
        """
        if self.model is None:
            raise ValueError("Model not loaded.")

        results = self.model.predict(frames, conf=0.4, verbose=False, device=self.device)

        centers = []
        for i, result in enumerate(results):
            if len(result.boxes) > 0 and hasattr(result, 'keypoints') and result.keypoints is not None:
                wrist_xy = result.keypoints.xy[0][0].cpu().numpy()
                if wrist_xy[0] == 0 and wrist_xy[1] == 0:
                    centers.append(None)
                else:
                    centers.append((wrist_xy[0], wrist_xy[1]))
            else:
                centers.append(None)
        
        return centers

    def localize_3d(self, frames):
        """
        Supports N-view triangulation (works with 2 or 3 cameras).
        """
        centers_2d = self.detect_hands_in_frames(frames)
        
        # Filter out cameras that didn't find the hand
        valid_indices = [i for i, pt in enumerate(centers_2d) if pt is not None]
        
        if len(valid_indices) < 2:
            # Need at least 2 cameras to triangulate
            return self.last_pos
            
        active_proj_matrices = [self.projection_matrices[i] for i in valid_indices]
        active_points_2d = [centers_2d[i] for i in valid_indices]
        
        # 1. Triangulate
        point_3d, error = triangulate_n_views(active_proj_matrices, active_points_2d)
        
        # 2. Reject if high error
        if error > self.error_threshold:
            return self.last_pos
            
        # 3. Reject if Outside Workspace
        if not is_within_bounds(point_3d, self.workspace_bounds):
            return self.last_pos

        # 4. Smooth with One Euro Filter
        t = time.time()
        self.last_pos = self.filter(t, point_3d)
            
        return self.last_pos
