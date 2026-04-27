import json
import os
import numpy as np
from tqdm import tqdm
import random

def project_points(xyz, K):
    """Project 3D coordinates to 2D pixels using the intrinsic matrix K."""
    xyz = np.array(xyz)
    K = np.array(K)
    uvw = K @ xyz.T
    uv = uvw[:2, :] / uvw[2, :]
    return uv.T

def convert_to_yolo_pose(root_path, output_path, img_size=224, val_split=0.1):
    """
    Converts FreiHAND dataset to YOLOv8-Pose format with a train/val split.
    
    Args:
        root_path: Path to the raw FreiHAND data.
        output_path: Path where the YOLO dataset structure will be created.
        img_size: FreiHAND images are 224x224.
        val_split: Percentage of data to use for validation (0.0 to 1.0).
    """
    # Paths
    img_dir = os.path.join(root_path, 'training', 'rgb')
    xyz_path = os.path.join(root_path, 'training_xyz.json')
    k_path = os.path.join(root_path, 'training_K.json')
    
    # Load JSONs
    with open(xyz_path, 'r') as f:
        all_xyz = json.load(f)
    with open(k_path, 'r') as f:
        all_K = json.load(f)
        
    num_samples = len(all_xyz)
    indices = list(range(num_samples))
    random.shuffle(indices)
    
    split_idx = int(num_samples * (1 - val_split))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    splits = {
        'train': train_indices,
        'val': val_indices
    }
    
    for split_name, idx_list in splits.items():
        yolo_img_dir = os.path.join(output_path, split_name, 'images')
        yolo_lbl_dir = os.path.join(output_path, split_name, 'labels')
        os.makedirs(yolo_img_dir, exist_ok=True)
        os.makedirs(yolo_lbl_dir, exist_ok=True)
        
        print(f"Converting {len(idx_list)} images for split: {split_name}...")
        import shutil
        success_count = 0
        missing_example = ""
        
        for i in tqdm(idx_list):
            img_name = f"{i:08d}.jpg"
            src_img_path = os.path.join(img_dir, img_name)
            
            if not os.path.exists(src_img_path):
                if not missing_example:
                    missing_example = src_img_path
                continue
                
            # 1. Project 3D points to 2D
            points_2d = project_points(all_xyz[i], all_K[i])
            
            # 2. Get Bounding Box from keypoints
            x_min, y_min = np.min(points_2d, axis=0)
            x_max, y_max = np.max(points_2d, axis=0)
            
            # Add padding to bbox (20%)
            w_raw = x_max - x_min
            h_raw = y_max - y_min
            x_min -= w_raw * 0.1
            y_min -= h_raw * 0.1
            x_max += w_raw * 0.1
            y_max += h_raw * 0.1
            
            # 3. Normalize for YOLO (0.0 to 1.0)
            x_center = ((x_min + x_max) / 2) / img_size
            y_center = ((y_min + y_max) / 2) / img_size
            width = (x_max - x_min) / img_size
            height = (y_max - y_min) / img_size
            
            # 4. Format Keypoints: [x, y, visibility]
            kp_string = ""
            for kp in points_2d:
                kx = kp[0] / img_size
                ky = kp[1] / img_size
                v = 2 if (0 <= kx <= 1 and 0 <= ky <= 1) else 0
                kp_string += f" {kx:.6f} {ky:.6f} {v}"
                
            # 5. Write label file
            label_name = f"{i:08d}.txt"
            with open(os.path.join(yolo_lbl_dir, label_name), 'w') as f:
                f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}{kp_string}\n")
                
            # 6. Copy image (Use copy to avoid 'run twice' bugs)
            shutil.copy(src_img_path, os.path.join(yolo_img_dir, img_name))
            success_count += 1
            
        if success_count == 0:
            print(f"\nFound 0 images for {split_name}!")
            print(f"The script was looking for files like: {missing_example}")
        else:
            print(f"\nSuccessfully converted {success_count} / {len(idx_list)} images for {split_name}.")

if __name__ == "__main__":
    convert_to_yolo_pose('data/freihand', 'data/freihand_yolo')
