import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

def calculate_grasp_intensity(xyz):
    """
    Calculates a heuristic grasp intensity value from 3D keypoints.
    0 = Fully Open, 1 = Fully Closed.
    """
    xyz = np.array(xyz)
    # Keypoints indices:
    # 0: Wrist
    # 4, 8, 12, 16, 20: Fingertips (Thumb, Index, Middle, Ring, Pinky)
    # 9: Middle finger MCP (base)
    
    wrist = xyz[0]
    fingertips = xyz[[4, 8, 12, 16, 20]]
    middle_base = xyz[9]
    
    # 1. Average distance from wrist to fingertips
    dists = np.linalg.norm(fingertips - wrist, axis=1)
    avg_dist = np.mean(dists)
    
    # 2. Normalize by hand scale (wrist to middle finger base)
    hand_scale = np.linalg.norm(middle_base - wrist)
    if hand_scale == 0:
        return 0
        
    normalized_openness = avg_dist / hand_scale
    return normalized_openness

def prepare_labels(root_path, output_csv):
    xyz_path = os.path.join(root_path, 'training_xyz.json')
    
    if not os.path.exists(xyz_path):
        print(f"Error: {xyz_path} not found.")
        return

    with open(xyz_path, 'r') as f:
        all_xyz = json.load(f)
        
    print(f"Processing {len(all_xyz)} samples...")
    results = []
    
    for i, xyz in enumerate(tqdm(all_xyz)):
        intensity = calculate_grasp_intensity(xyz)
        img_name = f"{i:08d}.jpg"
        results.append({'image_id': img_name, 'raw_intensity': intensity})
        
    df = pd.DataFrame(results)
    
    # Min-Max scale and Invert
    # Higher raw_intensity means more "open" (fingers far from wrist)
    # We want 1 for closed, 0 for open.
    
    min_val = df['raw_intensity'].min()
    max_val = df['raw_intensity'].max()
    
    # Normalize to [0, 1] where 1 is MAX OPEN
    df['normalized'] = (df['raw_intensity'] - min_val) / (max_val - min_val)
    
    # Invert so 1 is CLOSED, 0 is OPEN
    df['grasp_intensity'] = 1.0 - df['normalized']
    
    # Save to CSV
    df[['image_id', 'grasp_intensity']].to_csv(output_csv, index=False)
    print(f"Saved grasp labels to {output_csv}")
    
    # Print some stats
    print(f"Min raw: {min_val:.4f}, Max raw: {max_val:.4f}")
    print(df['grasp_intensity'].describe())

if __name__ == "__main__":
    # Assuming the FreiHAND dataset is in data/freihand
    os.makedirs('data/grasp_intensity', exist_ok=True)
    prepare_labels('data/freihand', 'data/grasp_intensity/grasp_labels.csv')
