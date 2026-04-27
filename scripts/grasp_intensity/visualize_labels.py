import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def visualize_bins(csv_path='data/grasp_intensity/grasp_labels.csv', img_dir='data/freihand/training/rgb'):
    if not os.path.exists(csv_path):
        print(f"CSV not found at {csv_path}. Run data_prep.py first.")
        return

    df = pd.read_csv(csv_path)
    
    # Define 11 bins: [0, 0.1, 0.2, ..., 1.0]
    bins = np.linspace(0, 1, 11)
    fig, axes = plt.subplots(1, 11, figsize=(25, 5))
    fig.suptitle('Grasp Intensity Sanity Check (0=Open, 1=Closed)', fontsize=16)

    for i, target in enumerate(bins):
        # Find the image closest to this target intensity
        idx = (df['grasp_intensity'] - target).abs().idxmin()
        row = df.iloc[idx]
        
        img_path = os.path.join(img_dir, row['image_id'])
        if os.path.exists(img_path):
            img = Image.open(img_path)
            axes[i].imshow(img)
            axes[i].set_title(f"Target: {target:.1f}\nActual: {row['grasp_intensity']:.3f}")
        else:
            axes[i].text(0.5, 0.5, 'Missing', ha='center')
            axes[i].set_title(f"Target: {target:.1f}")
        
        axes[i].axis('off')

    plt.tight_layout()
    output_path = 'scripts/grasp_intensity/sanity_check.png'
    plt.savefig(output_path)
    print(f"Sanity check plot saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    visualize_bins()
