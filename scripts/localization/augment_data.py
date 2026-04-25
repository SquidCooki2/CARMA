import cv2
import numpy as np
import os
import glob

def process_labels(label_path, transform_type="none"):
    """
    Reads a YOLO label file, forces the class to 0, and updates the bounding 
    box coordinates if a geometric transformation was applied.
    """
    if not os.path.exists(label_path):
        return []
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
        
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            # Force class to 0 (Mimics relabel_data.py)
            class_id = '0'
            x_c = float(parts[1])
            y_c = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
            
            # Adjust coordinates for geometric augmentations
            if transform_type == "hflip":
                x_c = 1.0 - x_c
            elif transform_type == "vflip":
                y_c = 1.0 - y_c
                
            # Ensure coordinates stay within 0-1 bounds just in case
            x_c = max(0.0, min(1.0, x_c))
            y_c = max(0.0, min(1.0, y_c))
                
            new_lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
            
    return new_lines

def augment_image(img, aug_type):
    """
    Applies a specific augmentation technique to the image.
    """
    if aug_type == "hflip":
        return cv2.flip(img, 1)
    
    elif aug_type == "vflip":
        return cv2.flip(img, 0)
        
    elif aug_type == "brightness":
        # Increase brightness
        return cv2.convertScaleAbs(img, alpha=1.0, beta=50)
        
    elif aug_type == "contrast":
        # Increase contrast
        return cv2.convertScaleAbs(img, alpha=1.5, beta=0)
        
    elif aug_type == "blur":
        # Gaussian Blur
        return cv2.GaussianBlur(img, (9, 9), 0)
        
    elif aug_type == "noise":
        # Add random Gaussian noise
        noise = np.random.normal(0, 25, img.shape).astype(np.float32)
        noisy_img = cv2.add(img.astype(np.float32), noise)
        return np.clip(noisy_img, 0, 255).astype(np.uint8)
        
    elif aug_type == "hsv":
        # Random hue shift and saturation increase
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + 15) % 180  # Shift hue
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255) # Increase saturation
        hsv = hsv.astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
    elif aug_type == "grayscale":
        # Convert to grayscale, but keep 3 channels for YOLO compatibility
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
    return img

def run_augmentation():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.abspath(os.path.join(script_dir, "../../data/egohands/train"))
    
    images_dir = os.path.join(train_dir, "images")
    labels_dir = os.path.join(train_dir, "labels")
    
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print(f"Training data directories not found at {train_dir}!")
        return
        
    # Only grab original files (skip files that already have "_aug_" in the name)
    all_image_files = glob.glob(os.path.join(images_dir, "*.jpg"))
    original_image_files = [f for f in all_image_files if "_aug_" not in f]
    
    total = len(original_image_files)
    
    if total == 0:
        print("No original images found to augment.")
        return
        
    augmentations = [
        "hflip", "vflip", "brightness", "contrast", 
        "blur", "noise", "hsv", "grayscale"
    ]
    
    print(f"Found {total} original images. Applying 8 augmentations to each...")
    
    count = 0
    for img_path in original_image_files:
        filename = os.path.basename(img_path)
        name, ext = os.path.splitext(filename)
        
        label_path = os.path.join(labels_dir, f"{name}.txt")
        
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        for aug in augmentations:
            # 1. Process Label FIRST
            aug_lines = process_labels(label_path, aug)
            
            # 2. Only save IF we have labels
            if aug_lines:
                aug_img = augment_image(img, aug)
                aug_img_path = os.path.join(images_dir, f"{name}_aug_{aug}{ext}")
                cv2.imwrite(aug_img_path, aug_img)
                
                aug_label_path = os.path.join(labels_dir, f"{name}_aug_{aug}.txt")
                with open(aug_label_path, 'w') as f:
                    f.writelines(aug_lines)
            
        count += 1
        if count % 100 == 0:
            print(f"Processed {count}/{total} original images...")
            
    print(f"Augmentation complete! Generated {count * 8} new images and labels.")

if __name__ == "__main__":
    run_augmentation()
