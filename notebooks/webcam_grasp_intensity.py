import cv2
import torch
import numpy as np
from PIL import Image
import sys
import os

# Add scripts directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))

from grasp_intensity.inference import GraspIntensityPredictor
from ultralytics import YOLO

def run_demo(yolo_model_path='models/hand_localization/weights/best.pt', 
             grasp_model_path='models/grasp_intensity_best'):
    
    print("Loading models...")
    # 1. Load YOLOv8 for hand detection
    if not os.path.exists(yolo_model_path):
        print(f"Warning: YOLO model not found at {yolo_model_path}. Hand cropping will be disabled.")
        detector = None
    else:
        detector = YOLO(yolo_model_path)

    # 2. Load Grasp Intensity Predictor
    try:
        predictor = GraspIntensityPredictor(grasp_model_path)
    except Exception as e:
        print(f"Error loading grasp model: {e}")
        return

    # 3. Initialize Webcam
    cap = cv2.VideoCapture(1401)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Demo started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        intensity = 0.0
        display_frame = frame.copy()

        if detector:
            # Run detection
            results = detector(frame, conf=0.5, verbose=False)
            
            for result in results:
                boxes = result.boxes
                if len(boxes) > 0:
                    # Get the most confident box
                    box = boxes[0]
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    
                    # Add padding to the crop (ViT performs better with some context)
                    pad_w = int((x2 - x1) * 0.2)
                    pad_h = int((y2 - y1) * 0.2)
                    cx1 = max(0, x1 - pad_w)
                    cy1 = max(0, y1 - pad_h)
                    cx2 = min(w, x2 + pad_w)
                    cy2 = min(h, y2 + pad_h)

                    hand_crop = frame[cy1:cy2, cx1:cx2]
                    
                    if hand_crop.size > 0:
                        # Predict Intensity
                        intensity = predictor.predict(hand_crop)
                        
                        # Draw bounding box and value
                        cv2.rectangle(display_frame, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2)
                        label = f"Grasp: {intensity:.2f}"
                        cv2.putText(display_frame, label, (cx1, cy1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 4. Draw Progress Bar for Grasp Intensity
        bar_x, bar_y = 50, 50
        bar_w, bar_h = 20, 200
        # Background bar
        cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        # Intensity fill
        fill_h = int(intensity * bar_h)
        cv2.rectangle(display_frame, (bar_x, bar_y + bar_h - fill_h), 
                      (bar_x + bar_w, bar_y + bar_h), (0, 255, 255), -1)
        cv2.putText(display_frame, "Closed", (bar_x - 10, bar_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Hand Grasp Intensity Demo", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_demo()
