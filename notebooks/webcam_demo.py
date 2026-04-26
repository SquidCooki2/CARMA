import cv2
from ultralytics import YOLO
import os
import numpy as np

def run_webcam_demo(model_path):
    debug = False

    if not debug:
        print(f"Loading YOLO-Pose model from: {model_path}")
        try:
            model = YOLO(model_path)
        except Exception as e:
            print(f"Failed to load model: {e}")
            print("Falling back to yolov8n-pose.pt for testing.")
            model = YOLO('yolov8n-pose.pt')
    
    # Open the webcam 
    # (Defaulting to 1400 range based on your previous logs, fallback to 0 if needed)
    cam = 1400 
    print(f"Attempting to open webcam at index: {cam}")
    cap = cv2.VideoCapture(cam)
    
    if not cap.isOpened():
        print(f"Error: Could not open webcam at index {cam}. Trying index 0...")
        cam = 0
        cap = cv2.VideoCapture(cam)
        if not cap.isOpened():
            print("Error: Could not open any webcam.")
            return

    # Force MJPEG to test bandwidth limits if needed
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Starting webcam demo.")
    print("Press 'q' to quit.")
    print("Press 'd' for next camera, 'a' for previous camera.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting loop.")
            break
            
        if not debug:
            # Run YOLO-Pose inference
            results = model.predict(frame, conf=0.5, verbose=False)
            
            # The .plot() function handles drawing the bounding box AND the 21 keypoint skeleton
            annotated_frame = results[0].plot()
            
            # We explicitly draw a large RED dot over Keypoint 0 (the Wrist)
            # because this is the specific point HandLocalizer uses for 3D triangulation
            if len(results[0].boxes) > 0 and hasattr(results[0], 'keypoints') and results[0].keypoints is not None:
                wrist_xy = results[0].keypoints.xy[0][0].cpu().numpy()
                if wrist_xy[0] != 0 and wrist_xy[1] != 0: # Check if visible
                    cx, cy = int(wrist_xy[0]), int(wrist_xy[1])
                    cv2.circle(annotated_frame, (cx, cy), 8, (0, 0, 255), -1) # Red circle
                    cv2.putText(annotated_frame, "WRIST (TARGET)", (cx + 10, cy), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Display the resulting frame
            cv2.imshow('Hand Pose Detection Demo', annotated_frame)
        else:
            cv2.imshow('Hand Pose Detection Demo', frame)
            
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('d') or key == ord('a'):
            cam = cam + 1 if key == ord('d') else max(0, cam - 1)
            print(f"Switching to camera index: {cam}")
            
            cap.release()
            cap = cv2.VideoCapture(cam)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Automatically find the latest training run
    model_root = os.path.join(script_dir, "../models")
    
    if os.path.exists(model_root):
        train_folders = [f for f in os.listdir(model_root) if f.startswith('train')]
        if train_folders:
            latest_train = sorted(train_folders)[-1]
            model_file = os.path.join(model_root, latest_train, "weights/best.pt")
        else:
            model_file = "yolov8n-pose.pt"
    else:
        model_file = "yolov8n-pose.pt"
        
    run_webcam_demo(model_file)
