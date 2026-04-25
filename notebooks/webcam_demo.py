
import cv2
from ultralytics import YOLO
import os

def run_webcam_demo(model_path):
    # 1. Load the fine-tuned model
    debug = False

    if not debug:
        print(f"Loading model from: {model_path}")
        model = YOLO(model_path)
    
    # 2. Open the webcam (0 is usually the default camera)
    cam = 1403
    print(f"Attempting to open webcam at index: {cam}")
    cap = cv2.VideoCapture(cam)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting webcam demo. Press 'q' to quit.")
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break
            
        # 3. Run YOLO inference
        if not debug:
            results = model.predict(frame, conf=0.5, verbose=False)
        
            # 4. Draw detections on the frame
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Get confidence
                    conf = box.conf[0].cpu().numpy()
                    
                    # Draw bounding box (Green)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Add label with confidence
                    label = f"Hand: {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Mark the center point (for triangulation demo)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        # 5. Display the resulting frame
        cv2.imshow('Hand Detection Demo', frame)
        
        key = cv2.waitKey(1) & 0xFF
        # Break the loop on 'q' key press
        if key == ord('q'):
            break

        elif key == ord('d') or key == ord('a'):
            cam = cam + 1 if key == ord('d') else max(0, cam - 1)
            print(f"Switching to camera index: {cam}")
            
            cap.release()
            cap = cv2.VideoCapture(cam)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_file = os.path.join(script_dir, "../models/hand_detector/weights/best.pt")
    run_webcam_demo(model_file)
