from ultralytics import YOLO
import os

def train_hand_detector(data_yaml_path, epochs=50, imgsz=640, freeze=10):
    """
    Fine-tunes a YOLOv8n model on hand detection data.
    """
    # 1. Load a pre-trained YOLOv8n model (small and fast)
    model = YOLO('yolov8n.pt') 
    
    # 2. Train the model
    # freeze=10 freezes the backbone (first 10 layers), 
    # focusing training on the detection head.
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        project='../models',
        name='hand_detector',
        freeze=freeze,
        exist_ok=True
    )
    
    return results

if __name__ == "__main__":
    # Point to the unified YAML configuration in the data folder
    # Using absolute path to avoid issues with working directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "../../data/egohands.yaml")
    
    train_hand_detector(data_path)
