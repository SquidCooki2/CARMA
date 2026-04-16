from ultralytics import YOLO
import os

def train_hand_detector(data_yaml_path, epochs=50, imgsz=640):
    """
    Fine-tunes a YOLOv8n model on hand detection data.
    """
    # 1. Load a pre-trained YOLOv8n model (small and fast)
    model = YOLO('yolov8n.pt') 
    
    # 2. Train the model
    # We'll freeze the backbone if you only want to train the last layer, 
    # but for hands, fine-tuning the whole network usually gives better results.
    # To freeze the first 10 layers: model.model.freeze(10)
    
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        project='../models',
        name='hand_detector',
        device='cpu' # Use '0' for GPU if available
    )
    
    return results

if __name__ == "__main__":
    # Ensure you have the dataset downloaded and extracted in data/egohands
    data_path = os.path.abspath("../data/egohands.yaml")
    train_hand_detector(data_path)
