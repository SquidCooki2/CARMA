import torch
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification
import os

class GraspIntensityPredictor:
    def __init__(self, model_path="models/grasp_intensity_best"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path {model_path} not found. Please train the model first.")
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = ViTImageProcessor.from_pretrained(model_path)
        self.model = ViTForImageClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()

    def predict(self, image):
        """
        Predicts grasp intensity for a PIL image.
        Returns a float between 0 and 1.
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            # Assume numpy array (OpenCV)
            image = Image.fromarray(image).convert("RGB")
            
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # The model outputs a continuous value (regression)
            intensity = outputs.logits.item()
            
        # Optional: Clip to [0, 1] just in case
        intensity = max(0.0, min(1.0, intensity))
        return intensity

if __name__ == "__main__":
    # Example usage
    try:
        predictor = GraspIntensityPredictor()
        # test_img = "path/to/test.jpg"
        # if os.path.exists(test_img):
        #     print(f"Intensity: {predictor.predict(test_img):.4f}")
        print("Predictor initialized successfully.")
    except Exception as e:
        print(f"Error: {e}")
