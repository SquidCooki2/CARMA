import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from torchvision.transforms import ColorJitter, RandomRotation, Compose, Resize, ToTensor, Normalize

class GraspDataset(Dataset):
    def __init__(self, df, img_dir, processor, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['image_id']
        label = self.df.iloc[idx]['grasp_intensity']
        img_path = os.path.join(self.img_dir, img_name)
        
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        # ViT processor expects a PIL image or numpy array
        # If we applied torchvision transforms, we might have a tensor
        # But processor usually handles normalization too.
        # Let's keep it simple: apply augmentation then processor
        
        inputs = self.processor(images=image, return_tensors="pt")
        # Remove batch dimension added by processor
        pixel_values = inputs['pixel_values'].squeeze(0)
        
        return {
            'pixel_values': pixel_values,
            'labels': torch.tensor(label, dtype=torch.float32)
        }

def train():
    model_name = "google/vit-base-patch16-224-in21k"
    img_dir = 'data/freihand/training/rgb'
    csv_path = 'data/grasp_intensity/grasp_labels.csv'
    
    if not os.path.exists(csv_path):
        print(f"Labels not found at {csv_path}. Run data_prep.py first.")
        return

    df = pd.read_csv(csv_path)
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    
    processor = ViTImageProcessor.from_pretrained(model_name)
    
    # Augmentations
    train_transform = Compose([
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        RandomRotation(15),
    ])

    train_dataset = GraspDataset(train_df, img_dir, processor, transform=train_transform)
    val_dataset = GraspDataset(val_df, img_dir, processor)

    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=1,
        problem_type="regression"
    )

    training_args = TrainingArguments(
        output_dir="./models/grasp_intensity_vit",
        per_device_train_batch_size=32,
        evaluation_strategy="steps",
        num_train_epochs=3,
        fp16=torch.cuda.is_available(),
        save_steps=1000,
        eval_steps=1000,
        logging_steps=100,
        learning_rate=2e-5,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="none",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    
    # Save the final model and processor
    trainer.save_model("models/grasp_intensity_best")
    processor.save_pretrained("models/grasp_intensity_best")

if __name__ == "__main__":
    train()
