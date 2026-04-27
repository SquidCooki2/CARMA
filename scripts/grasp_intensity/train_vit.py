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

        # Let's keep it simple: apply augmentation then processor
        
        inputs = self.processor(images=image, return_tensors="pt")
        # Remove batch dimension added by processor
        pixel_values = inputs['pixel_values'].squeeze(0)
        
        return {
            'pixel_values': pixel_values,
            'labels': torch.tensor(label, dtype=torch.float32)
        }

def train(img_dir='data/freihand/training/rgb', csv_path='data/grasp_intensity/grasp_labels.csv', output_dir='models/grasp_intensity_best', checkpoint_dir='./tmp_trainer_output'):
    model_name = "google/vit-base-patch16-224-in21k"
    
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
        output_dir=checkpoint_dir,
        per_device_train_batch_size=32,
        eval_strategy="steps",
        num_train_epochs=3,
        fp16=torch.cuda.is_available(),
        save_steps=500, # Save every 500 steps
        eval_steps=500,
        logging_steps=100,
        learning_rate=2e-5,
        save_total_limit=3, # Keep only last 3 checkpoints to save space on Drive
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="none",
        load_best_model_at_end=True,
        resume_from_checkpoint=True # Automatically try to resume if files exist
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    
    # Save the final model and processor to the specified output_dir
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Model and processor saved to {output_dir}")

if __name__ == "__main__":
    train()
