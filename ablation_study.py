import torch
import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from sklearn.metrics import f1_score

seed = 65465534
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------------------------
# Dataset Class with Custom Transforms
# ---------------------------
class DreaditDataset(Dataset):
    def __init__(self, csv_file, image_folder, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.transform = transform
        
        self.valid_indices = []
        for idx in range(len(self.data)):
            image_filename = f"image_{idx}.png"
            image_path = os.path.join(self.image_folder, image_filename)
            if os.path.exists(image_path):
                self.valid_indices.append(idx)
        
        print(f"Filtered dataset size: {len(self.valid_indices)} out of {len(self.data)}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, index):
        actual_idx = self.valid_indices[index]
        post = self.data.iloc[actual_idx]['post']
        
        post_lower = post.lower()
        if "yes" in post_lower:
            post_label = 1
        elif "no" in post_lower:
            post_label = 0
        else:
            return None  # Skip ambiguous data

        image_filename = f"image_{actual_idx}.png"
        image_path = os.path.join(self.image_folder, image_filename)
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return post, image, post_label

# ---------------------------
# Define Transforms for Training and Validation
# ---------------------------
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

# ---------------------------
# DataLoader Setup
# ---------------------------
csv_path_train = "dreaddit_data/train.csv"
image_folder_train = "dreadit_images_prompt_new/train/images"
dataset_train = DreaditDataset(csv_path_train, image_folder_train, transform=train_transform)
dataset_train = [d for d in dataset_train if d is not None]
dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)

csv_path_val = "dreaddit_data/test.csv"
image_folder_val = "dreadit_images_prompt_new/test/images"
dataset_val = DreaditDataset(csv_path_val, image_folder_val, transform=val_transform)
dataset_val = [d for d in dataset_val if d is not None]
dataloader_val = DataLoader(dataset_val, batch_size=32, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ================================================================
# 1. IMAGE-ONLY MODEL (Fine-tuning Entire Model)
# ================================================================
class CLIPImageClassifier(nn.Module):
    def __init__(self, clip_model):
        super(CLIPImageClassifier, self).__init__()
        self.clip = clip_model
        image_dim = clip_model.visual_projection.out_features  
        self.dropout = nn.Dropout(0.2)  # slightly reduced dropout
        self.classifier = nn.Linear(image_dim, 2)

    def forward(self, image):
        image_features = self.clip.get_image_features(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        features = self.dropout(image_features)
        logits = self.classifier(features)
        return logits

def train_and_evaluate_image_only():
    print("Training IMAGE-ONLY model (fine-tuning entire model)")
    
    clip_model_base = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model_base.to(device)
    
    model = CLIPImageClassifier(clip_model_base).to(device)
    
    # Initially freeze CLIP backbone to let classifier adjust
    for param in model.clip.parameters():
        param.requires_grad = False

    optimizer = optim.AdamW([
        {"params": model.clip.parameters(), "lr": 5e-6},       # lower learning rate for CLIP backbone
        {"params": model.classifier.parameters(), "lr": 5e-5},   # moderate learning rate for classifier
    ], weight_decay=5e-5)  # reduced weight decay

    # Use cosine annealing with warm restarts for smoother LR decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

    epochs = 30
    best_val_acc = 0.0
    patience = 10
    patience_counter = 0
    best_model_path = "./trained_models/best_image_only_model.pth"

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        
        # Unfreeze the CLIP backbone after a few epochs
        if epoch == 3:
            for param in clip_model_base.parameters():
                param.requires_grad = True

        for post, image, label in dataloader_train:
            image = image.to(device)
            label = label.to(device)
            
            optimizer.zero_grad()
            outputs = model(image)
            loss = nn.CrossEntropyLoss()(outputs, label)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == label).sum().item()
            total += label.size(0)
            total_loss += loss.item()
        
        scheduler.step()  # update scheduler
        train_acc = correct / total

        # Validation
        model.eval()
        correct_val, total_val = 0, 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for post, image, label in dataloader_val:
                image = image.to(device)
                label = label.to(device)
                outputs = model(image)
                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == label).sum().item()
                total_val += label.size(0)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(label.cpu().numpy())
        val_acc = correct_val / total_val
        f1 = f1_score(all_labels, all_preds, average='weighted')
        print(f"[Image-Only] Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Val F1: {f1:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping for image-only model.")
                break

    print(f"Best Image-Only Validation Accuracy: {best_val_acc:.4f}")
    model.load_state_dict(torch.load(best_model_path))
    torch.save(model.state_dict(), "./trained_models/final_image_only_model.pth")
    print("Final image-only model saved.")

# ================================================================
# Run Experiments
# ================================================================
if __name__ == "__main__":
    os.makedirs("./trained_models", exist_ok=True)
    
    # Train and evaluate the image-only model
    train_and_evaluate_image_only()
