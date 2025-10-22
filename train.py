import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import os
import numpy as np
from collections import defaultdict
from models import *
import argparse
from tqdm import tqdm

data_dir = "./data"
model_dir = "./model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
class train:
    def __init__(self, model_choice=None):
        self.model = None
        self.model_choice = model_choice
        self.train_loader = None
        self.val_loader = None
        self.num_classes = None
        self.best_weights = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.best_acc = 0

    def get_model(self):
        try:
            model_fn = globals()[self.model_choice]  # look up function by name
            self.model = model_fn()
        except KeyError:
            raise ValueError(f"Unknown model choice: {self.model_choice}")
    
        self.model.to(device)
        
    def get_data_loaders(self, batch_size=32, num_workers=4):
        # ImageNet normalization
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # Transforms (same for train/val here)
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])

        # Use single source directory
        full_dataset = datasets.ImageFolder(os.path.join(data_dir, "pre"))

        # Stratified split by class
        val_split = 0.2
        class_indices = defaultdict(list)
        for idx, (_, label) in enumerate(full_dataset.samples):
            class_indices[label].append(idx)

        train_indices, val_indices = [], []
        np.random.seed(42)

        for class_label, indices in class_indices.items():
            indices = np.array(indices)
            np.random.shuffle(indices)
            split_point = int(len(indices) * (1 - val_split))
            train_indices.extend(indices[:split_point].tolist())
            val_indices.extend(indices[split_point:].tolist())

        print(f"Total samples: {len(full_dataset)}")
        print(f"Training samples: {len(train_indices)}")
        print(f"Validation samples: {len(val_indices)}")

        # Create Subsets (keep correct indices, apply transforms after)
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)

        # Assign transforms AFTER splitting
        train_dataset.dataset.transform = transform
        val_dataset.dataset.transform = transform

        # Dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        self.num_classes = len(full_dataset.classes)

    def set_freeze_mode(self, freeze_backbone=True):
        for param in self.model.parameters():
            param.requires_grad = not freeze_backbone

        if freeze_backbone:
            # Unfreeze only classifier
            if hasattr(self.model, "fc"):
                for p in self.model.fc.parameters():
                    p.requires_grad = True
            elif hasattr(self.model, "classifier"):
                for p in self.model.classifier.parameters():
                    p.requires_grad = True

    def train_one_epoch(self):
        self.model.train()
        total_loss, total_correct, total = 0, 0, 0

        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)

        for x, y in progress_bar:
            x, y = x.to(device), y.to(device)
            self.optimizer.zero_grad()
            out = self.model(x)
            loss = self.criterion(out, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * x.size(0)
            _, pred = out.max(1)
            total_correct += pred.eq(y).sum().item()
            total += y.size(0)
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * total_correct / total:.2f}%'
            })

        avg_loss = total_loss / total
        avg_acc = 100.0 * total_correct / total
        return avg_loss, avg_acc

    def validate(self):
        self.model.eval()
        total_loss, total_correct, total = 0, 0, 0

        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(device), y.to(device)
                out = self.model(x)
                loss = self.criterion(out, y)

                total_loss += loss.item() * x.size(0)
                _, pred = out.max(1)
                total_correct += pred.eq(y).sum().item()
                total += y.size(0)

        return total_loss / total, 100.0 * total_correct / total

    def train_model(self, epochs, lr, freeze_backbone):
        self.set_freeze_mode(freeze_backbone)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr
        )
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.5)
        for epoch in range(epochs):
            train_loss, train_acc = self.train_one_epoch()
            val_loss, val_acc = self.validate()
            self.scheduler.step()

            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.best_weights = self.model.state_dict()
                print(f"New best model (val_acc={self.best_acc:.2f}%)")

        print(f"Finished training, Best acc: {self.best_acc:.2f}%")

    def run(self):
        self.get_data_loaders(batch_size=64)
        self.get_model()

        # print("=== Phase 1: Train classifier only ===")
        # self.train_model(epochs=5, lr=0.001, freeze_backbone=True)

        print("=== Phase 2: Fine-tune entire model ===")
        self.train_model(epochs=3, lr=0.001, freeze_backbone=False)

        timestamp = np.datetime64('now').astype('str').replace(':', '-').replace(' ', '_')
        model_path = os.path.join(model_dir, self.model_choice)
        os.makedirs(model_path, exist_ok=True)
        print(f"Saving best model to {model_path}/{timestamp}_{self.best_acc}.pth")
        model_disk = os.path.join(model_path, f"{timestamp}_{self.best_acc}.pth")
        torch.save(self.model.state_dict(), model_disk)

if __name__ == "__main__":
    # Train with frozen backbone first (faster, less memory)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_choice", type=str, required=True, help="model choice")
    args = parser.parse_args()
    model_choice = args.model_choice

    trainer = train(model_choice=model_choice)
    trainer.run()