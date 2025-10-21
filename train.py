import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import os
import numpy as np
from collections import defaultdict
from models import *

data_dir = "./data"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def get_model(model_name):
    match model_name:
        case "resnet50_v1":
            return resnet50_v1()
        case "resnet50_v2":
            return resnet50_v2()
        case "resnet101":
            return resnet101_v1()
        
# ============================================
# 2. DATA PREPARATION
# ============================================

def get_data_loaders(batch_size=32, num_workers=4):
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
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, len(full_dataset.classes)


# ============================================
# 3. TRAINING FUNCTION
# ============================================

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

# ============================================
# 4. MAIN TRAINING LOOP
# ============================================

def train_model(model_name, epochs=10, 
                batch_size=32, lr=0.001, freeze_backbone=True):
    """
    Train a transfer learning model
    
    Args:
        data_dir: Path to data directory
        model_name: Model architecture to use
        epochs: Number of training epochs
        batch_size: Batch size (reduce if OOM)
        lr: Learning rate
        freeze_backbone: If True, only train final layer initially
    """
    
    # Get data loaders
    train_loader, val_loader, num_classes = get_data_loaders(
        batch_size=batch_size
    )
    print(f"Number of classes: {num_classes}")
    
    # Create model
    model = get_model(model_name)
    
    # Optionally freeze backbone
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze final layer
        if hasattr(model, 'fc'):
            for param in model.fc.parameters():
                param.requires_grad = True
        elif hasattr(model, 'classifier'):
            for param in model.classifier.parameters():
                param.requires_grad = True
    
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Training loop
    best_acc = 0.0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 40)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f'best_{model_name}.pth')
            print(f"Saved best model with accuracy: {best_acc:.2f}%")
    
    print(f"\nTraining complete! Best validation accuracy: {best_acc:.2f}%")
    return model

if __name__ == "__main__":
    # Train with frozen backbone first (faster, less memory)
    # print("Phase 1: Training with frozen backbone")
    # model = train_model(
    #     model_name="resnet50",  # Change model here
    #     epochs=5,
    #     batch_size=64,  # Adjust based on your VRAM usage
    #     lr=0.001,
    #     freeze_backbone=True
    # )
    
    # Fine-tune entire model (optional)
    print("\nPhase 2: Fine-tuning entire model")
    model = train_model(
        model_name="resnet50",
        epochs=5,
        batch_size=100,  # Reduce batch size for full fine-tuning
        lr=0.0001,  # Lower learning rate
        freeze_backbone=False
    )