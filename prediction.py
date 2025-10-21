import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import csv
import numpy as np
model = "resnet50"
model_path = "best_resnet50.pth"
data_dir = "./data/test-renamed_images"
timestamp = np.datetime64('now').astype('str').replace(':', '-').replace(' ', '_')
output_csv = f"{model}_{timestamp}.csv"
train_dir = "./data/train"

class_names = os.listdir(train_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def get_model(model_name, num_classes, pretrained=False):
    if model_name == "resnet50":
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "resnet101":
        model = models.resnet101(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

num_classes = len(class_names)
model = get_model(model, num_classes, pretrained=False)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

results = []
image_files = sorted([
    f for f in os.listdir(data_dir)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

with torch.no_grad():
    for idx, filename in enumerate(image_files, start=1):
        img_path = os.path.join(data_dir, filename)
        img = Image.open(img_path).convert("RGB")
        input_tensor = transform(img).unsqueeze(0).to(device)

        outputs = model(input_tensor)
        _, pred = torch.max(outputs, 1)
        label = class_names[pred.item()]

        results.append([idx, label])

        print(f"{idx}: {filename} -> {label}")

with open(output_csv, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "character"])
    writer.writerows(results)

print(f"\nSaved predictions to {output_csv}")
