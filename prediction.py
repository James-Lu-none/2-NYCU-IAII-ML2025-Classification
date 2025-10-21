import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import csv
import numpy as np
from torchvision import datasets
from models import *
import argparse

test_dir = "./data/test-renamed_images"
train_dir = "./data/train"

parser = argparse.ArgumentParser()
parser.add_argument("--model_choice", type=str, required=True, help="model choice: xgboost, randomforest, regression")

args = parser.parse_args()
model_choice = args.model_choice

timestamp = np.datetime64('now').astype('str').replace(':', '-').replace(' ', '_')
output_dir = os.path.join("output",model_choice)
os.makedirs(output_dir, exist_ok=True)
output_csv = os.path.join(output_dir,f"{timestamp}.csv")

train_dataset = datasets.ImageFolder(train_dir)
class_names = train_dataset.classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def get_model(model_name):
    try:
        model_fn = globals()[model_name]  # look up function by name
        model = model_fn()
        return model
    except KeyError:
        raise ValueError(f"Unknown model choice: {model_name}")


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
model = get_model(model_choice)
model_path = os.path.join("model", model_choice)
latest_model_state = os.listdir(model_path)
model.load_state_dict(torch.load(os.path.join(model_path, latest_model_state[-1]), map_location=device))
model = model.to(device)
model.eval()

results = []
image_files = sorted(
    [f for f in os.listdir(test_dir) if f.lower().endswith((".jpg"))],
    key=lambda x: int(os.path.splitext(x)[0])
)

with torch.no_grad():
    for idx, filename in enumerate(image_files, start=1):
        img_path = os.path.join(test_dir, filename)
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
