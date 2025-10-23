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
from tqdm import tqdm
from datetime import datetime

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

# choose latest model state by file name in format yyyy_mm_ddThh-mm-ss
files = [f for f in os.listdir(model_path) if os.path.isfile(os.path.join(model_path, f))]
dated_files = []
for f in files:
    name, ext = os.path.splitext(f)
    try:
        dt = datetime.strptime(name, "%Y_%m_%dT%H-%M-%S")
        dated_files.append((dt, f))
    except ValueError:
        continue

if dated_files:
    dated_files.sort()
    latest_model_state = [dated_files[-1][1]]
else:
    if not files:
        raise FileNotFoundError(f"No model files found in {model_path}")
    latest_model_state = [max(files, key=lambda x: os.path.getmtime(os.path.join(model_path, x)))]

print(f"using model state: {latest_model_state[0]}")
model.load_state_dict(torch.load(os.path.join(model_path, latest_model_state[-1]), map_location=device))
model = model.to(device)
model.eval()

results = []
image_files = sorted(
    [f for f in os.listdir(test_dir) if f.lower().endswith((".jpg"))],
    key=lambda x: int(os.path.splitext(x)[0])
)

with torch.no_grad():
    progress_bar = tqdm(image_files, desc="Predicting", unit="image")
    for idx, filename in enumerate(progress_bar):
        img_path = os.path.join(test_dir, filename)
        img = Image.open(img_path).convert("RGB")
        input_tensor = transform(img).unsqueeze(0).to(device)

        outputs = model(input_tensor)
        _, pred = torch.max(outputs, 1)
        label = class_names[pred.item()]

        results.append([idx + 1, label])

        # print(f"{idx}: {filename} -> {label}")

with open(output_csv, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "character"])
    writer.writerows(results)

print(f"\nSaved predictions to {output_csv}")
