import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from models import *
import argparse
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class visualization:
    def __init__(self, model_choice=None, model_state_path=None, dataset=None):
        self.model = None
        self.model_choice = model_choice
        self.model_state_path = model_state_path
        self.dataset = dataset

    def get_model(self):
        try:
            model_fn = globals()[self.model_choice]  # look up function by name
            self.model = model_fn()
        except KeyError:
            raise ValueError(f"Unknown model choice: {self.model_choice}")
    
        self.model.to(device)
    
    def run(self):
        self.get_model()
        self.model.load_state_dict(torch.load(self.model_state_path, map_location=device))
        
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])

        sample_image_path = os.path.join("./data", self.dataset)
        # load a image from dataset and visualize the image after passing the first convolutional layer
        sample_image = datasets.ImageFolder(root=sample_image_path, transform=transform)
        sample_loader = DataLoader(sample_image, batch_size=1, shuffle=True)
    
        self.model.eval()
        with torch.no_grad():
            images, _ = next(iter(sample_loader))
            images = images.to(device)

            # Forward pass through each major stage
            x = self.model.conv1(images)
            self.visualize_features(x, "conv1")

            x = self.model.layer1(x)
            self.visualize_features(x, "layer1")

            x = self.model.layer2(x)
            self.visualize_features(x, "layer2")

            x = self.model.layer3(x)
            self.visualize_features(x, "layer3")

            x = self.model.layer4(x)
            self.visualize_features(x, "layer4")

    def visualize_features(self, features, layer_name):
        # Upsample deep layers for better visualization
        if features.shape[2] < 64:
            features = F.interpolate(features, scale_factor=4, mode='bilinear', align_corners=False)

        features = features.cpu().squeeze(0)  # remove batch dim
        num_features = features.shape[0]

        plt.figure(figsize=(15, 15))
        for i in range(min(num_features, 64)):
            plt.subplot(8, 8, i + 1)
            img = features[i]
            # Normalize each feature map independently for better contrast
            img = (img - img.min()) / (img.max() - img.min() + 1e-5)
            plt.imshow(img, cmap='gray')
            plt.axis('off')

        plt.suptitle(f"Feature maps after {layer_name}", fontsize=16)
        plt.tight_layout()
        out_path = os.path.join("visualizations", f"feature_maps_{layer_name}.png")
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_choice", type=str, required=True, help="model choice")
    parser.add_argument("--model_state_path", type=str, required=True, help="model state path for fine-tuning")
    parser.add_argument("--dataset", type=str, required=True, help="training dataset")
    args = parser.parse_args()
    model_choice = args.model_choice
    model_state_path = args.model_state_path
    dataset = args.dataset

    visualize = visualization(model_choice=model_choice, model_state_path=model_state_path, dataset=dataset)
    visualize.run()