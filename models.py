import torch.nn as nn
from torchvision import models

num_classes = 50
def resnet50_v1():
    model = models.resnet50(weights='DEFAULT')
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def resnet50_v2():
    model = models.resnet50(weights='DEFAULT')
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    return model

def resnet101_v1():
    model = models.resnet101(weights='DEFAULT')
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
