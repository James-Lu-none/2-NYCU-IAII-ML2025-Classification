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

def resnet152_v1():
    model = models.resnet152(weights='DEFAULT')
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def resnet152_v2():
    model = models.resnet152(weights='DEFAULT')
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes)
    )
    return model

def resnext101_64x4d_v1():
    model = models.resnext101_64x4d(weights='DEFAULT')
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def resnext101_64x4d_v2():
    model = models.resnext101_64x4d(weights='DEFAULT')
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes)
    )
    return model

def convnext_small_v1():
    model = models.convnext_small(weights='DEFAULT')
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    return model

def convnext_base_v1():
    model = models.convnext_base(weights='DEFAULT')
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    return model

def convnext_large_v1():
    model = models.convnext_large(weights='DEFAULT')
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    return model
