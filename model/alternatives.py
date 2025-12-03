import torch.nn as nn
from torchvision import models
from utils.constants import NUM_CLASSES


def build_mobile_net(num_classes=NUM_CLASSES):
    model = models.mobilenet_v2(weights="DEFAULT")
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model
  
  
def build_resnet18(num_classes=NUM_CLASSES):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
