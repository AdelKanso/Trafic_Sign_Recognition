
import torch.nn as nn
from torchvision import models


def build_mobile_net(num_classes=43):
    model = models.mobilenet_v2(weights="DEFAULT")
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model
  
  
def build_resnet18(num_classes=43):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model