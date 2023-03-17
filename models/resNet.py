import torch.nn as nn
from torchvision import datasets, models, transforms


class ResNet18(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=False)
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x


class ResNet50(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet50, self).__init__()
        self.model = models.resnet50(pretrained=False)
        self.model.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x
