import torch
import torch.nn as nn
import torchvision.models as models

class ResNet50(nn.Module):
    def __init__(self, in_channels=4, num_classes=1000):
        super(ResNet50, self).__init__()
        self.resnet50 = models.resnet50(pretrained=False)
        self.resnet50.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet50.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        return self.resnet50(x)

