"""Models for the facial keypoint study."""

import torch
import torch.nn as nn
from torchvision import models


class MLP2(nn.Module):
    def __init__(self, num_outputs, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 96, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_outputs),
        )

    def forward(self, x):
        return self.net(x)


class ShallowCNN(nn.Module):
    def __init__(self, num_outputs):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 12 * 12, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_outputs),
        )

    def forward(self, x):
        return self.head(self.features(x))


def build_resnet18_pretrained(num_outputs):
    """torchvision ResNet18 (ImageNet1K_V1) with 1-channel conv1 and fresh fc.

    conv1 weight = mean of original 3-channel weight along input dim,
    preserving activation magnitude vs any single RGB channel.
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    m = models.resnet18(weights=weights)
    old = m.conv1
    new = nn.Conv2d(
        1,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=(old.bias is not None),
    )
    with torch.no_grad():
        new.weight.copy_(old.weight.mean(dim=1, keepdim=True))
    m.conv1 = new
    m.fc = nn.Linear(m.fc.in_features, num_outputs)
    return m


MODEL_BUILDERS = {
    "mlp2": (lambda n: MLP2(n), 1e-3),
    "cnn_shallow": (lambda n: ShallowCNN(n), 1e-3),
    "resnet18_pretrained": (lambda n: build_resnet18_pretrained(n), 1e-4),
}
