"""Model definitions."""

import torch.nn as nn
import torchvision.models as models


class ImprovedBaslineCNN(nn.Module):
    """Small CNN used as the baseline model."""

    def __init__(self, num_classes=2, dropout_rate=0.3):
        super().__init__()
        self.dropout_rate = dropout_rate

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate),
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class ResNet18Classifier(nn.Module):
    """ResNet-18 classifier."""

    def __init__(self, num_classes=2, dropout_rate=0.3, pretrained=False):
        super().__init__()
        self.dropout_rate = dropout_rate

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.resnet = models.resnet18(weights=weights)

        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.resnet(x)


def get_model(architecture="baseline", num_classes=2, dropout_rate=0.3, pretrained=False):
    """Create a model by name."""
    if architecture == "baseline":
        return ImprovedBaslineCNN(num_classes=num_classes, dropout_rate=dropout_rate)
    if architecture == "resnet18":
        return ResNet18Classifier(num_classes=num_classes, dropout_rate=dropout_rate, pretrained=pretrained)
    raise ValueError(f"Unknown architecture: {architecture}. Choose 'baseline' or 'resnet18'.")
