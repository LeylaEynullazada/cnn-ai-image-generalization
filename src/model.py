"""CNN models for real vs AI-generated image classification.

This module provides multiple architectures for binary classification:
- baseline_cnn: Improved small CNN with BatchNorm and unified dropout
- resnet18: ResNet-18 with skip connections for better multi-scale artifact detection
"""
import torch
import torch.nn as nn
import torchvision.models as models


# ============================================================================
# IMPROVED BASELINE CNN
# ============================================================================

class ImprovedBaslineCNN(nn.Module):
    """
    Improved small CNN for binary classification (real vs AI-generated).
    
    Features:
    - 4 convolutional blocks with BatchNorm for stable training
    - Unified dropout rate control across all layers
    - Adaptive average pooling for size invariance
    - Designed for 128x128 images with good receptive field coverage
    
    Input: (batch, 3, H, W) e.g. (32, 3, 128, 128)
    Output: (batch, num_classes) logits
    """

    def __init__(self, num_classes=2, dropout_rate=0.3):
        """
        Args:
            num_classes: Number of output classes (default 2 for real/fake)
            dropout_rate: Dropout probability for all dropout layers (0.0 to 1.0)
        """
        super().__init__()
        self.dropout_rate = dropout_rate
        
        # Feature extraction blocks with batch norm and unified dropout
        self.features = nn.Sequential(
            # Block 1: 128x128 -> 64x64
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate),
            
            # Block 2: 64x64 -> 32x32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate),
            
            # Block 3: 32x32 -> 16x16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate),
            
            # Block 4: 16x16 -> 8x8
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate),
            
            # Global pooling: 8x8 -> 1x1
            nn.AdaptiveAvgPool2d(1),
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ============================================================================
# RESNET-18 MODEL
# ============================================================================

class ResNet18Classifier(nn.Module):
    """
    ResNet-18 for binary classification (real vs AI-generated).
    
    Features:
    - Skip connections for better gradient flow in deeper networks
    - Multi-scale receptive field for detecting artifacts at different scales
    - Optional pretrained ImageNet weights as starting point
    - Tunable dropout before classification head
    
    Skip connections are crucial for learning general AI artifacts because:
    - They allow information to flow through many layers without degradation
    - They encourage residual learning (learning differences, not absolute mappings)
    - Multi-scale features help distinguish real images from generator-specific patterns
    
    Input: (batch, 3, H, W) e.g. (32, 3, 128, 128)
    Output: (batch, num_classes) logits
    """

    def __init__(self, num_classes=2, dropout_rate=0.3, pretrained=False):
        """
        Args:
            num_classes: Number of output classes (default 2 for real/fake)
            dropout_rate: Dropout probability before final classification (0.0 to 1.0)
            pretrained: If True, load ImageNet pretrained weights as initialization
        """
        super().__init__()
        self.dropout_rate = dropout_rate
        
        # Load ResNet-18 backbone (with or without pretrained weights)
        self.resnet = models.resnet18(pretrained=pretrained)
        
        # Replace final classification layer
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.resnet(x)


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def get_model(architecture="baseline", num_classes=2, dropout_rate=0.3, pretrained=False):
    """
    Factory function to create and return a model.
    
    Args:
        architecture: Model architecture to use. Options:
            - "baseline": Improved small CNN with BatchNorm and dropout (default)
            - "resnet18": ResNet-18 with skip connections
        num_classes: Number of output classes (default 2 for real/fake)
        dropout_rate: Dropout probability for all dropout layers (0.0 to 1.0)
        pretrained: If True and architecture=="resnet18", load ImageNet pretrained weights
    
    Returns:
        Initialized PyTorch model
    
    Examples:
        # Use improved baseline with 0.3 dropout
        model = get_model(architecture="baseline", dropout_rate=0.3)
        
        # Use ResNet-18 with higher regularization
        model = get_model(architecture="resnet18", dropout_rate=0.5)
        
        # Use ResNet-18 with pretrained ImageNet weights
        model = get_model(architecture="resnet18", pretrained=True)
    """
    if architecture == "baseline":
        return ImprovedBaslineCNN(num_classes=num_classes, dropout_rate=dropout_rate)
    elif architecture == "resnet18":
        return ResNet18Classifier(num_classes=num_classes, dropout_rate=dropout_rate, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown architecture: {architecture}. Choose 'baseline' or 'resnet18'.")