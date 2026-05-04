"""Dataset loading for real vs AI-generated image classification.

This module provides data augmentation strategies designed to improve cross-generator
generalization. The key insight is that AI generators produce images with distinct
artifacts (textures, color patterns, noise characteristics). By using realistic
augmentations during training, we encourage the model to learn general "AI-ness"
rather than generator-specific visual patterns.
"""
from pathlib import Path

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


# ImageNet normalization statistics for proper preprocessing
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size=128, augmentation_strength=1.0):
    """
    Training transforms with data augmentation to improve generalization.
    
    Augmentations are designed to:
    1. Create size/scale variations (RandomResizedCrop) - helps model ignore resolution artifacts
    2. Simulate realistic transformations (rotation, flip) - improves robustness
    3. Vary color properties (ColorJitter) - prevents overfitting to color-based generator signatures
    4. Add blur (GaussianBlur) - helps distinguish real textures from AI artifacts
    
    Args:
        image_size: Target image size after augmentation (default 128)
        augmentation_strength: Controls intensity of augmentations (0.0-1.0+)
                             - 0.0: no augmentation (just resize + normalize)
                             - 1.0: standard augmentation (recommended)
                             - >1.0: stronger augmentation for high regularization
    
    Returns:
        transforms.Compose: Augmentation pipeline for training
    """
    assert augmentation_strength >= 0.0, "augmentation_strength must be >= 0.0"
    
    # Scale augmentation intensity with the strength parameter
    color_jitter_strength = 0.2 * augmentation_strength
    gaussian_blur_prob = 0.3 * augmentation_strength
    
    augmentations = [
        # Random crop and resize to handle generator-specific resolutions
        # Scale range [0.8, 1.0] ensures we see most images but with some cropping
        transforms.RandomResizedCrop(
            image_size,
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        
        # Horizontal flip for robustness (AI generators may have directional biases)
        transforms.RandomHorizontalFlip(p=0.5),
        
        # Color jitter to prevent overfitting to color-based generator signatures
        # Range controls: brightness, contrast, saturation, hue
        transforms.ColorJitter(
            brightness=color_jitter_strength,
            contrast=color_jitter_strength,
            saturation=color_jitter_strength,
            hue=0.1 * augmentation_strength,  # Smaller hue range (±0.1)
        ),
        
        # Slight rotation for robustness to orientation
        transforms.RandomRotation(degrees=5),
        
        # Gaussian blur helps model learn real texture properties vs AI artifacts
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))],
            p=gaussian_blur_prob,
        ),
    ]
    
    # Convert to tensor and normalize
    augmentations.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    
    return transforms.Compose(augmentations)


def get_eval_transforms(image_size=128):
    """
    Validation/test transforms with NO augmentation (deterministic).
    
    These transforms preserve the original image properties for unbiased evaluation:
    - Resize to standard size
    - Convert to tensor
    - Normalize with ImageNet statistics
    
    Args:
        image_size: Target image size (default 128)
    
    Returns:
        transforms.Compose: Deterministic evaluation pipeline
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_transforms(image_size=128, is_train=True, augmentation_strength=1.0):
    """
    Factory function for getting appropriate transforms based on mode.
    
    Args:
        image_size: Target image size (default 128)
        is_train: If True, return augmented transforms; if False, return deterministic transforms
        augmentation_strength: Controls intensity of training augmentations (only used if is_train=True)
    
    Returns:
        transforms.Compose: Appropriate transform pipeline for the mode
    """
    if is_train:
        return get_train_transforms(image_size, augmentation_strength)
    else:
        return get_eval_transforms(image_size)


def get_dataloaders(
    data_dir=".",
    batch_size=32,
    image_size=128,
    num_workers=0,
    augmentation_strength=1.0,
):
    """
    Create train and test dataloaders with appropriate transforms.
    
    Args:
        data_dir: Path to directory containing 'train_subset' and 'test_subset' folders
        batch_size: Batch size for dataloaders
        image_size: Target image size (default 128)
        num_workers: Number of workers for data loading
        augmentation_strength: Controls intensity of training augmentations (default 1.0)
    
    Returns:
        Tuple of (train_loader, test_loader, class_names)
    """
    data_path = Path(data_dir)
    train_dir = data_path / "train_subset"
    test_dir = data_path / "test_subset"

    # Training transforms: with augmentation
    train_transform = get_transforms(
        image_size,
        is_train=True,
        augmentation_strength=augmentation_strength,
    )
    
    # Test transforms: deterministic, no augmentation
    test_transform = get_transforms(image_size, is_train=False)

    train_dataset = ImageFolder(root=str(train_dir), transform=train_transform)
    test_dataset = ImageFolder(root=str(test_dir), transform=test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader, train_dataset.classes
