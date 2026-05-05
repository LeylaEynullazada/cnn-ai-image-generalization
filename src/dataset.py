"""Dataset and transform helpers."""

import os
from pathlib import Path

import torch
from PIL import ImageFile
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

ImageFile.LOAD_TRUNCATED_IMAGES = True


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size=128, augmentation_strength=1.0):
    """Build training transforms."""
    assert augmentation_strength >= 0.0, "augmentation_strength must be >= 0.0"

    color_jitter_strength = 0.2 * augmentation_strength
    gaussian_blur_prob = 0.3 * augmentation_strength

    return transforms.Compose([
        transforms.RandomResizedCrop(
            image_size,
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=color_jitter_strength,
            contrast=color_jitter_strength,
            saturation=color_jitter_strength,
            hue=0.1 * augmentation_strength,
        ),
        transforms.RandomRotation(degrees=5),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))],
            p=gaussian_blur_prob,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms(image_size=128):
    """Build evaluation transforms."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_transforms(image_size=128, is_train=True, augmentation_strength=1.0):
    """Return train or evaluation transforms."""
    if is_train:
        return get_train_transforms(image_size, augmentation_strength)
    return get_eval_transforms(image_size)


def get_dataloaders(
    data_dir=".",
    batch_size=32,
    image_size=128,
    num_workers=None,
    augmentation_strength=1.0,
):
    """Create train and validation dataloaders."""
    data_path = Path(data_dir)
    train_dir = data_path / "train_subset"
    test_dir = data_path / "test_subset"

    train_dataset = ImageFolder(
        root=str(train_dir),
        transform=get_transforms(image_size, is_train=True, augmentation_strength=augmentation_strength),
    )
    test_dataset = ImageFolder(
        root=str(test_dir),
        transform=get_transforms(image_size, is_train=False),
    )

    if num_workers is None:
        cpu_count = os.cpu_count() or 1
        num_workers = min(4, max(0, cpu_count - 1))

    pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    return train_loader, test_loader, train_dataset.classes
