"""Dataset loading for real vs fake image classification"""
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


def get_transforms(image_size=128, is_train=True):
    """Basic preprocessing: resize, to tensor, normalize."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_dataloaders(data_dir=".", batch_size=32, image_size=128, num_workers=0):
    
    data_path = Path(data_dir)
    train_dir = data_path / "train"
    test_dir = data_path / "test"

    transform = get_transforms(image_size)
    train_dataset = ImageFolder(root=str(train_dir), transform=transform)
    test_dataset = ImageFolder(root=str(test_dir), transform=transform)

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
