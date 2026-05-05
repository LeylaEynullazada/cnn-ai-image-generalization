"""Basic checks for dataset transforms."""

import sys
from pathlib import Path

import torch
from torchvision.transforms.functional import to_pil_image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.dataset import get_eval_transforms, get_train_transforms


def main():
    image = to_pil_image(torch.rand(3, 128, 128))

    train_transform = get_train_transforms(image_size=128, augmentation_strength=0.5)
    eval_transform = get_eval_transforms(image_size=128)

    train_output = train_transform(image)
    eval_output = eval_transform(image)

    assert train_output.shape == (3, 128, 128)
    assert eval_output.shape == (3, 128, 128)
    assert train_output.dtype == torch.float32
    assert eval_output.dtype == torch.float32

    print("Dataset transform checks passed.")


if __name__ == "__main__":
    main()
