"""Basic model checks."""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.model import get_model


def check_forward(architecture):
    model = get_model(architecture=architecture, dropout_rate=0.3)
    x = torch.randn(2, 3, 128, 128)
    y = model(x)
    assert y.shape == (2, 2)


def check_train_step():
    model = get_model(architecture="baseline", dropout_rate=0.3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    x = torch.randn(4, 3, 128, 128)
    y = torch.randint(0, 2, (4,))
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()


def main():
    check_forward("baseline")
    check_forward("resnet18")
    check_train_step()
    print("Model checks passed.")


if __name__ == "__main__":
    main()
