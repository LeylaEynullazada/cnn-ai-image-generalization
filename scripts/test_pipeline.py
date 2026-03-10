"""Quick test that model and training run (uses dummy data)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.model import get_model
from src.train import train_one_epoch, validate

if __name__ == "__main__":
    model = get_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(4, 3, 128, 128)
    y = torch.randint(0, 2, (4,))
    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    train_one_epoch(model, loader, criterion, optimizer, "cpu")
    print("Skeleton test passed.")
