"""Basic evaluation checks."""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.evaluate import evaluate_and_report


def main():
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(10, 2),
    )
    x = torch.randn(16, 10)
    y = torch.randint(0, 2, (16,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)

    results = evaluate_and_report(
        model,
        loader,
        device="cpu",
        class_names=["fake", "real"],
        scenario_name="validation_check",
        save_dir=None,
        plot_results=False,
    )

    assert "metrics" in results
    assert "roc_metrics" in results
    print("Evaluation checks passed.")


if __name__ == "__main__":
    main()
