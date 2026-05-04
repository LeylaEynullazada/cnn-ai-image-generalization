"""
Simple experiment runner: loops over architectures, augmentation strengths, and dropout rates.
Saves aggregated results to results/experiments.csv

Note: Add leave-one-generator-out logic where indicated below.
"""
import csv
import json
import os
import random
import shutil
import sys
from pathlib import Path
from time import time
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import get_model
from src.dataset import get_dataloaders
from src.train import train_full
from src.evaluate import evaluate_and_report


KAGGLE_INPUT_DIR = Path("/kaggle/input/ai-generated-images-vs-real-images")
KAGGLE_WORKING_DATA_DIR = Path("/kaggle/working/data")
KAGGLE_WORKING_RESULTS_DIR = Path("/kaggle/working/results")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_kaggle_environment():
    return Path("/kaggle/input").exists() and Path("/kaggle/working").exists()


def detect_class_name(file_path):
    parts = [part.lower() for part in file_path.parts]
    if any(part in {"fake", "ai", "generated", "ai-generated", "ai_generated"} for part in parts):
        return "fake"
    if any(part in {"real", "human", "original"} for part in parts):
        return "real"
    return None


def collect_images_by_class(root_dir):
    class_to_files = {"fake": [], "real": []}
    if not root_dir.exists():
        return class_to_files

    for file_path in root_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
            class_name = detect_class_name(file_path.relative_to(root_dir))
            if class_name in class_to_files:
                class_to_files[class_name].append(file_path)
    return class_to_files


def sample_and_copy_subset(source_dir, target_dir, train_count=300, test_count=100, seed=42):
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    train_dir = target_path / "train_subset"
    test_dir = target_path / "test_subset"

    if (train_dir.exists() and test_dir.exists() and any(train_dir.iterdir()) and any(test_dir.iterdir())):
        return target_path

    class_to_files = collect_images_by_class(source_path)
    if not any(class_to_files.values()):
        return None

    rng = random.Random(seed)
    for class_name, files in class_to_files.items():
        rng.shuffle(files)
        selected_train = files[: min(train_count, len(files))]
        remaining = files[len(selected_train):]
        selected_test = remaining[: min(test_count, len(remaining))]

        for split_name, selected_files in (("train_subset", selected_train), ("test_subset", selected_test)):
            split_dir = target_path / split_name / class_name
            split_dir.mkdir(parents=True, exist_ok=True)
            for index, file_path in enumerate(selected_files):
                destination = split_dir / f"{class_name}_{index:04d}{file_path.suffix.lower()}"
                if not destination.exists():
                    shutil.copy2(file_path, destination)

    return target_path


def prepare_data_dir(data_dir):
    data_path = Path(data_dir)
    train_dir = data_path / "train_subset"
    test_dir = data_path / "test_subset"
    if train_dir.exists() and test_dir.exists():
        return data_path

    if is_kaggle_environment():
        prepared = sample_and_copy_subset(KAGGLE_INPUT_DIR, KAGGLE_WORKING_DATA_DIR)
        if prepared is not None:
            return prepared

    return data_path


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def run():
    default_data_dir = str(KAGGLE_WORKING_DATA_DIR if is_kaggle_environment() else Path("data"))
    default_output_dir = str(KAGGLE_WORKING_RESULTS_DIR if is_kaggle_environment() else Path("results"))

    parser = argparse.ArgumentParser(description="Run experiments sweep")
    parser.add_argument("--data_dir", type=str, default=default_data_dir, help="Path with train_subset/test_subset folders")
    parser.add_argument("--epochs_baseline", type=int, default=15)
    parser.add_argument("--epochs_resnet", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=str, default=default_output_dir)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--smoke", action="store_true", help="Run a quick smoke test (1 epoch, 1 combo)")
    args = parser.parse_args()

    architectures = ["baseline", "resnet18"]
    augmentation_strengths = [0.0, 0.5, 1.0]
    dropout_rates = [0.3, 0.5]

    data_path = prepare_data_dir(args.data_dir)
    results_dir = Path(args.output_dir)
    ensure_dir(results_dir)
    csv_path = results_dir / "experiments.csv"

    header = [
        "architecture",
        "augmentation_strength",
        "dropout_rate",
        "train_time_sec",
        "final_val_loss",
        "final_val_acc",
        "final_precision",
        "final_recall",
        "final_f1",
        "roc_auc",
        "checkpoint_path",
        "history_path",
    ]

    # Write header if file doesn't exist (preserve existing file)
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    # Smoke test overrides
    if args.smoke:
        architectures = ["baseline"]
        augmentation_strengths = [0.5]
        dropout_rates = [0.3]


    for arch in architectures:
        for aug in augmentation_strengths:
            for drop in dropout_rates:
                print(f"Running: arch={arch}, aug={aug}, drop={drop}")
                # Create dataloaders (expects train_subset/ and test_subset/ under data/)
                # If smoke mode or data folders missing, create tiny synthetic loaders for quick testing
                use_dummy = args.smoke or not (data_path.exists() and (data_path / "train_subset").exists() and (data_path / "test_subset").exists())
                if use_dummy:
                    # synthetic small dataset
                    from torch.utils.data import DataLoader, TensorDataset
                    import torch
                    x = torch.randn(16, 3, 128, 128)
                    y = torch.randint(0, 2, (16,))
                    train_loader = DataLoader(TensorDataset(x, y), batch_size=args.batch_size or 8)
                    val_loader = DataLoader(TensorDataset(x, y), batch_size=args.batch_size or 8)
                    class_names = ["fake", "real"]
                else:
                    train_loader, val_loader, class_names = get_dataloaders(
                        data_dir=str(data_path), batch_size=args.batch_size, augmentation_strength=aug
                    )

                # Create model
                model = get_model(architecture=arch, dropout_rate=drop)

                # Where to save checkpoint for this run
                run_dir = results_dir / f"{arch}_aug{aug}_drop{drop}"
                ensure_dir(run_dir)
                checkpoint_path = run_dir / "best.pth"

                # Choose realistic defaults per architecture unless overridden
                if arch == "baseline":
                    num_epochs = args.epochs_baseline
                    default_lr = 1e-3
                else:
                    num_epochs = args.epochs_resnet
                    default_lr = 3e-4
                lr = args.learning_rate if args.learning_rate is not None else default_lr

                # Smoke override
                if args.smoke:
                    num_epochs = 1

                print(f"Starting training: arch={arch}, aug={aug}, drop={drop}, epochs={num_epochs}, lr={lr}")
                t0 = time()
                history = train_full(
                    model,
                    train_loader,
                    val_loader,
                    num_epochs=num_epochs,
                    lr=lr,
                    device=None,
                    save_path=str(checkpoint_path),
                    seed=42,
                    monitor="val_f1",
                    mode="max",
                    patience=args.patience,
                    save_optimizer=True,
                    weight_decay=args.weight_decay,
                )
                train_time = time() - t0

                # Save training history
                history_path = run_dir / "history.json"
                try:
                    with open(history_path, "w") as hf:
                        json.dump(history, hf)
                except Exception:
                    # fallback: ignore history saving errors
                    history_path = None

                # Evaluate using evaluate_and_report (current model state)
                results = evaluate_and_report(
                    model,
                    val_loader,
                    device=None,
                    class_names=class_names,
                    scenario_name=f"{arch}_aug{aug}_drop{drop}",
                    save_dir=str(run_dir),
                    plot_results=True,
                )

                # Extract final metrics
                final_val_loss = history.get("val_loss", [None])[-1]
                final_val_acc = None
                final_precision = None
                final_recall = None
                final_f1 = None
                roc_auc = None
                try:
                    metrics = results.get("metrics", {})
                    final_val_acc = metrics.get("accuracy")
                    final_precision = metrics.get("precision")
                    final_recall = metrics.get("recall")
                    final_f1 = metrics.get("f1")
                except Exception:
                    pass
                try:
                    roc = results.get("roc_metrics")
                    if roc is not None:
                        roc_auc = roc.get("roc_auc")
                except Exception:
                    pass

                # Append to CSV
                with open(csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        arch,
                        aug,
                        drop,
                        f"{train_time:.2f}",
                        final_val_loss,
                        final_val_acc,
                        final_precision,
                        final_recall,
                        final_f1,
                        roc_auc,
                        str(checkpoint_path),
                        str(history_path) if history_path is not None else "",
                    ])

                print(f"Completed: arch={arch}, aug={aug}, drop={drop} | val_f1={final_f1} val_acc={final_val_acc} roc_auc={roc_auc}")

    print(f"All experiments complete. Results saved to: {csv_path}")


if __name__ == "__main__":
    run()
