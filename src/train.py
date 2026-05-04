"""Training utilities with early stopping, checkpointing and reproducibility."""

import os
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm import tqdm


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch. Returns average loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Train")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        pbar.set_postfix(loss=loss.item(), acc=100.0 * correct / total)

    return running_loss / len(train_loader), 100.0 * correct / total


def validate(model, val_loader, criterion, device, return_preds: bool = False):
    """Validate model.

    If return_preds is True, also return (y_true, y_pred) for metric computation.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    ys_true = []
    ys_pred = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            if return_preds:
                ys_true.extend(labels.cpu().numpy().tolist())
                ys_pred.extend(predicted.cpu().numpy().tolist())

    avg_loss = running_loss / len(val_loader)
    acc = 100.0 * correct / total
    if return_preds:
        return avg_loss, acc, (np.array(ys_true), np.array(ys_pred))
    return avg_loss, acc


def train_full(
    model,
    train_loader,
    val_loader,
    num_epochs: int = 10,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
    save_path: str = "checkpoints/best.pth",
    weight_decay: float = 0.0,
    seed: int = 42,
    monitor: str = "val_loss",
    mode: str = "min",
    patience: int = 5,
    save_optimizer: bool = True,
    scheduler=None,
):
    """
    Full training loop with early stopping and checkpointing.

    Args:
        model: PyTorch model
        train_loader, val_loader: DataLoaders
        num_epochs: number of epochs
        lr: learning rate
        device: torch.device or None (auto-select)
        save_path: path to save best checkpoint
        seed: random seed for reproducibility
        monitor: 'val_loss' or 'val_f1'
        mode: 'min' for val_loss, 'max' for val_f1
        patience: epochs to wait for improvement
        save_optimizer: whether to save optimizer state
        scheduler: optional LR scheduler instance

    Returns:
        history dict
    """
    # reproducibility
    set_seed(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}

    best_score = float("inf") if mode == "min" else -float("inf")
    epochs_no_improve = 0
    d = os.path.dirname(save_path)
    if d:
        os.makedirs(d, exist_ok=True)

    for epoch in range(num_epochs):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

        if monitor == "val_f1":
            v_loss, v_acc, (y_true, y_pred) = validate(model, val_loader, criterion, device, return_preds=True)
            v_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        else:
            v_loss, v_acc = validate(model, val_loader, criterion, device, return_preds=False)
            v_f1 = None

        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)
        history["val_f1"].append(v_f1)

        # display
        if v_f1 is not None:
            print(
                f"Epoch {epoch+1}/{num_epochs} | Train Loss: {t_loss:.4f} Acc: {t_acc:.2f}% | Val Loss: {v_loss:.4f} Acc: {v_acc:.2f}% F1: {v_f1:.4f}"
            )
        else:
            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {t_loss:.4f} Acc: {t_acc:.2f}% | Val Loss: {v_loss:.4f} Acc: {v_acc:.2f}%")

        # compute current score for monitoring
        current = v_loss if monitor == "val_loss" else (v_f1 if v_f1 is not None else v_acc)

        improved = (current < best_score) if mode == "min" else (current > best_score)
        if improved:
            best_score = current
            epochs_no_improve = 0
            # save checkpoint
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "history": history,
            }
            if save_optimizer:
                checkpoint["optimizer_state_dict"] = optimizer.state_dict()
            if scheduler is not None:
                checkpoint["scheduler_state_dict"] = scheduler.state_dict()
            torch.save(checkpoint, save_path)
        else:
            epochs_no_improve += 1

        # step scheduler if present
        if scheduler is not None:
            try:
                scheduler.step()
            except Exception:
                pass

        # early stopping
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered. No improvement for {patience} epochs.")
            break

    return history
