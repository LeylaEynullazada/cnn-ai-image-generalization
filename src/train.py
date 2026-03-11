"""Training loop for real vs fake classification."""

import torch
import torch.nn as nn
from tqdm import tqdm
import os

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


def validate(model, val_loader, criterion, device):
    """Validate model. Returns average loss and accuracy."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / len(val_loader), 100.0 * correct / total

def train_full(model, train_loader, val_loader, num_epochs=10, lr=1e-3, device=None, save_path="checkpoints/best.pth"):
    """
    Full training loop with checkpointing.
    Returns: history dict with train_loss, train_acc, val_loss, val_acc per epoch.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    d = os.path.dirname(save_path)
    if d:
        os.makedirs(d, exist_ok=True)

    for epoch in range(num_epochs):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc = validate(model, val_loader, criterion, device)
        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {t_loss:.4f} Acc: {t_acc:.2f}% | Val Loss: {v_loss:.4f} Acc: {v_acc:.2f}%")
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "val_acc": v_acc}, save_path)
    return history
