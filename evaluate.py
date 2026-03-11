"""Evaluation utilities for real vs fake classification."""
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def evaluate(model, dataloader, device):
    """Get all predictions and labels. Returns: y_true, y_pred, y_probs."""
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def get_metrics(y_true, y_pred, class_names=None):
    """Compute accuracy, confusion matrix, classification report."""
    if class_names is None:
        class_names = ["fake", "real"]
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names)
    return acc, cm, report


def plot_confusion_matrix(cm, class_names=None):
    """Plot confusion matrix."""
    import matplotlib.pyplot as plt
    if class_names is None:
        class_names = ["fake", "real"]
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar()
    plt.xticks(range(len(class_names)), class_names)
    plt.yticks(range(len(class_names)), class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.show()
