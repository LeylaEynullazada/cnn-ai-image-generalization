"""Evaluation utilities for real vs fake classification."""
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
)


def precision_recall_f1(y_true, y_pred, average="macro"):
    """Return precision, recall, F1 score."""
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average)
    return p, r, f1

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

def show_predictions(model, dataloader, device, class_names, num_examples=8):
    """Show sample images with predictions (correct=green, incorrect=red)."""
    import matplotlib.pyplot as plt
    model.eval()
    images_shown = 0
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.cpu()
            outputs = model(images)
            preds = outputs.argmax(1).cpu()
            for i in range(images.size(0)):
                if images_shown >= num_examples:
                    break
                ax = axes[images_shown]
                img = images[i].cpu().permute(1, 2, 0).numpy()
                img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)
                ax.imshow(img)
                true_label = list(class_names)[int(labels[i])]
                pred_label = list(class_names)[int(preds[i])]
                color = "green" if preds[i] == labels[i] else "red"
                ax.set_title(f"True: {true_label}\nPred: {pred_label}", color=color, fontsize=9)
                ax.axis("off")
                images_shown += 1
            if images_shown >= num_examples:
                break
    for j in range(images_shown, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    plt.show()
