"""Evaluation utilities for real vs AI-generated image classification.

This module provides comprehensive evaluation metrics and visualization for binary
classification tasks. It supports both standard validation/test evaluation and
specialized scenarios like leave-one-generator-out experiments.

Key capabilities:
- Multiple metrics: accuracy, precision, recall, F1, ROC-AUC, confusion matrix
- Per-class metrics for detailed analysis
- Visualization: confusion matrix plots, ROC curves
- File saving for reproducible results
- Flexible design for different evaluation scenarios
"""
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)


def evaluate(model, dataloader, device):
    """
    Get all predictions and labels from a dataloader.
    
    Args:
        model: PyTorch model in evaluation mode
        dataloader: DataLoader for evaluation
        device: Device to run inference on (cpu or cuda)
    
    Returns:
        Tuple of (y_true, y_pred, y_probs):
        - y_true: Ground truth labels (numpy array)
        - y_pred: Predicted labels (numpy array)
        - y_probs: Prediction probabilities for all classes (numpy array)
    """
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs)
    )


def get_metrics(y_true, y_pred, class_names=None):
    """
    Compute comprehensive metrics for binary classification.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names (default: ["fake", "real"])
    
    Returns:
        Dictionary containing:
        - accuracy: Overall accuracy
        - precision: Macro-averaged precision
        - recall: Macro-averaged recall
        - f1: Macro-averaged F1-score
        - precision_per_class: Per-class precision dict
        - recall_per_class: Per-class recall dict
        - f1_per_class: Per-class F1-score dict
        - confusion_matrix: Confusion matrix
        - classification_report: Full classification report string
    """
    if class_names is None:
        class_names = ["fake", "real"]
    
    # Overall metrics
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    
    # Per-class metrics
    p_per_class, r_per_class, f1_per_class, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=None, zero_division=0
    )
    
    # Confusion matrix and report
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    
    # Create per-class dictionaries
    precision_per_class = {class_names[i]: p_per_class[i] for i in range(len(class_names))}
    recall_per_class = {class_names[i]: r_per_class[i] for i in range(len(class_names))}
    f1_per_class = {class_names[i]: f1_per_class[i] for i in range(len(class_names))}
    
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "precision_per_class": precision_per_class,
        "recall_per_class": recall_per_class,
        "f1_per_class": f1_per_class,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def get_roc_metrics(y_true, y_probs, class_names=None):
    """
    Compute ROC-AUC metrics for binary classification.
    
    Args:
        y_true: Ground truth labels
        y_probs: Prediction probabilities from softmax (shape: [N, num_classes])
        class_names: List of class names (default: ["fake", "real"])
    
    Returns:
        Dictionary containing:
        - roc_auc: ROC-AUC score
        - fpr: False positive rate array
        - tpr: True positive rate array
        - thresholds: Threshold array for ROC curve
    """
    if class_names is None:
        class_names = ["fake", "real"]
    
    # For binary classification, use probability of positive class (index 1)
    y_probs_positive = y_probs[:, 1]
    roc_auc = roc_auc_score(y_true, y_probs_positive)
    fpr, tpr, thresholds = roc_curve(y_true, y_probs_positive)
    
    return {
        "roc_auc": roc_auc,
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
    }


def print_metrics(metrics, roc_metrics=None):
    """
    Pretty-print evaluation metrics to console.
    
    Args:
        metrics: Dictionary from get_metrics()
        roc_metrics: Optional dictionary from get_roc_metrics()
    """
    print("\n" + "="*70)
    print("EVALUATION METRICS")
    print("="*70)
    
    print(f"\nOverall Performance:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    
    if roc_metrics:
        print(f"\nROC Analysis:")
        print(f"  ROC-AUC:   {roc_metrics['roc_auc']:.4f}")
    
    print(f"\nPer-Class Metrics:")
    print(f"  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
    for class_name in metrics['precision_per_class'].keys():
        p = metrics['precision_per_class'][class_name]
        r = metrics['recall_per_class'][class_name]
        f1 = metrics['f1_per_class'][class_name]
        print(f"  {class_name:<12} {p:>10.4f} {r:>10.4f} {f1:>10.4f}")
    
    print(f"\nConfusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"  {'':>8} {'Predicted Fake':>18} {'Predicted Real':>18}")
    print(f"  {'Actual Fake':>8} {cm[0,0]:>18} {cm[0,1]:>18}")
    print(f"  {'Actual Real':>8} {cm[1,0]:>18} {cm[1,1]:>18}")
    
    print(f"\nDetailed Classification Report:")
    print(metrics['classification_report'])
    print("="*70 + "\n")


def plot_confusion_matrix(cm, class_names=None, save_path=None):
    """
    Plot and optionally save confusion matrix.
    
    Args:
        cm: Confusion matrix from sklearn
        class_names: List of class names (default: ["fake", "real"])
        save_path: Optional path to save the plot (e.g., "confusion_matrix.png")
    """
    import matplotlib.pyplot as plt
    
    if class_names is None:
        class_names = ["fake", "real"]
    
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar()
    plt.xticks(range(len(class_names)), class_names, fontsize=12)
    plt.yticks(range(len(class_names)), class_names, fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
    
    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=16, fontweight="bold"
            )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ Confusion matrix saved to: {save_path}")
    
    return plt.gcf()


def plot_roc_curve(fpr, tpr, roc_auc, save_path=None):
    """
    Plot and optionally save ROC curve.
    
    Args:
        fpr: False positive rate array
        tpr: True positive rate array
        roc_auc: ROC-AUC score
        save_path: Optional path to save the plot (e.g., "roc_curve.png")
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(8, 7))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Classifier")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve - Binary Classification", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ ROC curve saved to: {save_path}")
    
    return plt.gcf()


def show_predictions(model, dataloader, device, class_names, num_examples=8):
    """
    Display sample images with predictions (correct=green, incorrect=red).
    
    Args:
        model: PyTorch model
        dataloader: DataLoader with images and labels
        device: Device to run inference on
        class_names: List of class names
        num_examples: Number of examples to show
    """
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
    return fig


def evaluate_and_report(
    model,
    dataloader,
    device,
    class_names=None,
    scenario_name="Evaluation",
    save_dir=None,
    plot_results=True,
):
    """
    Complete evaluation pipeline: compute metrics, print report, optionally save visualizations.
    
    This function is designed for both standard validation and specialized scenarios like
    leave-one-generator-out experiments.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation
        device: Device to run inference on (cpu or cuda)
        class_names: List of class names (default: ["fake", "real"])
        scenario_name: Name of evaluation scenario for display (e.g., "Leave-one-out: DALL-E")
        save_dir: Directory to save plots. If None, plots are not saved.
        plot_results: If True, return matplotlib figures; if False, skip plot generation
    
    Returns:
        Dictionary containing:
        - metrics: Metrics dictionary from get_metrics()
        - roc_metrics: ROC metrics dictionary from get_roc_metrics()
        - y_true: Ground truth labels
        - y_pred: Predicted labels
        - y_probs: Prediction probabilities
        - figures: Dictionary of matplotlib figures (if plot_results=True)
    """
    if class_names is None:
        class_names = ["fake", "real"]
    
    # Evaluate model
    y_true, y_pred, y_probs = evaluate(model, dataloader, device)
    
    # Compute metrics
    metrics = get_metrics(y_true, y_pred, class_names)
    roc_metrics = get_roc_metrics(y_true, y_probs, class_names)
    
    # Print results
    print(f"\n{'='*70}")
    print(f"EVALUATION: {scenario_name}")
    print(f"{'='*70}")
    print_metrics(metrics, roc_metrics)
    
    # Generate and save visualizations
    figures = {}
    if plot_results:
        # Confusion matrix plot
        fig_cm = plot_confusion_matrix(
            metrics["confusion_matrix"],
            class_names,
            save_path=Path(save_dir) / "confusion_matrix.png" if save_dir else None
        )
        figures["confusion_matrix"] = fig_cm
        
        # ROC curve plot
        fig_roc = plot_roc_curve(
            roc_metrics["fpr"],
            roc_metrics["tpr"],
            roc_metrics["roc_auc"],
            save_path=Path(save_dir) / "roc_curve.png" if save_dir else None
        )
        figures["roc_curve"] = fig_roc
    
    return {
        "metrics": metrics,
        "roc_metrics": roc_metrics,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_probs": y_probs,
        "figures": figures,
    }
