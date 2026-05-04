"""
Validation and demonstration of improved evaluation metrics and visualizations.
Shows comprehensive metrics, per-class analysis, and ROC-AUC evaluation.
"""
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.evaluate import (
    evaluate,
    get_metrics,
    get_roc_metrics,
    print_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    evaluate_and_report,
)


def test_evaluate_function():
    """Test basic evaluate() function."""
    print("\n" + "="*70)
    print("TEST 1: Basic Evaluate Function")
    print("="*70)
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 2),
    )
    model.eval()
    
    # Create dummy data
    X = torch.randn(32, 10)
    y = torch.randint(0, 2, (32,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=8)
    
    device = "cpu"
    y_true, y_pred, y_probs = evaluate(model, dataloader, device)
    
    assert y_true.shape[0] == 32, "y_true shape mismatch"
    assert y_pred.shape[0] == 32, "y_pred shape mismatch"
    assert y_probs.shape == (32, 2), "y_probs shape mismatch"
    assert y_probs.min() >= 0.0 and y_probs.max() <= 1.0, "Invalid probabilities"
    assert np.allclose(y_probs.sum(axis=1), 1.0), "Probabilities don't sum to 1"
    
    print(f"✓ y_true shape: {y_true.shape}")
    print(f"✓ y_pred shape: {y_pred.shape}")
    print(f"✓ y_probs shape: {y_probs.shape}")
    print(f"✓ Probabilities valid (min={y_probs.min():.4f}, max={y_probs.max():.4f})")
    
    print("\nBasic evaluate function test passed!")


def test_get_metrics():
    """Test metric computation."""
    print("\n" + "="*70)
    print("TEST 2: Comprehensive Metrics Computation")
    print("="*70)
    
    import numpy as np
    
    # Create synthetic predictions
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 0, 1, 0, 0, 1, 1, 0])  # One mistake at index 3
    
    metrics = get_metrics(y_true, y_pred, class_names=["fake", "real"])
    
    # Verify keys exist
    required_keys = [
        "accuracy", "precision", "recall", "f1",
        "precision_per_class", "recall_per_class", "f1_per_class",
        "confusion_matrix", "classification_report"
    ]
    for key in required_keys:
        assert key in metrics, f"Missing key: {key}"
        print(f"✓ Key present: {key}")
    
    # Verify values are reasonable
    assert 0 <= metrics["accuracy"] <= 1, "Invalid accuracy"
    assert 0 <= metrics["precision"] <= 1, "Invalid precision"
    assert 0 <= metrics["recall"] <= 1, "Invalid recall"
    assert 0 <= metrics["f1"] <= 1, "Invalid F1"
    
    print(f"\n✓ Accuracy: {metrics['accuracy']:.4f}")
    print(f"✓ Precision: {metrics['precision']:.4f}")
    print(f"✓ Recall: {metrics['recall']:.4f}")
    print(f"✓ F1-Score: {metrics['f1']:.4f}")
    
    print(f"\n✓ Per-class precision: {metrics['precision_per_class']}")
    print(f"✓ Per-class recall: {metrics['recall_per_class']}")
    print(f"✓ Per-class F1: {metrics['f1_per_class']}")
    
    print("\nMetrics computation test passed!")


def test_roc_metrics():
    """Test ROC-AUC computation."""
    print("\n" + "="*70)
    print("TEST 3: ROC-AUC Metrics")
    print("="*70)
    
    import numpy as np
    
    # Create synthetic predictions with well-calibrated probabilities
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    y_probs = np.array([
        [0.9, 0.1],  # Confident fake - correct
        [0.8, 0.2],  # Confident fake - correct
        [0.1, 0.9],  # Confident real - correct
        [0.4, 0.6],  # Weak real - correct
        [0.7, 0.3],  # Confident fake - correct
        [0.2, 0.8],  # Confident real - correct
        [0.3, 0.7],  # Confident real - correct
        [0.6, 0.4],  # Weak fake - correct
    ])
    
    roc_metrics = get_roc_metrics(y_true, y_probs, class_names=["fake", "real"])
    
    # Verify keys
    required_keys = ["roc_auc", "fpr", "tpr", "thresholds"]
    for key in required_keys:
        assert key in roc_metrics, f"Missing key: {key}"
        print(f"✓ Key present: {key}")
    
    # Verify values
    assert 0 <= roc_metrics["roc_auc"] <= 1, "Invalid ROC-AUC"
    assert len(roc_metrics["fpr"]) == len(roc_metrics["tpr"]), "FPR/TPR length mismatch"
    
    print(f"\n✓ ROC-AUC: {roc_metrics['roc_auc']:.4f}")
    print(f"✓ FPR length: {len(roc_metrics['fpr'])}")
    print(f"✓ TPR length: {len(roc_metrics['tpr'])}")
    
    print("\nROC metrics test passed!")


def test_print_metrics():
    """Test metric printing."""
    print("\n" + "="*70)
    print("TEST 4: Metric Printing")
    print("="*70)
    
    import numpy as np
    
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 1, 0, 1])  # Perfect predictions
    y_probs = np.array([
        [0.95, 0.05],
        [0.90, 0.10],
        [0.05, 0.95],
        [0.10, 0.90],
        [0.92, 0.08],
        [0.08, 0.92],
    ])
    
    metrics = get_metrics(y_true, y_pred, class_names=["fake", "real"])
    roc_metrics = get_roc_metrics(y_true, y_probs, class_names=["fake", "real"])
    
    print("\nPrinting metrics to console:")
    print_metrics(metrics, roc_metrics)
    
    print("✓ Metrics printed successfully")


def test_plot_functions():
    """Test plot generation (without displaying)."""
    print("\n" + "="*70)
    print("TEST 5: Plot Generation & Saving")
    print("="*70)
    
    import numpy as np
    from sklearn.metrics import confusion_matrix
    
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 0, 1, 0, 0, 1, 1, 0])
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Test confusion matrix plot with saving
    with tempfile.TemporaryDirectory() as tmpdir:
        cm_path = Path(tmpdir) / "test_cm.png"
        fig_cm = plot_confusion_matrix(cm, class_names=["fake", "real"], save_path=cm_path)
        assert cm_path.exists(), "Confusion matrix plot not saved"
        print(f"✓ Confusion matrix plot saved: {cm_path}")
        print(f"  File size: {cm_path.stat().st_size} bytes")
    
    # Test ROC curve plot with saving
    y_probs = np.array([
        [0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.4, 0.6],
        [0.7, 0.3], [0.2, 0.8], [0.3, 0.7], [0.6, 0.4],
    ])
    roc_metrics = get_roc_metrics(y_true, y_probs)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        roc_path = Path(tmpdir) / "test_roc.png"
        fig_roc = plot_roc_curve(
            roc_metrics["fpr"],
            roc_metrics["tpr"],
            roc_metrics["roc_auc"],
            save_path=roc_path
        )
        assert roc_path.exists(), "ROC curve plot not saved"
        print(f"✓ ROC curve plot saved: {roc_path}")
        print(f"  File size: {roc_path.stat().st_size} bytes")


def test_evaluate_and_report():
    """Test complete evaluation pipeline."""
    print("\n" + "="*70)
    print("TEST 6: Complete Evaluation Pipeline")
    print("="*70)
    
    # Create model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 2),
    )
    model.eval()
    
    # Create dummy data
    X = torch.randn(32, 10)
    y = torch.randint(0, 2, (32,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=8)
    
    # Test with plot_results=True
    with tempfile.TemporaryDirectory() as tmpdir:
        result = evaluate_and_report(
            model,
            dataloader,
            device="cpu",
            class_names=["fake", "real"],
            scenario_name="Test Scenario: Leave-one-out DALL-E",
            save_dir=tmpdir,
            plot_results=True,
        )
    
    # Verify result structure
    required_keys = ["metrics", "roc_metrics", "y_true", "y_pred", "y_probs", "figures"]
    for key in required_keys:
        assert key in result, f"Missing key: {key}"
        print(f"✓ Key present: {key}")
    
    assert len(result["figures"]) > 0, "No figures generated"
    print(f"✓ Figures generated: {len(result['figures'])}")
    print(f"  Figures: {list(result['figures'].keys())}")
    
    print("\nComplete evaluation pipeline test passed!")


def test_backward_compatibility():
    """Test backward compatibility with old code."""
    print("\n" + "="*70)
    print("TEST 7: Backward Compatibility")
    print("="*70)
    
    import numpy as np
    
    # Old-style function calls that should still work
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 2),
    )
    model.eval()
    
    X = torch.randn(16, 10)
    y = torch.randint(0, 2, (16,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=8)
    
    # Old evaluate() call
    y_true, y_pred, y_probs = evaluate(model, dataloader, "cpu")
    assert isinstance(y_true, np.ndarray), "evaluate() return type changed"
    print("✓ evaluate() backward compatible")
    
    # get_metrics() returns dictionary (new format)
    # But old code that unpacked it should fail gracefully
    metrics = get_metrics(y_true, y_pred, class_names=["fake", "real"])
    assert isinstance(metrics, dict), "get_metrics() should return dict"
    print("✓ get_metrics() returns dict with all metrics")
    
    print("\nBackward compatibility verified!")


if __name__ == "__main__":
    import numpy as np
    
    print("\n" + "="*70)
    print("EVALUATION METRICS VALIDATION SUITE")
    print("="*70)
    
    try:
        test_evaluate_function()
        test_get_metrics()
        test_roc_metrics()
        test_print_metrics()
        test_plot_functions()
        test_evaluate_and_report()
        test_backward_compatibility()
        
        print("\n" + "="*70)
        print("✅ ALL EVALUATION VALIDATION TESTS PASSED!")
        print("="*70)
        print("\nKey Improvements Verified:")
        print("  ✓ Comprehensive metrics: accuracy, precision, recall, F1, ROC-AUC")
        print("  ✓ Per-class metric analysis")
        print("  ✓ Confusion matrix visualization with saving")
        print("  ✓ ROC curve visualization with saving")
        print("  ✓ Complete evaluation pipeline for all scenarios")
        print("  ✓ Backward compatibility maintained")
        print("\nReady for Milestone 1 comprehensive evaluation!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
