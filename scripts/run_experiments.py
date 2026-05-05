"""Run a small experiment sweep and write metrics, checkpoints, and reports."""
import csv
import json
import os
import random
import shutil
import sys
from pathlib import Path
from time import time
import argparse
from html import escape

import torch

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


def prepare_data_dir(data_dir, train_count=300, test_count=100, seed=42):
    data_path = Path(data_dir)
    train_dir = data_path / "train_subset"
    test_dir = data_path / "test_subset"
    if train_dir.exists() and test_dir.exists():
        return data_path

    if is_kaggle_environment():
        prepared = sample_and_copy_subset(
            KAGGLE_INPUT_DIR,
            KAGGLE_WORKING_DATA_DIR,
            train_count=train_count,
            test_count=test_count,
            seed=seed,
        )
        if prepared is not None:
            return prepared

    return data_path


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def parse_csv_list(value, cast=str):
    if value is None:
        return None
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def load_best_checkpoint(model, checkpoint_path, device):
    if not checkpoint_path.exists():
        return model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    return model


def save_run_report(results, report_dir):
    ensure_dir(report_dir)
    metrics = results.get("metrics", {})
    roc_metrics = results.get("roc_metrics") or {}

    def json_value(value):
        return value.item() if hasattr(value, "item") else value

    report_path = report_dir / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(metrics.get("classification_report", ""))

    summary = {
        "accuracy": json_value(metrics.get("accuracy")),
        "precision": json_value(metrics.get("precision")),
        "recall": json_value(metrics.get("recall")),
        "f1": json_value(metrics.get("f1")),
        "roc_auc": json_value(roc_metrics.get("roc_auc")),
    }
    summary_path = report_dir / "metrics_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return report_path, summary_path


def format_metric(value):
    if value in (None, ""):
        return ""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def relative_html_path(path, base_dir):
    try:
        return Path(os.path.relpath(path, base_dir)).as_posix()
    except ValueError:
        return Path(path).as_posix()


def write_notebook_report(csv_path, results_dir):
    if not csv_path.exists():
        return None

    rows = []
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    report_path = results_dir / "notebook_report.html"
    metric_columns = [
        "architecture",
        "augmentation_strength",
        "dropout_rate",
        "train_time_sec",
        "final_val_acc",
        "final_precision",
        "final_recall",
        "final_f1",
        "roc_auc",
    ]

    html_parts = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<style>",
        "body{font-family:Arial,sans-serif;margin:24px;color:#222}",
        "table{border-collapse:collapse;width:100%;margin-bottom:24px}",
        "th,td{border:1px solid #ddd;padding:8px;text-align:left;font-size:13px}",
        "th{background:#f3f3f3}",
        ".run{margin:28px 0;padding-top:8px;border-top:2px solid #ddd}",
        ".figs{display:flex;gap:20px;flex-wrap:wrap}",
        ".fig{max-width:460px}",
        ".fig img{max-width:100%;border:1px solid #ddd}",
        "pre{background:#f7f7f7;padding:12px;overflow:auto;font-size:12px}",
        "</style></head><body>",
        "<h1>Experiment Report</h1>",
        f"<p>Runs completed: {len(rows)}</p>",
        "<h2>Metrics</h2>",
        "<table><thead><tr>",
    ]

    html_parts.extend(f"<th>{escape(column)}</th>" for column in metric_columns)
    html_parts.append("</tr></thead><tbody>")
    for row in rows:
        html_parts.append("<tr>")
        for column in metric_columns:
            html_parts.append(f"<td>{escape(format_metric(row.get(column)))}</td>")
        html_parts.append("</tr>")
    html_parts.append("</tbody></table>")

    for row in rows:
        run_name = f"{row.get('architecture')}_aug{row.get('augmentation_strength')}_drop{row.get('dropout_rate')}"
        report_dir_value = row.get("report_dir") or str(results_dir / "reports" / run_name)
        report_dir = Path(report_dir_value)
        cm_path = report_dir / "confusion_matrix.png"
        roc_path = report_dir / "roc_curve.png"
        text_path = report_dir / "classification_report.txt"

        html_parts.append(f"<div class='run'><h2>{escape(run_name)}</h2>")
        html_parts.append("<div class='figs'>")
        if cm_path.exists():
            rel_path = relative_html_path(cm_path, report_path.parent)
            html_parts.append(f"<div class='fig'><h3>Confusion Matrix</h3><img src='{escape(rel_path)}'></div>")
        if roc_path.exists():
            rel_path = relative_html_path(roc_path, report_path.parent)
            html_parts.append(f"<div class='fig'><h3>ROC Curve</h3><img src='{escape(rel_path)}'></div>")
        html_parts.append("</div>")
        if text_path.exists():
            with open(text_path) as f:
                html_parts.append(f"<h3>Classification Report</h3><pre>{escape(f.read())}</pre>")
        html_parts.append("</div>")

    html_parts.append("</body></html>")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))
    return report_path


def run():
    default_data_dir = str(KAGGLE_WORKING_DATA_DIR if is_kaggle_environment() else Path("data"))
    default_output_dir = str(KAGGLE_WORKING_RESULTS_DIR if is_kaggle_environment() else Path("results"))

    parser = argparse.ArgumentParser(description="Run experiments sweep")
    parser.add_argument("--data_dir", type=str, default=default_data_dir, help="Path with train_subset/test_subset folders")
    parser.add_argument("--epochs_baseline", type=int, default=8)
    parser.add_argument("--epochs_resnet", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=str, default=default_output_dir)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--max_train_per_class", type=int, default=500)
    parser.add_argument("--max_test_per_class", type=int, default=150)
    parser.add_argument("--time_budget_hours", type=float, default=4.0)
    parser.add_argument("--min_minutes_for_next_run", type=float, default=20.0)
    parser.add_argument("--architectures", type=str, default="baseline,resnet18")
    parser.add_argument("--augmentation_strengths", type=str, default="0.5,1.0")
    parser.add_argument("--dropout_rates", type=str, default="0.3")
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True, help="Save figures for each run")
    parser.add_argument("--notebook_report", action=argparse.BooleanOptionalAction, default=True, help="Write an HTML report for notebook display")
    parser.add_argument("--smoke", action="store_true", help="Run a quick smoke test (1 epoch, 1 combo)")
    args = parser.parse_args()

    architectures = parse_csv_list(args.architectures, str)
    augmentation_strengths = parse_csv_list(args.augmentation_strengths, float)
    dropout_rates = parse_csv_list(args.dropout_rates, float)

    data_path = prepare_data_dir(
        args.data_dir,
        train_count=args.max_train_per_class,
        test_count=args.max_test_per_class,
        seed=42,
    )
    results_dir = Path(args.output_dir)
    reports_dir = results_dir / "reports"
    ensure_dir(results_dir)
    ensure_dir(reports_dir)
    csv_path = results_dir / "experiments.csv"
    run_started_at = time()
    deadline = run_started_at + (args.time_budget_hours * 3600)
    min_seconds_for_next_run = args.min_minutes_for_next_run * 60
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Experiment budget: {args.time_budget_hours:.2f} hours")
    print(f"Planned combinations: {len(architectures) * len(augmentation_strengths) * len(dropout_rates)}")

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
        "report_dir",
    ]

    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    if args.smoke:
        architectures = ["baseline"]
        augmentation_strengths = [0.5]
        dropout_rates = [0.3]


    for arch in architectures:
        for aug in augmentation_strengths:
            for drop in dropout_rates:
                print(f"Running: arch={arch}, aug={aug}, drop={drop}")
                seconds_left = deadline - time()
                if not args.smoke and seconds_left < min_seconds_for_next_run:
                    print(f"Skipping remaining runs: only {seconds_left / 60:.1f} minutes left in budget.")
                    if args.notebook_report:
                        report_path = write_notebook_report(csv_path, results_dir)
                        print(f"Notebook report saved to: {report_path}")
                    print(f"Partial results saved to: {csv_path}")
                    return

                use_dummy = args.smoke or not (data_path.exists() and (data_path / "train_subset").exists() and (data_path / "test_subset").exists())
                if use_dummy:
                    from torch.utils.data import DataLoader, TensorDataset
                    x = torch.randn(16, 3, 128, 128)
                    y = torch.randint(0, 2, (16,))
                    train_loader = DataLoader(TensorDataset(x, y), batch_size=args.batch_size or 8)
                    val_loader = DataLoader(TensorDataset(x, y), batch_size=args.batch_size or 8)
                    class_names = ["fake", "real"]
                else:
                    train_loader, val_loader, class_names = get_dataloaders(
                        data_dir=str(data_path),
                        batch_size=args.batch_size,
                        image_size=args.image_size,
                        num_workers=args.num_workers,
                        augmentation_strength=aug,
                    )

                model = get_model(architecture=arch, dropout_rate=drop)

                run_name = f"{arch}_aug{aug}_drop{drop}"
                run_dir = results_dir / run_name
                report_dir = reports_dir / run_name
                ensure_dir(run_dir)
                ensure_dir(report_dir)
                checkpoint_path = run_dir / "best.pth"

                if arch == "baseline":
                    num_epochs = args.epochs_baseline
                    default_lr = 1e-3
                else:
                    num_epochs = args.epochs_resnet
                    default_lr = 3e-4
                lr = args.learning_rate if args.learning_rate is not None else default_lr

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

                history_path = run_dir / "history.json"
                try:
                    with open(history_path, "w") as hf:
                        json.dump(history, hf)
                except Exception:
                    history_path = None

                model = load_best_checkpoint(model, checkpoint_path, device)
                results = evaluate_and_report(
                    model,
                    val_loader,
                    device=device,
                    class_names=class_names,
                    scenario_name=run_name,
                    save_dir=str(report_dir),
                    plot_results=args.plots,
                )
                save_run_report(results, report_dir)
                try:
                    import matplotlib.pyplot as plt
                    plt.close("all")
                except Exception:
                    pass

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
                        str(report_dir),
                    ])
                if args.notebook_report:
                    report_path = write_notebook_report(csv_path, results_dir)
                    print(f"Notebook report saved to: {report_path}")

                print(f"Completed: arch={arch}, aug={aug}, drop={drop} | val_f1={final_f1} val_acc={final_val_acc} roc_auc={roc_auc}")

    if args.notebook_report:
        report_path = write_notebook_report(csv_path, results_dir)
        print(f"Notebook report saved to: {report_path}")
    print(f"All experiments complete. Results saved to: {csv_path}")


if __name__ == "__main__":
    run()
