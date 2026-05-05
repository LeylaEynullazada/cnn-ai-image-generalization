# CNN AI Image Generalization
This project studies binary classification of real versus AI-generated images using convolutional neural networks. The original motivation was to test whether a detector learns general AI-image cues or mostly learns patterns tied to the data distribution. The implemented experiments compare a small custom CNN with ResNet-18 and test whether stronger dropout improves validation performance.


The code is written as reusable PyTorch modules for data loading, model definition, training, and evaluation. The main experiment runner saves checkpoints, histories, metrics, confusion matrices, ROC curves, sample predictions, and a notebook-friendly HTML report.

## Team Responsibilities

Replace the ID field before submission.

| Team member | ID | Contributions |
|---|---:|---|
| Raul Agayev | TODO | Dataset setup, PyTorch data pipeline, baseline CNN and ResNet-18 experiments, evaluation plots, result analysis, repository organization |

## Research Question

Can a CNN trained to distinguish real images from AI-generated images learn useful visual cues, and how do model architecture and dropout affect performance? The main comparison in the completed experiments is:

- baseline CNN vs. ResNet-18
- ResNet-18 with dropout 0.3 vs. dropout 0.5

The proposal also described leave-one-generator-out testing across Stable Diffusion, Midjourney, and DALL-E. The final runs reported here use the public dataset's available `train` and `test` folders with `fake` and `real` labels. Generator-level labels were not available in the folder structure used for these runs, so leave-one-generator-out remains a limitation rather than a completed result.

## Dataset

Dataset: [AI-Generated Images vs Real Images](https://www.kaggle.com/datasets/tristanzhang32/ai-generated-images-vs-real-images) by Tristan Zhang.

The dataset contains real and AI-generated images arranged in `train/fake`, `train/real`, `test/fake`, and `test/real` folders. Images are resized to 128x128 and normalized with ImageNet statistics. Training uses random resized crop, horizontal flip, color jitter, rotation, and occasional Gaussian blur.

## Repository Layout

```text
src/dataset.py          Data transforms and dataloaders
src/model.py            Baseline CNN and ResNet-18 model definitions
src/train.py            Training loop, validation, early stopping
src/evaluate.py         Metrics and plots
scripts/run_experiments.py
                        Experiment runner
main.ipynb              Grading notebook
REPORT.md               Results and interpretation
validate_*.py           Lightweight checks
```

## Setup

```bash
pip install -r requirements.txt
```

The project uses PyTorch, torchvision, NumPy, scikit-learn, matplotlib, and tqdm.

## Data Layout

The experiment runner expects:

```text
data/
  train_subset/
    fake/
    real/
  test_subset/
    fake/
    real/
```

On Kaggle, avoid copying the full dataset into `/kaggle/working`. Use symlinks:

```bash
mkdir -p /kaggle/working/data
ln -s /kaggle/input/datasets/tristanzhang32/ai-generated-images-vs-real-images/train /kaggle/working/data/train_subset
ln -s /kaggle/input/datasets/tristanzhang32/ai-generated-images-vs-real-images/test /kaggle/working/data/test_subset
```

## Running Experiments

Baseline CNN:

```bash
python scripts/run_experiments.py \
  --data_dir /kaggle/working/data \
  --time_budget_hours 2 \
  --batch_size 64 \
  --num_workers 2 \
  --architectures baseline \
  --augmentation_strengths 0.5 \
  --dropout_rates 0.3 \
  --epochs_baseline 2 \
  --patience 1
```

ResNet-18 with dropout 0.3:

```bash
python scripts/run_experiments.py \
  --data_dir /kaggle/working/data \
  --time_budget_hours 3 \
  --batch_size 64 \
  --num_workers 2 \
  --architectures resnet18 \
  --augmentation_strengths 0.5 \
  --dropout_rates 0.3 \
  --epochs_resnet 1 \
  --patience 1
```

ResNet-18 with dropout 0.5:

```bash
python scripts/run_experiments.py \
  --data_dir /kaggle/working/data \
  --time_budget_hours 2 \
  --batch_size 64 \
  --num_workers 2 \
  --architectures resnet18 \
  --augmentation_strengths 0.5 \
  --dropout_rates 0.5 \
  --epochs_resnet 3 \
  --patience 1
```

Outputs are written to:

```text
results/experiments.csv
results/reports/<run_name>/confusion_matrix.png
results/reports/<run_name>/roc_curve.png
results/reports/<run_name>/sample_predictions.png
results/reports/<run_name>/classification_report.txt
```

## Main Results

| Model | Augmentation | Dropout | Accuracy | Precision | Recall | F1 | ROC-AUC | Train time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline CNN | 0.5 | 0.3 | 0.7102 | 0.7137 | 0.7102 | 0.7090 | 0.7910 | interrupted after training |
| ResNet-18 | 0.5 | 0.3 | 0.7319 | 0.7631 | 0.7319 | 0.7237 | 0.8436 | 3128.76 |
| ResNet-18 | 0.5 | 0.5 | 0.7478 | 0.7698 | 0.7478 | 0.7425 | 0.8419 | 6266.12 |

ResNet-18 outperformed the baseline CNN. Increasing dropout from 0.3 to 0.5 improved ResNet-18 accuracy and macro-F1, although ROC-AUC stayed almost the same. This suggests that stronger dropout helped the final decision threshold and class balance slightly, but did not substantially improve ranking quality.

## Interpretation

The baseline CNN reached macro-F1 0.7090. ResNet-18 with dropout 0.3 improved macro-F1 to 0.7237 and ROC-AUC to 0.8436. ResNet-18 with dropout 0.5 produced the best macro-F1 at 0.7425 and the best accuracy at 0.7478.

The confusion matrices showed that the ResNet models were better at identifying fake images than real images. For the dropout 0.5 run, fake recall was 0.8907, while real recall was 0.6048. This means the model often labeled real images as fake. The result is useful, but it also shows that accuracy alone is not enough to describe detector behavior.

## Limitations

- The completed runs use the dataset's standard train/test split, not leave-one-generator-out testing, otherwise it would take too much time to train.
- Some images are very large, which slowed image decoding during training and evaluation.
- The baseline training was interrupted by kaggle/colab after early stopping, so its checkpoint was evaluated separately and the training time was not recorded in the final table.
- The results should be interpreted as an architecture and regularization comparison, not as proof of robustness to unseen generators.

## Validation

Quick checks:

```bash
python validate_dataset.py
python validate_architecture.py
python validate_evaluation.py
```

Smoke run:

```bash
python scripts/run_experiments.py --smoke
```

To create a sample-prediction figure from an existing checkpoint:

```bash
python scripts/show_predictions.py \
  --data_dir /kaggle/working/data \
  --architecture resnet18 \
  --dropout_rate 0.5 \
  --checkpoint /kaggle/working/results/resnet18_aug0.5_drop0.5/best.pth \
  --output_path /kaggle/working/results/reports/resnet18_aug0.5_drop0.5/sample_predictions.png
```
`checkpoints/` and `results/`, which are ignored by git.