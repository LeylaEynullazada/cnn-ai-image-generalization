# Project Report

## Goal

The project trains convolutional neural networks to classify images as real or AI-generated. The completed experiments focus on whether a larger CNN architecture and stronger dropout improve performance on the dataset's held-out test split.

The original proposal targeted leave-one-generator-out evaluation. The available training runs reported here do not complete that protocol because the data used in the final Kaggle run was organized only by `fake` and `real` folders. The results are therefore reported as a standard train/test architecture and regularization study.

## Experimental Setup

- Dataset: Kaggle AI-Generated Images vs Real Images.
- Classes: `fake` and `real`.
- Image size: 128x128.
- Optimizer: Adam.
- Loss: cross-entropy.
- Early stopping: validation macro-F1.
- Evaluation metrics: accuracy, macro precision, macro recall, macro-F1, ROC-AUC, confusion matrix, ROC curve.

The main comparisons were:

1. Baseline CNN with dropout 0.3.
2. ResNet-18 with dropout 0.3.
3. ResNet-18 with dropout 0.5.

## Results

| Model | Augmentation | Dropout | Accuracy | Precision | Recall | F1 | ROC-AUC | Train time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline CNN | 0.5 | 0.3 | 0.7102 | 0.7137 | 0.7102 | 0.7090 | 0.7910 | interrupted after training |
| ResNet-18 | 0.5 | 0.3 | 0.7319 | 0.7631 | 0.7319 | 0.7237 | 0.8436 | 3128.76 |
| ResNet-18 | 0.5 | 0.5 | 0.7478 | 0.7698 | 0.7478 | 0.7425 | 0.8419 | 6266.12 |

## Analysis

ResNet-18 performed better than the smaller baseline CNN. Its macro-F1 improved from 0.7090 to 0.7237 with the same augmentation strength and dropout 0.3. ROC-AUC also improved from 0.7910 to 0.8436, which suggests the ResNet model separated the classes better overall.

Increasing dropout in ResNet-18 from 0.3 to 0.5 improved accuracy from 0.7319 to 0.7478 and macro-F1 from 0.7237 to 0.7425. ROC-AUC changed only slightly, from 0.8436 to 0.8419. This means dropout 0.5 improved the final class predictions but did not meaningfully improve the ranking of fake versus real probabilities.

The confusion matrix for ResNet-18 with dropout 0.5 showed stronger recall for fake images than real images. The model classified many real images as fake, so its behavior is not balanced even when accuracy is near 75 percent. Macro-F1 is therefore more informative than accuracy alone.

## What Worked

- The PyTorch pipeline ran end-to-end on Kaggle GPU.
- The modular code produced checkpoints, histories, CSV metrics, confusion matrices, and ROC curves.
- ResNet-18 improved over the baseline CNN.
- Stronger dropout gave the best macro-F1 among completed runs.

## What Did Not Work

- The full leave-one-generator-out experiment was not completed.
- Large input images slowed training and produced PIL decompression warnings.
- The notebook environment had a pandas import issue, so some final reporting was done with plain Python instead of pandas.
- One baseline run was interrupted during final evaluation, but the best checkpoint was saved and evaluated separately.

## Conclusion

The completed experiments show that ResNet-18 is a better detector than the smaller baseline CNN on the standard train/test split. Dropout 0.5 gave the best macro-F1, so stronger regularization helped in this setting. The results do not yet prove generalization to unseen generators; they mainly support the architecture and regularization comparison.

The next step would be to construct generator-aware splits and repeat the same evaluation with Stable Diffusion, Midjourney, and DALL-E held out one at a time.
