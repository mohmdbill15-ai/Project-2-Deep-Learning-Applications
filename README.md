# Project 2: Scene Classification with Transfer Learning

**Deep Learning | CNN | PyTorch | ResNet-18**

---

## 1. Project Overview

This project explores deep learning and transfer learning using PyTorch to classify natural and urban scenes into six categories.

A pretrained ResNet-18 convolutional neural network (CNN) is adapted to the Intel Image Classification dataset.

Two transfer learning strategies are evaluated:

- **Feature Extraction:** Freeze the pretrained backbone and train only the classification head.
- **Fine-Tuning:** Unfreeze the final residual block (`layer4`) and classification head to adapt the model to the target dataset.

## 2. Dataset Description

**Dataset:** Intel Image Classification  
**Source:** [Kaggle - Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)

### Dataset Distribution

| Scene Category | Training Images | Test Images |
|---|---:|---:|
| Buildings | 2,191 | 437 |
| Forest | 2,271 | 474 |
| Glacier | 2,404 | 553 |
| Mountain | 2,512 | 525 |
| Sea | 2,274 | 510 |
| Street | 2,382 | 501 |
| **Total** | **14,034** | **3,000** |

### Training and Validation Split

The original training dataset is divided using an 85:15 ratio.

| Dataset Subset | Number of Images |
|---|---:|
| Training | 11,929 |
| Validation | 2,105 |
| Testing | 3,000 |

The separate test set is reserved for final model evaluation.

## 3. Data Preprocessing and Augmentation

Images are preprocessed to ensure compatibility with ResNet-18.

| Processing Step | Implementation | Purpose |
|---|---|---|
| Resizing and cropping | 224 × 224 pixels | Standardizes image dimensions |
| Normalization | ImageNet mean and standard deviation | Matches the pretrained model's input distribution |
| Random horizontal flip | Applied during training | Improves robustness |
| Random affine transformation | Subtle geometric transformations | Helps reduce overfitting |

### ImageNet Normalization Parameters

```python
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```

## 4. Model Architecture

The project uses **ResNet-18**, a residual convolutional neural network pretrained on ImageNet.

### Model Pipeline

1. **Input:** RGB image of size 224 × 224 pixels.
2. **Convolutional layers:** Extract edges, textures, and patterns.
3. **Residual blocks:** Learn increasingly complex visual features.
4. **Classification head:** Maps extracted features to six output classes.
5. **Prediction:** Assigns the image to the most likely scene category.

The original fully connected layer is replaced with a new layer containing six output neurons.

## 5. Transfer Learning Experiments

### Phase 1: Feature Extraction

All pretrained convolutional layers are frozen using `requires_grad = False`. Only the newly
