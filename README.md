Project 2: Scene Classification with Transfer Learning

Deep Learning | CNN | PyTorch | ResNet-18

1. Project Overview

This project explores deep learning and transfer learning using PyTorch to classify natural and urban scenes into six categories.

A pretrained ResNet-18 convolutional neural network (CNN) is adapted to the Intel Image Classification dataset. Instead of training the model from scratch, the project uses knowledge learned from ImageNet to improve training efficiency and classification performance.

Two transfer learning strategies are evaluated:

Feature Extraction: Freeze the pretrained backbone and train only the classification head.

Fine-Tuning: Unfreeze the final residual block (layer4) and classification head to adapt the model to the target dataset.

The performance of both approaches is compared using validation accuracy, test accuracy, and error analysis.

2. Dataset Description

Dataset: Intel Image Classification
Source: Kaggle - Intel Image Classification

The dataset contains six scene categories: buildings, forest, glacier, mountain, sea, and street.

Dataset Distribution

Scene Category

Training Images

Test Images

Buildings

2,191

437

Forest

2,271

474

Glacier

2,404

553

Mountain

2,512

525

Sea

2,274

510

Street

2,382

501

Total

14,034

3,000

Training and Validation Split

The original training dataset is divided into training and validation subsets using an 85:15 ratio.

Dataset Subset

Number of Images

Training

11,929

Validation

2,105

Testing

3,000

The separate test set is reserved for final model evaluation.

3. Data Preprocessing and Augmentation

Images are preprocessed to ensure compatibility with the pretrained ResNet-18 architecture.

Processing Step

Implementation

Purpose

Resizing and cropping

224 × 224 pixels

Standardizes image dimensions

Normalization

ImageNet mean and standard deviation

Matches the pretrained model's input distribution

Random horizontal flip

Applied during training

Improves robustness to image variations

Random affine transformation

Subtle geometric transformations

Helps reduce overfitting

The ImageNet normalization parameters are:

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

These transformations help the model learn meaningful visual features while maintaining compatibility with the pretrained weights.

4. Model Architecture

The project uses ResNet-18, a residual convolutional neural network pretrained on ImageNet.

The model processes images through the following stages:

Input: Receives an RGB image of size 224 × 224 pixels.

Convolutional layers: Extract low-level features such as edges, textures, and patterns.

Residual blocks: Learn increasingly complex visual representations.

Classification head: Maps the extracted features to six output classes.

Prediction: Assigns the image to the most likely scene category.

The original fully connected layer is replaced with a new classification layer containing six output neurons, one for each scene category.

5. Transfer Learning Experiments

Phase 1: Feature Extraction

In the first phase, all pretrained convolutional layers are frozen by setting requires_grad = False. Only the newly initialized fully connected classification head is trained.

Objectives:

Preserve the pretrained ImageNet features.

Reduce computational requirements during training.

Establish a baseline for scene classification.

Results:

Metric

Result

Trainable parameters

3,078

Validation accuracy

90.8%

Feature extraction provides a strong baseline by using the visual representations learned during ImageNet pretraining.

Phase 2: Fine-Tuning

In the second phase, the final residual block (layer4) and the classification head are unfrozen. Earlier convolutional layers remain frozen.

This allows the network to adapt its high-level visual features to the target scene categories.

Objectives:

Improve task-specific feature representations.

Adapt the model to scene-specific textures and structures.

Improve classification accuracy.

Results:

Metric

Result

Trainable parameters

Approximately 8.4 million

Validation accuracy

94.1%

Fine-tuning improves the model's ability to distinguish visually similar scene categories by updating deeper feature representations.

6. Performance Comparison

The following table summarizes the reported results from both training phases.

Evaluation Metric

Feature Extraction

Fine-Tuning

Trainable parameters

3,078

Approximately 8.4 million

Validation accuracy

90.8%

94.1%

Training cost

Lower

Higher

Feature adaptation

Limited

Greater

Overall performance

Strong baseline

Higher reported accuracy

Fine-tuning improves validation accuracy by 3.3 percentage points compared with feature extraction.

Final Test Performance

Test Accuracy: 93.8%

The final model is evaluated on the separate test set containing 3,000 images.

At the reported accuracy of 93.8%, the model correctly classifies approximately 2,814 images and incorrectly classifies approximately 186 images.

The test set is intended for final evaluation rather than model selection or hyperparameter tuning.

7. Error Analysis

Although the model achieves strong overall accuracy, some scene categories share visual characteristics that can lead to incorrect predictions.

Buildings vs. Street

Urban images frequently contain both building facades and roads. A street image may be classified as a building when architectural structures dominate the image.

Glacier vs. Mountain

Glaciers and mountains can share similar textures, snow-covered surfaces, rocky formations, and landscape structures. These similarities can make accurate classification more difficult.

Key Observations

The results suggest that the model learns useful visual features for distinguishing natural and urban environments. However, classification errors can occur when different categories have overlapping visual characteristics.

Potential improvements include:

Generating a confusion matrix to identify frequent classification errors.

Examining misclassified images and prediction probabilities.

Experimenting with appropriate data augmentation techniques.

Comparing ResNet-18 with other pretrained architectures.

Evaluating precision, recall, and F1-score for each class.

8. Key Findings

Transfer learning is effective: Pretrained ImageNet features provide a strong starting point for scene classification.

Fine-tuning improves accuracy: Updating the final residual block allows the model to learn more task-specific features.

Computational cost matters: Feature extraction is less computationally demanding, while fine-tuning requires additional resources.

Visual similarities affect classification: Confusion between buildings and streets, as well as glaciers and mountains, highlights the limitations of visual scene classification.

9. Conclusion

This project demonstrates an end-to-end image classification pipeline using PyTorch and transfer learning. A pretrained ResNet-18 model is adapted to classify six categories from the Intel Image Classification dataset.

The feature extraction approach achieves a reported validation accuracy of 90.8%, while fine-tuning the final residual block improves validation accuracy to 94.1%. The final model achieves a reported test accuracy of 93.8%.

The results demonstrate that adapting pretrained models can improve task-specific performance without training an entire network from scratch. The error analysis also highlights the importance of evaluating class-wise performance and improving the model's ability to distinguish visually similar categories.

10. Technical Summary

Component

Specification

Programming language

Python

Deep learning framework

PyTorch

Model architecture

ResNet-18

Pretraining dataset

ImageNet

Target dataset

Intel Image Classification

Number of classes

6

Input image size

224 × 224 pixels

Training split

85% of the original training dataset

Validation split

15% of the original training dataset

Feature extraction accuracy

90.8% validation

Fine-tuning accuracy

94.1% validation

Final test accuracy

93.8%

Week 6 Proposal: AI-Based Manufacturing Defect Classification

Building on the transfer learning techniques explored in this project, the next phase focuses on applying deep learning to industrial automation and quality assurance.

Objective

Develop an image classification system capable of identifying manufacturing defects using transfer learning and a pretrained CNN architecture.

Dataset

Use a labeled manufacturing defect image dataset sourced from Kaggle or another suitable public repository. Before training, document the dataset source, license, class labels, and sample distribution.

Methodology

Implement a feature extraction baseline using a frozen pretrained backbone.

Fine-tune selected layers to improve defect classification.

Evaluate the model using accuracy, precision, recall, F1-score, and a normalized confusion matrix.

Compare the results of both training strategies.

Expected Application

The proposed system aims to support automated quality inspection by identifying defective products and reducing dependence on manual inspection.

Expected Outcome

A reproducible deep learning pipeline demonstrating how transfer learning can be applied to manufacturing quality control and industrial defect detection.
