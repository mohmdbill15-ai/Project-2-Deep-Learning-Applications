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




## Week 6: Pediatric pneumonia detection from chest X-rays with ResNet-18

**Who benefits:** Clinicians (radiologists and general doctors) in busy or under-resourced pediatric clinics, who could use the model to help prioritise chest X-rays for review.

**Why it matters:** A missed pneumonia delays antibiotics and can become severe in young children. A false alarm adds unnecessary work and possibly unnecessary treatment.

**Dataset:** Chest X-Ray Images (Pneumonia), Kermany et al. 2018, via Kaggle (paultimothymooney/chest-xray-pneumonia); CC BY 4.0. I used 5,216 train + 624 test = 5,840 images, 2 classes (NORMAL, PNEUMONIA). Labels were assigned by expert physicians (per the dataset's source paper).

**Data audit:**
- Imbalance ratio 2.9 (train: 3,875 PNEUMONIA vs 1,341 NORMAL).
- Exact duplicates (MD5): 62 files across the train and test folders combined.
- Test images with a near-duplicate in train: **not computed** (see the code cell below).
- Split by patient ID parsed from the filename (`GROUP_PATTERN`); train/val is 4,354 / 862. Groups shared by train and val: 0. Test is the dataset's own test folder, and I did not check its overlap with train groups.

**Imbalance handling:** Class-weighted cross-entropy loss with weights NORMAL = 1.935 and PNEUMONIA = 0.674 (sampler off).

**Results (test, used once):**
- Macro-F1 0.919 at the default argmax decision.
- Per-class recall: PNEUMONIA 0.985, NORMAL 0.829.
- Binary operating point, threshold t = 0.355 (set on val for recall ≥ 0.95): sensitivity 0.992, specificity 0.803, PPV 0.894.
- AUC 0.979.

**Grad-CAM:**
- On correctly classified NORMAL images, the model looks at the lung fields and the mediastinum, which is plausible.
- On the three NORMAL→PNEUMONIA errors, the heat sits outside the lungs: the upper image border, the arm and shoulder area, and the image corners.
- Shortcut risk: this suggests the model may react to acquisition differences (framing, zoom, positioning, edge artifacts) rather than pathology. I only inspected 6 images (3 right, 3 wrong) and did not run a masking or ablation test, so the risk is not ruled out.

**Limitations:**
- Data source: a single hospital (Guangzhou Women and Children's Medical Center).
- Population: pediatric patients aged about 1–5 only.
- Conditions: one scanner and protocol, with no adult, other-site or other-device data. The binary label does not separate bacterial from viral pneumonia, and only the labels from the original paper are available.
- Test specificity is only 0.80 at the high-sensitivity threshold, so about 1 in 5 healthy children is flagged.
- Not for: autonomous diagnosis, use on adults, or use on images from other hospitals or scanners without re-validation.

## The four questions from slide 34

**1. Who could be harmed and how?**
- Children with pneumonia that the model misses: sensitivity was 0.992 on test, but the few false negatives mean delayed treatment, and clinicians who over-trust the tool may stop looking.
- Healthy children who are false positives (about 20% at this threshold): unnecessary follow-up, radiation exposure or antibiotics.
- Patients at sites that differ from the training data, where the error rates could be much worse.

**2. Where was the model not validated?**
- Other hospitals, scanners and exposure settings.
- Adults and children outside the 1–5 age range.
- Other pneumonia types and co-existing conditions (effusion, TB, COVID).
- Subgroups: no per-age, per-sex or per-device breakdown was done.
- External test sets: only the dataset's own test folder was used.

**3. What is the human role?**
A clinician reads the image and makes the decision. The model is a second-reader or triage aid only. Positive and borderline cases always go to a human, and a negative output must never be the sole reason to discharge a patient. A clinician should be able to see the Grad-CAM and override the model.

**4. What evidence would change my conclusions?**
- A near-duplicate check showing many test images with a near-twin in train would mean the scores are inflated.
- Grad-CAM or masking tests showing that performance depends on borders, markers or framing rather than lung tissue.
- A drop in sensitivity or specificity on an external hospital's data.
- Per-subgroup results showing much worse performance for a particular age or device.
- A reader study showing no benefit over clinicians alone.
