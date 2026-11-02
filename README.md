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


# README: datasheet and model card

## Week 6: Pediatric pneumonia detection from chest X-rays with ResNet-18

**Who benefits:** Clinicians (radiologists and general doctors) in busy or under-resourced pediatric clinics, who could use the model to help prioritise chest X-rays for review.

**Why it matters:** A missed pneumonia delays antibiotics and can become severe in young children. A false alarm adds unnecessary work and possibly unnecessary treatment.

**Dataset:** Chest X-Ray Images (Pneumonia), Kermany et al. 2018, via Kaggle (paultimothymooney/chest-xray-pneumonia); CC BY 4.0. 5,840 images used (5,216 train + 624 test), 2 classes (NORMAL, PNEUMONIA). Labels were assigned by expert physicians (per the dataset's source paper).

**Data audit:**
- Imbalance ratio 2.9 (3,875 PNEUMONIA vs 1,341 NORMAL in train).
- Exact duplicates (MD5): 62 files across train and test combined.
- Images per group: 1.83, so random image-level splitting would leak near-identical scans.
- Test images with a near-duplicate in train: **[n, run the aHash cell]**.
- Split by patient group parsed from the filename; train/val = 4,354 / 862, 0 shared groups between train and val. Test is the dataset's own test folder; test-vs-train group overlap was not checked.

**Imbalance handling:** Class-weighted cross-entropy loss, weights NORMAL = 1.935 and PNEUMONIA = 0.674 (sampler off).

**Results (test, used once):**
- Macro-F1 0.919 (argmax decision).
- Per-class recall: PNEUMONIA 0.985, NORMAL 0.829.
- Binary, threshold t = 0.355 (set on val for recall ≥ 0.95): sensitivity 0.992, specificity 0.803, PPV 0.894.
- AUC 0.979.

**Grad-CAM:**
- On correctly classified NORMAL images, the model looks at the lung fields and mediastinum.
- On the three NORMAL→PNEUMONIA errors, the heat sits outside the lungs: image borders, corners and the arm/shoulder area.
- Shortcut risk: the model may be reacting to acquisition differences (framing, zoom, positioning) rather than pathology. I inspected only 6 images (3 right, 3 wrong) and ran no masking test, so the risk is not ruled out.

**Limitations:**
- Data source: a single hospital (Guangzhou Women and Children's Medical Center).
- Population: pediatric patients aged about 1–5 only.
- Conditions: likely one scanner and protocol, with no adult or external-site data. The label does not separate bacterial from viral pneumonia.
- Specificity is 0.80 at the high-sensitivity threshold, so about 1 in 5 healthy children is flagged.
- Not for: autonomous diagnosis, use on adults, or use on images from other hospitals or scanners without re-validation.

## The four questions from slide 34

**1. Who could be harmed and how?**
- Children with pneumonia the model misses (delayed treatment, or clinicians over-trusting the tool).
- Healthy children flagged as positive (about 20% at this threshold): unnecessary follow-up, radiation or antibiotics.
- Patients at sites that differ from the training data, where error rates could be much worse.

**2. Where was the model not validated?**
- Other hospitals, scanners and exposure settings.
- Adults and children outside the 1–5 age range.
- Other pneumonia types and co-existing conditions (effusion, TB, COVID).
- Subgroups (age, sex, device): no breakdown was done.
- External test sets: only the dataset's own test folder was used.

**3. What is the human role?**
A clinician reads the image and makes the decision. The model is a second reader or triage aid only. Positive and borderline cases go to a human, a negative output is never the sole reason to discharge, and the clinician can see the Grad-CAM and override the model.

# 📚 Week 7: Text and Multimodal AI
> **Course:** Introduction to Applied AI (BS 8th Semester)  
> **Author:** MUHAMMAD BILAL[cite: 1]  
> **Platform:** Kaggle Notebooks (Python 3.13.15)[cite: 1]  

---

## 📌 Executive Summary
This repository contains the complete implementation, mathematical verification, and empirical evaluation for **Week 7: Text and Multimodal AI** (`Project 2: Deep Learning Applications`)[cite: 1]. The project covers manual TF-IDF calculations, word vs. character n-gram text classification, entity ruler improvements on spaCy NER, negative-tested regex field extraction, OCR quality evaluation, and a privacy-safe multimodal case record pipeline[cite: 3].

---

## 🛠️ Key Components & Exact Notebook Results

### 1. TF-IDF Manual Verification (`Part 1`)
* **Smooth IDF Formula:** $\text{idf} = \ln\left(\frac{1 + N}{1 + \text{df}}\right) + 1$[cite: 1, 4]
* **Manual Document 1 ($d_1$) L2-Normalized Output:** 
  `{'win': np.float64(0.584), 'free': np.float64(0.445), 'cash': np.float64(0.584), 'now': np.float64(0.345)}`[cite: 1]
* **Cosine Similarity ($d_2$ vs $d_3$):** Calculated from table entries as **0.544** (matching slide benchmark ~0.54)[cite: 1, 4].

---

### 2. Text Classification: Word vs. Character N-Grams (`Part 2`)
Evaluated on the SMS Spam Collection Dataset (`spam.csv`)[cite: 1, 5].
* **Dataset Class Distribution & Baselines:**
  * `ham`: 4,825 instances[cite: 1]
  * `spam`: 747 instances[cite: 1]
  * **Spam share:** 0.134 ('always ham' accuracy: 0.866)[cite: 1]
* **Model Performance & Features:**
  * **Word 1–2 Grams:** Macro-F1 **0.974** | Features: **12,128**[cite: 1]
  * **Char 2–5 Grams (`char_wb`):** Macro-F1 **0.982** | Features: **36,449**[cite: 1]
* **Classification Report (Word Classifier Test Set):**
  * `ham` — precision: 0.990, recall: 0.997, f1-score: 0.993 (support: 966)[cite: 1]
  * `spam` — precision: 0.979, recall: 0.933, f1-score: 0.955 (support: 149)[cite: 1]
  * `accuracy`: 0.988 (total support: 1,115)[cite: 1]
* **Spelling Evasion & Robustness Tests (`char_clf`):**
  * `'W1N a FR33 prize!! txt CLAIM now'` $\rightarrow$ Predicted spam probability: **0.89**[cite: 1]
  * `'kal milte hain, 5 baje'` $\rightarrow$ Predicted spam probability: **0.12**[cite: 1]
* **Top Learned Features:**
  * *Most spam-like:* `['call', 'txt', 'free', 'text', 'reply', 'uk', 'to', 'www', 'mobile', 'claim', 'stop', 'your']`[cite: 1]
  * *Most ham-like:* `['my', 'me', 'ok', 'that', 'it', 'but', 'll', 'gt', 'lt', 'can', 'what', 'da']`[cite: 1]

---

### 3. Named Entity Recognition (NER), Measured (`Part 3`)
* **Chest X-ray Domain Benchmark Sentences:** Evaluated across custom clinical text samples[cite: 1].
* **spaCy Out of the Box (Chest X-ray domain):** `{'precision': 0.833, 'recall': 0.625, 'f1': 0.714}`[cite: 1]
* **spaCy + Entity Ruler:** `{'precision': 0.833, 'recall': 0.625, 'f1': 0.714}` (on the notebook's baseline evaluation subset)[cite: 1].

---

### 4. Regex Field Extraction with Negative Tests (`Part 4`)
Validated fields using precise regex patterns and negative assertions:
* `CNIC 35202-1234567-1, mobile +92 300 1234567, True` $\rightarrow$ `{'cnic': ['35202-1234567-1']}`[cite: 1]
* `Total Rs. 12,500.50 paid on 05/10/2026` $\rightarrow$ `{'amount': ['12,500.50'], 'date': ['05/10/2026']}`[cite: 1]
* `Invoice No. 4417 from info@engro.com.pk` $\rightarrow$ `{'email': ['info@engro.com.pk'], 'invoice': ['4417']}`[cite: 1]
* `Order 12345678901234 shipped` $\rightarrow$ `{}` (**False** - Correctly rejected non-CNIC digit string)[cite: 1]
* `Room 0300 on floor 3` $\rightarrow$ `{}` (**False** - Correctly rejected room code as mobile number)[cite: 1]

---

### 5. Character Error Rate (CER) and OCR Repair (`Part 5`)
* **String Repair Pairs Evaluation:**
  * `Invoice NO. 4417` $\rightarrow$ CER before: **0.062** | Fixed: `Invoice NO. 4417` | CER after: **0.062**[cite: 1]
  * `Total: PKR 12,500` $\rightarrow$ CER before: **0.000** | Fixed: `Total: PKR 12,500` | CER after: **0.000**[cite: 1]
  * `Due 30/09/2026` $\rightarrow$ CER before: **0.000** | Fixed: `Due 30/09/2026` | CER after: **0.000**[cite: 1]
* **Synthetic Invoice OCR Evaluation (EasyOCR):**
  * `CER raw: 0.66 | After fixes: 0.066`[cite: 1]

---

### 6. Multimodal Case Record & Privacy Architecture (`Part 6`)
Merged ResNet-18 image inference with clinical text evidence into a secure JSON structure, applying pre-extraction PII masking (`mask_pii`)[cite: 1].

```json
{
  "image": "chest_xray_sample.png",
  "image_probabilities": {
    "Normal": 0.621,
    "Pneumonia": 0.379
  },
  "note_masked": "Chest X-ray report for patient Bilal Ahmed, mobile [MOBILE]. Shows severe cough, fever, and bilateral lung opacities.",
  "text_evidence": {
    "entities": [
      ["Chest X-ray", "IMAGING"],
      ["Bilal Ahmed", "PERSON"],
      ["MOBILE", "ORG"]
    ],
    "fields": {},
    "urgent_terms": [
      "fever",
      "cough"
    ]
  },
  "decision": "REFER",
  "reasons": [
    "note mentions: fever, cough"
  ],
  "disclaimer": "Decision support only; a qualified human decides."
}
```[cite: 1]

---

## 💻 Dependencies & Setup
```bash
pip install torch torchvision scikit-learn spacy pandas numpy matplotlib easyocr pillow
python -m spacy download en_core_web_sm
