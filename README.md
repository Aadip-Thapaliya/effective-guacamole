# Microelectronics Defect Classification — Valeo Machine Vision Challenge

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.x-red.svg)](https://keras.io/)
[![Status](https://img.shields.io/badge/status-active-brightgreen.svg)]()
[![OpenCampus](https://img.shields.io/badge/course-ML%20with%20TensorFlow-blueviolet)]()

> **A deep learning pipeline for automated multi-class defect detection in microelectronic components using machine vision imagery from the Valeo ChallengeData competition.**

**Repository:** [https://github.com/Sl4artiB4rtF4rst/MachineVision_Valeo_ChallangeData](https://github.com/Sl4artiB4rtF4rst/MachineVision_Valeo_ChallangeData)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Background and Motivation](#2-background-and-motivation)
3. [Dataset Description](#3-dataset-description)
   - 3.1 [Dataset Source](#31-dataset-source)
   - 3.2 [Dataset Structure](#32-dataset-structure)
   - 3.3 [Feature Description](#33-feature-description)
   - 3.4 [Label Classes](#34-label-classes)
   - 3.5 [Class Distribution and Imbalance](#35-class-distribution-and-imbalance)
   - 3.6 [Image Properties](#36-image-properties)
4. [Exploratory Data Analysis](#4-exploratory-data-analysis)
5. [Literature Review](#5-literature-review)
6. [Methodology](#6-methodology)
   - 6.1 [Data Preprocessing and Augmentation](#61-data-preprocessing-and-augmentation)
   - 6.2 [Dataset Balancing Strategy](#62-dataset-balancing-strategy)
   - 6.3 [Data Splits](#63-data-splits)
   - 6.4 [Model Architectures](#64-model-architectures)
   - 6.5 [Training Configuration](#65-training-configuration)
   - 6.6 [Evaluation Metrics and Rationale](#66-evaluation-metrics-and-rationale)
7. [Baseline Model](#7-baseline-model)
8. [Model Definition and Evaluation](#8-model-definition-and-evaluation)
   - 8.1 [Model 1 — Shallow CNN](#81-model-1--shallow-cnn)
   - 8.2 [Model 2 — Two-Layer CNN](#82-model-2--two-layer-cnn)
   - 8.3 [Model 3 — Three-Layer CNN (Best)](#83-model-3--three-layer-cnn-best)
   - 8.4 [InceptionV3 — Transfer Learning (Feature Extraction)](#84-inceptionv3--transfer-learning-feature-extraction)
9. [Results Summary](#9-results-summary)
   - 9.1 [Model Comparison Table](#91-model-comparison-table)
   - 9.2 [Best Model Classification Report](#92-best-model-classification-report)
   - 9.3 [Key Findings and Insights](#93-key-findings-and-insights)
10. [Repository Structure](#10-repository-structure)
11. [Getting Started](#11-getting-started)
    - 11.1 [Prerequisites](#111-prerequisites)
    - 11.2 [Installation](#112-installation)
    - 11.3 [Data Setup](#113-data-setup)
    - 11.4 [Running the Pipeline](#114-running-the-pipeline)
12. [Hyperparameter Reference](#12-hyperparameter-reference)
13. [Known Limitations](#13-known-limitations)
14. [Next Steps and Future Work](#14-next-steps-and-future-work)
15. [Contributors](#15-contributors)
16. [License](#16-license)
17. [Cover Image](#17-cover-image)

---

## 1. Project Overview

This project was developed as a group assignment for the **OpenCampus course "Machine Learning with TensorFlow"** (2024/2025). The objective is to build and evaluate a convolutional neural network (CNN) classification pipeline capable of identifying fabrication defects in microelectronic components from machine vision imagery.

The dataset is sourced from the **Valeo ChallengeData competition** (Challenge #157), hosted by [challengedata.ens.fr](https://challengedata.ens.fr/participants/challenges/157/). It consists of approximately 8,278 labeled grayscale images of microelectronic conductor tracks and contacts, where each image is assigned a label describing the fabrication or structural status of the depicted feature.

The task is a **multi-class image classification** problem with 6 primary training labels and an additional out-of-distribution class (*Drift*) present only in the test data. Correct identification of the *Drift* class is penalized more strongly in competition scoring, making it a special challenge requiring specific handling beyond standard supervised classification.

---

## 2. Background and Motivation

Automated visual inspection of microelectronics is a critical quality-control step in semiconductor and electronics manufacturing. Manual inspection is slow, expensive, and prone to human error — particularly for microscale structures that require specialist expertise to evaluate. Machine vision systems powered by deep learning offer the potential for high-throughput, consistent, and automated classification of fabrication defects.

This project explores the feasibility of training CNN-based classifiers on a real-world industrial dataset with the following challenges:

- **Severe class imbalance** — the dominant `Missing` class accounts for approximately 79% of all labeled images
- **Visually similar classes** — some defect types (e.g., `Lift-off blanc` vs. `Lift-off noir`, or `Short circuit MOS` vs. `GOOD`) are nearly indistinguishable by visual inspection alone
- **Variable image resolution** — images range from approximately 530×530 to over 1,260×1,260 pixels
- **Unknown imaging modality** — the images appear to come from either optical microscopy or scanning electron microscopy (SEM); no scale bar is provided
- **Out-of-distribution test class** — the *Drift* class represents structurally different data not present in the training labels, requiring detection of unknown-class inputs

---

## 3. Dataset Description

### 3.1 Dataset Source

| Property | Value |
|---|---|
| Platform | [ChallengeData ENS](https://challengedata.ens.fr/participants/challenges/157/) |
| Dataset Owner | Valeo (French-Chinese electronics company) |
| Challenge ID | #157 |
| License | As per ChallengeData platform terms |

### 3.2 Dataset Structure

```
<base_file_path>/
├── input_train/
│   └── input_train/       # 8,278 training images (grayscale, variable resolution)
├── input_test/
│   └── input_test/        # 1,055 test images
├── Y_train_eVW9jym.csv    # Training labels CSV
└── X_test_*.csv           # Test file manifest
```

The label CSV contains 5 columns: an unnamed index, `filename`, `window`, `lib`, and `Label`.

```
   Unnamed: 0   filename              window  lib     Label
0  0            15b3bab7c186...       2003    Die01   Missing
1  1            1856617e1ac2...       2003    Die01   GOOD
2  2            19066cce773b...       2003    Die01   Missing
```

**Total training samples:** 8,278
**Total test samples:** 1,055

### 3.3 Feature Description

| Feature | Type | Values | Description |
|---|---|---|---|
| `filename` | String | Hash-based filename | Image file identifier mapping to the image directory |
| `window` | Integer | `2003`, `2005` | Year of manufacture or inspection; likely correlates to production batch |
| `lib` | String | `Die01`, `Die02`, `Die03`, `Die04` | Identifies the die type; different die types are visually distinguishable from each other |

The `window` and `lib` features provide supplementary context that may aid classification, particularly in disambiguating visually similar defect classes that are correlated with specific die types or production years.

### 3.4 Label Classes

The label represents the fabrication status of the microelectronic feature depicted — specifically, whether a bridge-like conductive structure is intact, deformed, or missing.

| Label ID | Label Name (French) | Label Name (English) | Description |
|---|---|---|---|
| 0 | `GOOD` | Good | Fully functioning structure with all conductive film layers and bridge intact |
| 1 | `Boucle plate` | Flat Loop | Bridge-like feature is present but displays an abnormal, flattened surface structure; may correlate with `Die` or `Window` features |
| 2 | `Lift-off blanc` | White Lift-off | A layer appears missing from the conductive track (indicated by different brightness/texture); bridge is present; some suspected mislabeled samples |
| 3 | `Lift-off noir` | Black Lift-off | Visually almost identical to White Lift-off on visual inspection; uncertain distinguishability by image data alone |
| 4 | `Missing` | Missing | Bridge-like structure or contacts entirely absent from the depicted area |
| 5 | `Short circuit MOS` | Short Circuit MOS | No visible difference from `GOOD` on visual inspection; limited signal for image-based classification |
| 6 *(test only)* | `Drift` | Drift / Unknown | Data that does not belong to any of the above categories; classified separately; incorrectly labeling this class incurs an additional scoring penalty |

> **Note on label quality:** Visual inspection of the training data reveals suspected mislabeled samples in both `Lift-off blanc` and `Lift-off noir` classes. Some images that appear to be `GOOD` have been labeled as `Lift-off blanc`. This is a known data quality issue observed during EDA.

### 3.5 Class Distribution and Imbalance

The label distribution is highly skewed:

| Class | Approximate Count | Share of Dataset |
|---|---|---|
| Missing | ~6,500 | ~79% |
| GOOD | ~1,200 | ~14% |
| White Lift-off (Lift-off blanc) | ~280 | ~3% |
| Short Circuit MOS | ~85 | ~1% |
| Flat Loop (Boucle plate) | ~75 | ~1% |
| Black Lift-off (Lift-off noir) | ~65 | ~1% |

The extreme dominance of the `Missing` class poses significant risks of biased models. A naive classifier that always predicts `Missing` would achieve approximately 79% raw accuracy while completely failing at defect categorization. Dataset balancing or class-weighted loss are therefore essential components of any viable training strategy.

### 3.6 Image Properties

- **Color mode:** Grayscale (8-bit, single channel)
- **Resolution range:** 530×530 px (min) to 1,470×1,260 px (max)
- **Mean resolution:** approximately 737 × 631 px
- **Resolution clusters:** Images are clustered around 5 discrete resolution levels, suggesting images were captured at fixed magnification settings
- **Data integrity:** Zero corrupted files, zero zero-byte files, zero duplicate pairs (verified programmatically)
- **File naming:** Hash-based filenames with no embedded metadata

---

## 4. Exploratory Data Analysis

A full exploratory data analysis is available in `1_DatasetCharacteristics/exploratory_data_analysis.ipynb`. The following summarizes the key findings.

**Image Quality and Integrity**

All 8,278 training images and 1,055 test images loaded successfully as valid PIL images. No missing values were found in the label CSV, and every filename in the label table maps to a corresponding image file on disk. No zero-byte or corrupted files were detected.

**Resolution Distribution**

Image resolutions are not uniformly distributed. Five distinct resolution clusters are observed, consistent with fixed magnification levels in an industrial inspection setup. The lowest resolution is approximately 530×530 px and the highest exceeds 1,200×1,200 px. All resolutions are considered sufficient for the classification task; in practice, images are downsampled to 128×128 or 299×299 px for model input, significantly below the native resolution.

**Visual Class Separability**

Visual inspection of representative images confirms that `GOOD` and structurally defective classes (`Missing`, etc.) are readily distinguishable by human inspection. The `Missing` class in particular is highly visually distinct. However, the following pairs present significant visual overlap:

- `Lift-off blanc` vs. `Lift-off noir` — nearly identical appearance
- `Short circuit MOS` vs. `GOOD` — no discernible visual difference
- Some `Flat loop` samples appear visually similar to `GOOD`

**Feature Correlation**

Preliminary analysis of the `lib` (Die type) and `window` (year) features suggests they may carry discriminative signal for certain label classes. For example, `Flat loop` behavior appears correlated with specific `Die` types. Further analysis of conditional label distributions per `Die` and `window` value is warranted.

**Frequency-Domain and Edge Analysis**

FFT-based texture analysis and edge sharpness metrics (Sobel + Canny) were computed as part of a strategic EDA pass. Entropy analysis confirms that images across classes have sufficient information density for a neural network to learn discriminative features.

---

## 5. Literature Review

The project draws on the following works, available in full in `0_LiteratureReview/`:

### Source 1: A Comprehensive Review of CNNs for Defect Detection in Industrial Applications

**Objective:** Review of CNN-based defect detection methods across industrial applications.
**Methods:** Survey of CNNs, object detection architectures, and image classification models. Heavy emphasis on transfer learning from standard pretrained models (ImageNet-pretrained VGG, ResNet, Inception, EfficientNet).
**Outcomes:** Provides a map of common approaches and best practices in machine vision for industrial quality control. Documents the dominance of transfer learning approaches and their typical performance ranges.
**Relevance:** Guided the selection of pretrained architectures for transfer learning experiments and provided context for interpreting our results relative to the broader field.

### Source 2: Image-Based Defect Detection in Lithium-Ion Battery Electrodes Using CNNs

**Objective:** Defect detection in cross-sectional images of Li-ion battery electrodes.
**Methods:** CNN with and without transfer learning on a dataset of approximately 3,200 images (400 defective).
**Outcomes:** Fine-tuned pretrained models achieve F1 approximately 0.99; models trained from scratch achieve up to F1 approximately 0.85. Dataset size was found sufficient for both approaches.
**Relevance:** Closely mirrors our task — similar dataset size, similar defect detection framing, and confirms the viability of transfer learning on small industrial datasets.

### Source 3: Neural Network for Nanoscience Scanning Electron Microscope Image Recognition

**Objective:** Classification of SEM images into different nanomaterial categories.
**Methods:** CNNs and transfer learning on datasets of a few hundred to a few thousand images per category (10 classes total).
**Outcomes:** 85–95% accuracy depending on class; Inception-v3 achieved the best accuracy with the fastest training time.
**Relevance:** Directly comparable to our imaging setup (SEM-style images), dataset scale, and classification objective. Results in this paper set the expectation range for what is achievable with our data.

---

## 6. Methodology

### 6.1 Data Preprocessing and Augmentation

Images are loaded using Keras `ImageDataGenerator` with `flow_from_dataframe()`, which maps pandas DataFrame rows to image files on disk. This approach is preferred due to its compatibility with the hash-based filename structure of the dataset.

**Standard preprocessing** applied to all splits:

- Pixel intensity normalization: rescaling by `1/255` to map values to `[0, 1]`
- Grayscale channel mode (`color_mode="grayscale"`) for custom CNN models
- RGB channel mode (`color_mode="rgb"`) for transfer learning models (InceptionV3 expects 3-channel input)
- Resize to target resolution (default: `299×299` px)

**Data augmentation** (tested as optional):

```python
ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=10,
    shear_range=5,
    zoom_range=0.05,
    rescale=1.0/255.0,
    validation_split=val_split
)
```

> **Finding:** Data augmentation was found to **decrease** model performance across all tested configurations. The likely cause is that the images are highly structured and orientation-invariant augmentations (particularly flips and rotations) introduce variation that does not reflect real-world variance in the data. Future experiments will explore more conservative augmentation strategies (e.g., subtle brightness/contrast jitter only).

### 6.2 Dataset Balancing Strategy

Given the extreme class imbalance (approximately 79% `Missing`), two strategies were evaluated:

**Undersampling (primary approach):** Each class is randomly undersampled to match the count of the smallest class (approximately 71 samples per class), producing a balanced dataset. While this significantly reduces the training data volume, it ensures the model receives equal exposure to each class during training.

**Unbalanced training:** Models are also trained on the full unbalanced dataset for comparison. Predictably, this produces high aggregate accuracy but poor minority-class performance.

**Planned:** Class-weighted loss as an alternative to undersampling, allowing all available data to be used while adjusting the gradient contribution per class.

### 6.3 Data Splits

| Split | Fraction | Source |
|---|---|---|
| Training | 64% | Balanced or unbalanced subset |
| Validation | 16% | Same source, `validation_split` applied within generator |
| Test | 20% | Held out via `train_test_split` before generator creation |

The test split is always performed before any generator initialization to prevent data leakage. The `random_state=42` seed is fixed throughout to ensure reproducibility.

### 6.4 Model Architectures

Four model families were implemented and evaluated:

#### Custom CNNs (Models 1–3)

All custom CNNs follow a sequential Conv→Pool→...→Flatten→Dense structure operating on single-channel (grayscale) input.

**Model 1 — Shallow CNN**

```
Input(299×299×1)
→ Conv2D(64, 3×3, ReLU) → MaxPool(2×2)
→ Flatten
→ Dense(64, ReLU)
→ Dense(6, Softmax)

Total params: ~89.7M (dominated by the first Dense layer after Flatten)
```

**Model 2 — Two-Layer CNN**

```
Input(299×299×1)
→ Conv2D(32, 3×3, ReLU) → MaxPool(2×2)
→ Conv2D(64, 3×3, ReLU) → MaxPool(2×2)
→ Flatten
→ Dense(128, ReLU)
→ Dense(6, Softmax)

Total params: ~43.7M
```

**Model 3 — Three-Layer CNN (Best)**

```
Input(299×299×1)
→ Conv2D(32, 3×3, ReLU) → MaxPool(2×2)
→ Conv2D(64, 3×3, ReLU) → MaxPool(2×2)
→ Conv2D(128, 3×3, ReLU) → MaxPool(2×2)
→ Flatten
→ Dense(128, ReLU)
→ Dense(64, ReLU)
→ Dense(6, Softmax)

Total params: ~20.2M
```

#### Transfer Learning — InceptionV3 (Feature Extraction)

```
InceptionV3(weights='imagenet', include_top=False, input_shape=(299×299×3))
→ [All base layers frozen]
→ GlobalAveragePooling2D()
→ Dropout(0.2)
→ Dense(6, Softmax)
```

InceptionV3 was chosen based on its strong performance in the SEM classification literature (Source 3) and its native support for the 299×299 input size used throughout this project. Feature extraction (all base layers frozen) was the first approach; fine-tuning is planned in future iterations.

### 6.5 Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam (default learning rate 0.001) |
| Loss Function | Categorical Crossentropy |
| Batch Size | 8 |
| Max Epochs | 100 |
| Early Stopping Monitor | `val_loss` |
| Early Stopping Min Delta | 0.01 |
| Early Stopping Patience | 7 (3 for transfer learning) |
| Restore Best Weights | True |
| Early Stop Start Epoch | 5 |
| Default Image Size | 299×299 px |

Early stopping with `restore_best_weights=True` ensures that the final model weights correspond to the epoch with the best validation loss, not the last training epoch. This is particularly important given the tendency of models to overfit on the small balanced training set.

### 6.6 Evaluation Metrics and Rationale

In the context of automated microelectronics inspection, the consequences of different error types are asymmetric:

- **Shipping a defective part** (false negative on a defect class) = high cost; the defect reaches the customer or downstream manufacturing
- **Discarding a good part** (false positive on a defect class) = lower cost; unnecessary waste but no downstream harm

This business context motivates the following metric choices:

| Metric | Primary Use |
|---|---|
| **Recall** | Primary metric for defect and Drift classes — minimize false negatives |
| **Precision** | Primary metric for the `GOOD` class — avoid incorrectly certifying defective parts |
| **F1-Score (weighted)** | Balanced trade-off used for overall model comparison and hyperparameter selection |
| **Accuracy** | Reported for completeness; weighted heavily by the dominant class on unbalanced sets |
| **Per-class F1-Score** | Used to assess minority class performance; the worst-class F1 is tracked closely |

All final results are evaluated on a held-out balanced test set to ensure meaningful per-class metrics that are not dominated by the `Missing` class.

---

## 7. Baseline Model

**Notebook:** `2_BaselineModel/`

### Architecture

A minimal CNN was selected as the baseline: one convolutional layer, one pooling layer, and one dense layer. No data augmentation was applied. Training was performed on the full unbalanced dataset with images downsampled to 128×128 px.

```
Input(128×128×1)
→ Conv2D(32, 3×3, ReLU) → MaxPool(2×2)
→ Flatten
→ Dense(64, ReLU)
→ Dense(6, Softmax)
```

### Rationale

Unlike tabular regression or time-series tasks, image classification has no natural non-neural baseline (such as a decision tree or random forest) without manual feature engineering. The simplest possible CNN was therefore chosen as the reference point. It is expressive enough to perform the classification in principle while leaving clear room for architectural improvement.

### Results

| Metric | Value |
|---|---|
| Overall Accuracy | 0.988 (unbalanced full test set) |
| Weighted F1-Score | 0.988 |
| Worst-class F1 (Flat Loop) | 0.61 |

**Classification Report (Unbalanced Test Set, n=1,656):**

```
                     precision  recall  f1-score  support
0_GOOD                    0.99    0.99      0.99      238
1_Flat loop               0.88    0.47      0.61       15
2_White lift-off          0.87    0.96      0.92       57
3_Black lift-off          0.93    1.00      0.96       13
4_Missing                 1.00    1.00      1.00     1316
5_Short circuit MOS       0.87    0.76      0.81       17
accuracy                                   0.99     1656
macro avg                 0.92    0.86      0.88     1656
weighted avg              0.99    0.99      0.99     1656
```

While the baseline delivers surprisingly high aggregate performance (inflated by the dominant `Missing` class), the worst-class F1 of 0.61 for `Flat loop` reveals the expected weakness on minority classes. All subsequent models are evaluated on a balanced test set for fair comparison.

---

## 8. Model Definition and Evaluation

**Notebook:** `3_Model/model_definition_evaluation.ipynb`

All models in this section were trained on the **balanced dataset** with images at **299×299 px** and **no data augmentation** unless otherwise noted. The test set is the balanced held-out split (86 samples, approximately 14–18 per class).

### 8.1 Model 1 — Shallow CNN

A near-baseline model with a single convolutional layer, included to measure the marginal gain from adding more convolutional depth relative to the true baseline.

**Test Results:**

| Accuracy | Precision | Recall | F1 Score (weighted) |
|---|---|---|---|
| 0.767 | 0.767 | 0.767 | 0.766 |

**Per-Class Performance:**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| 0_GOOD | 0.63 | 0.71 | 0.67 |
| 1_Flat loop | 0.79 | 0.61 | 0.69 |
| 2_White lift-off | 0.64 | 1.00 | 0.78 |
| 3_Black lift-off | 1.00 | 0.85 | 0.92 |
| 4_Missing | 1.00 | 1.00 | 1.00 |
| 5_Short circuit MOS | 0.82 | 0.60 | 0.69 |

Training converged in 14 epochs (early stopping). The model exhibits signs of overfitting: training accuracy reaches 1.00 while validation accuracy plateaus around 0.91.

### 8.2 Model 2 — Two-Layer CNN

**Test Results:**

| Accuracy | Precision | Recall | F1 Score (weighted) |
|---|---|---|---|
| 0.849 | 0.867 | 0.837 | 0.850 |

**Per-Class Performance:**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| 0_GOOD | 0.94 | 0.88 | 0.91 |
| 1_Flat loop | 0.93 | 0.72 | 0.81 |
| 2_White lift-off | 0.68 | 0.93 | 0.79 |
| 3_Black lift-off | 0.85 | 0.85 | 0.85 |
| 4_Missing | 1.00 | 0.78 | 0.88 |
| 5_Short circuit MOS | 0.82 | 0.93 | 0.88 |

Adding a second convolutional block delivers a substantial improvement: +8.2% accuracy and +8.4% weighted F1 over Model 1. Training converged in 9 epochs.

### 8.3 Model 3 — Three-Layer CNN (Best)

**Test Results:**

| Accuracy | Precision | Recall | F1 Score (weighted) |
|---|---|---|---|
| 0.884 | 0.893 | 0.872 | 0.883 |

**Per-Class Performance:**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| 0_GOOD | 0.84 | 0.94 | 0.89 |
| 1_Flat loop | 1.00 | 0.72 | 0.84 |
| 2_White lift-off | 0.76 | 0.93 | 0.84 |
| 3_Black lift-off | 0.85 | 0.85 | 0.85 |
| 4_Missing | 1.00 | 1.00 | 1.00 |
| 5_Short circuit MOS | 0.93 | 0.93 | 0.93 |

Model 3 is the best-performing architecture across all metrics. Adding a third convolutional layer and an additional dense layer allows the model to learn progressively more abstract features, yielding +3.5% accuracy and +3.3% weighted F1 over Model 2. The `Missing` class achieves perfect F1 of 1.00 and `Short circuit MOS` — the visually most ambiguous class — reaches F1 of 0.93. Training converged in 13 epochs.

### 8.4 InceptionV3 — Transfer Learning (Feature Extraction)

**Test Results:**

| Accuracy | Precision | Recall | F1 Score (weighted) |
|---|---|---|---|
| 0.826 | 0.827 | 0.779 | 0.827 |

**Per-Class Performance:**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| 0_GOOD | 0.74 | 1.00 | 0.85 |
| 1_Flat loop | 1.00 | 0.67 | 0.80 |
| 2_White lift-off | 0.63 | 0.86 | 0.73 |
| 3_Black lift-off | 0.91 | 0.77 | 0.83 |
| 4_Missing | 1.00 | 1.00 | 1.00 |
| 5_Short circuit MOS | 0.92 | 0.73 | 0.81 |

Despite using a significantly larger and more powerful pretrained backbone, InceptionV3 in feature-extraction mode underperforms Model 3 by approximately 5.6% weighted F1. The most likely causes are: (1) the pretrained weights encode RGB ImageNet features which do not transfer optimally to single-channel SEM-style images; and (2) pure feature extraction without fine-tuning limits the model's ability to adapt to the distribution shift between ImageNet and industrial microelectronics imagery. Training converged in 17 epochs.

---

## 9. Results Summary

### 9.1 Model Comparison Table

All results evaluated on the balanced held-out test set (86 samples, 299×299 px input, no augmentation).

| Model | Params | Test Accuracy | Precision | Recall | Weighted F1 | Epochs to Converge |
|---|---|---|---|---|---|---|
| Baseline CNN (128px, unbalanced) | — | 0.988 | 0.99 | 0.99 | 0.988 | — |
| Model 1 — Shallow CNN | ~89.7M | 0.767 | 0.767 | 0.767 | 0.766 | 14 |
| Model 2 — Two-Layer CNN | ~43.7M | 0.849 | 0.867 | 0.837 | 0.850 | 9 |
| **Model 3 — Three-Layer CNN** | **~20.2M** | **0.884** | **0.893** | **0.872** | **0.883** | **13** |
| InceptionV3 (Feature Extraction) | ~22.0M* | 0.826 | 0.827 | 0.779 | 0.827 | 17 |

*InceptionV3 base model frozen; only the classification head (~20K params) is trainable.

> The full model comparison across all hyperparameter configurations (image size, augmentation, balancing strategy) is available in `3_Model/Model_Performance_overview.xlsx` and `Model_Performance_overview.csv`.

### 9.2 Best Model Classification Report

**Model 3 — Three-Layer CNN** | Balanced Test Set | 299×299 px | No Augmentation

```
                     precision  recall  f1-score  support
0_GOOD                    0.84    0.94      0.89       17
1_Flat loop               1.00    0.72      0.84       18
2_White lift-off          0.76    0.93      0.84       14
3_Black lift-off          0.85    0.85      0.85       13
4_Missing                 1.00    1.00      1.00        9
5_Short circuit MOS       0.93    0.93      0.93       15
accuracy                                   0.88       86
macro avg                 0.90    0.90      0.89       86
weighted avg              0.90    0.88      0.88       86
```

### 9.3 Key Findings and Insights

**Architecture depth matters monotonically.** Moving from 1 to 3 convolutional blocks yields consistent, monotonic improvements across all evaluation metrics. The additional layers enable the model to extract progressively more abstract spatial hierarchies from the images, which is critical for distinguishing visually similar defect types.

**Transfer learning underperforms custom CNNs at this stage.** InceptionV3 in feature-extraction mode performed worse than Model 3, which has significantly fewer total parameters. The key limitation is the domain gap: InceptionV3 was trained on RGB natural images (ImageNet), while the task involves grayscale SEM-style industrial images with highly structured, periodic patterns. Fine-tuning the pretrained layers is expected to substantially close this gap and is a priority for future work.

**Data augmentation is counterproductive.** Across all tested models and configurations, augmentation consistently reduced performance. The industrial images exhibit strong structural regularity — conductor tracks are always oriented the same way, and the scale of features is consistent. Flips and rotations introduce augmented samples that do not correspond to any real-world variation in the imaging process, effectively adding noise to the training distribution.

**Higher resolution does not improve performance.** Models trained on 128×128 px images performed comparably to those trained on 299×299 px. The discriminative features for most classes appear to be captured at low resolutions, and the computational overhead of higher-resolution inputs is not justified by the performance gains observed.

**Class balancing is essential.** Without undersampling, the models achieve very high aggregate accuracy (~99%) but fail on minority classes (F1 as low as 0.61 for `Flat loop`). Balanced training dramatically improves minority class recall and F1, at the cost of working with a much smaller effective dataset.

**`Missing` is trivially classifiable.** The structural absence of the bridge feature makes `Missing` images highly distinct; all models — including the baseline — achieve near-perfect F1 on this class. The real challenge lies in distinguishing the visually similar defect subtypes and the `GOOD`/`Short circuit MOS` pair.

---

## 10. Repository Structure

```
MachineVision_Valeo_ChallangeData/
│
├── 0_LiteratureReview/
│   └── README.md                              # Literature review summaries and source links
│
├── 1_DatasetCharacteristics/
│   └── exploratory_data_analysis.ipynb        # Full EDA: distributions, quality checks, visualizations
│
├── 2_BaselineModel/
│   ├── baseline_model.ipynb                   # Baseline CNN training and evaluation
│   └── README.md                              # Baseline results and methodology notes
│
├── 3_Model/
│   ├── model_definition_evaluation.ipynb      # Main training notebook: Models 1–3 + InceptionV3
│   ├── model_definition_evaluation_testing.ipynb  # Experimental and testing notebook
│   ├── ManualDriftClassLabeling.ipynb         # Manual labeling script for Drift class identification
│   ├── Model Evaluations.ipynb                # Consolidated evaluation across all models
│   ├── Model_Performance_overview.csv         # Model performance comparison table (CSV)
│   ├── Model_Performance_overview.xlsx        # Model performance comparison table (Excel)
│   ├── models.py                              # Standalone model architecture definitions
│   ├── INSTRUCTIONS.md                        # Setup and usage instructions
│   └── README.md                              # Milestone 3 & 4 documentation
│
├── 4_Presentation/
│   └── [presentation files]                   # Final project presentation slides
│
├── CoverImage/
│   └── cover_image.png                        # Project cover image
│
├── Testing/Niklas/
│   └── [experimental notebooks]               # Sandbox and experimental work
│
├── machine-learning-with-tensorflow/
│   └── week-06/                               # Course week 6 Colab material
│
├── .gitignore
├── LICENSE                                     # Apache 2.0 License
└── README.md                                   # This file
```

---

## 11. Getting Started

### 11.1 Prerequisites

- Python 3.8 or higher
- pip package manager
- A CUDA-compatible GPU is strongly recommended for reasonable training times; CPU-only training is possible but slow at 299×299 px input resolution

**Required Python packages:**

| Package | Minimum Version | Purpose |
|---|---|---|
| tensorflow | 2.10 | Model training and inference |
| scikit-learn | 1.0 | Train/test splits, classification report |
| pandas | 1.3 | DataFrame handling and label CSV parsing |
| numpy | 1.21 | Array operations |
| matplotlib | 3.4 | Visualization and training plots |
| seaborn | 0.11 | Statistical distribution plots |
| Pillow | 8.0 | Image loading and validation |
| openpyxl | 3.0 | Excel output for performance overview tables |

### 11.2 Installation

Clone the repository and install all dependencies:

```bash
git clone https://github.com/Sl4artiB4rtF4rst/MachineVision_Valeo_ChallangeData.git
cd MachineVision_Valeo_ChallangeData

pip install tensorflow scikit-learn pandas numpy matplotlib seaborn Pillow openpyxl
```

For GPU support, ensure your CUDA and cuDNN versions are compatible with your TensorFlow version. Refer to the [TensorFlow GPU installation guide](https://www.tensorflow.org/install/gpu) for version compatibility tables.

### 11.3 Data Setup

1. Register and accept the terms at [ChallengeData ENS — Challenge #157](https://challengedata.ens.fr/participants/challenges/157/)
2. Download the training images archive and the label CSV file
3. Organize your local data directory as follows:

```
<your_base_file_path>/
├── input_train/
│   └── input_train/        # Extract all 8,278 training images here
└── Y_train_eVW9jym.csv     # Place the label CSV here
```

4. Update the `base_file_path` variable at the top of each notebook to point to your local data directory:

```python
base_file_path = '/your/local/path/to/challenge_data/'
image_path = base_file_path + '/input_train/input_train'
```

### 11.4 Running the Pipeline

**Step 1 — Exploratory Data Analysis**

Open and run `1_DatasetCharacteristics/exploratory_data_analysis.ipynb` first to validate your data setup, inspect the label distribution, and reproduce the EDA findings described in Section 4.

**Step 2 — Baseline Model**

Open `2_BaselineModel/baseline_model.ipynb` to train and evaluate the baseline CNN on the full unbalanced dataset. This establishes the reference performance point.

**Step 3 — Main Model Training and Evaluation**

Open `3_Model/model_definition_evaluation.ipynb`. Configure the key hyperparameters at the top of the notebook (see Section 12), then run all cells to train Models 1–3 and InceptionV3, evaluate each on the held-out test set, and save all results to disk.

**Step 4 — Review Consolidated Results**

Open `3_Model/Model Evaluations.ipynb` or inspect `Model_Performance_overview.xlsx` for a side-by-side comparison of all model runs across all tested hyperparameter configurations.

**Step 5 — Drift Class Labeling (optional)**

Open `3_Model/ManualDriftClassLabeling.ipynb` to label a subset of the public test data as `Drift` for use in future model improvements targeting the out-of-distribution detection task.

---

## 12. Hyperparameter Reference

The following hyperparameters are set at the top of `model_definition_evaluation.ipynb` and control the full training pipeline:

```python
# --- Data / Feature Selection ---
balanced_flag = True          # True = undersample to balanced dataset
                              # False = use full unbalanced data

# --- Train/Test Splits ---
test_split  = 0.20            # Fraction of data reserved for test set
val_split   = 0.20            # Fraction of remaining training data for validation
                              # Note: this fraction is applied AFTER test split

# --- Image Parameters ---
target_size = (299, 299)      # Input image resolution in pixels (width, height)
batch_size  = 8               # Samples per training batch

# --- Data Augmentation ---
aug_flag = False              # True = apply augmentation; False = no augmentation (recommended)

# --- Training ---
max_epochs          = 100     # Maximum training epochs (early stopping will trigger earlier)
loss_stop_patience  = 7       # Epochs without improvement before early stopping triggers
```

A structured naming convention based on these hyperparameters organizes all model outputs:

```python
hyperparam_name = 'ImgSz_{}_{}_{}'.format(target_size[0], augmentation_str, balance_str)
# Example output directory name: 'ImgSz_299_NoAug_balanced'
```

Each configuration creates its own subdirectory under `model_evaluation/` containing per-model classification reports (`.csv`) and saved model weights (`.keras` format).

---

## 13. Known Limitations

**Small effective training set after balancing.** Undersampling to approximately 71 samples per class results in only around 340 total training samples (after the test split). This is a fundamentally limiting factor; the models generalize well relative to this constraint, but further improvement will likely require either more labeled data for minority classes or alternative balancing strategies such as class-weighted loss.

**Drift class not yet handled in supervised training.** The `Drift` class — which carries additional scoring penalty in the competition — is not present in the training labels and cannot be learned via standard supervised training. A separate strategy such as anomaly detection, one-class classification, or manual labeling of test data is required and is currently in progress via `ManualDriftClassLabeling.ipynb`.

**Suspected label noise in minority classes.** Visual inspection identified likely mislabeled samples in `Lift-off blanc` and `Lift-off noir`. This introduces noise into the training signal for already low-count classes and may be suppressing achievable F1 for these categories.

**`Short circuit MOS` visual ambiguity.** This class is reportedly indistinguishable from `GOOD` by visual inspection. While Model 3 achieves F1 = 0.93 on this class in the balanced test set, performance may not generalize reliably to the full unbalanced test distribution, where the model has less exposure to this minority class during training.

**No fine-tuning for transfer learning.** InceptionV3 was only evaluated in feature-extraction mode (all base layers frozen). Fine-tuning the top convolutional layers is expected to substantially improve transfer learning performance and is scheduled as an immediate next step.

**No systematic hyperparameter search.** All hyperparameter variation to date has been performed manually by testing individual configurations. A structured grid or random search over key parameters (learning rate, batch size, dropout rate, convolutional filter counts) has not yet been performed.

**Single-modality input.** The structured metadata features `lib` (die type) and `window` (year) have not yet been incorporated as model inputs. These features likely carry discriminative signal, particularly for visually ambiguous classes correlated with specific die types.

---

## 14. Next Steps and Future Work

- [ ] **Transfer learning with fine-tuning:** Unfreeze the top N layers of InceptionV3 and fine-tune end-to-end on the target domain; also evaluate EfficientNet-B0/B3 and ResNet50
- [ ] **Alternative pretrained architectures:** Benchmark MobileNetV3, EfficientNet variants, and architectures pretrained on grayscale or medical/industrial imagery
- [ ] **Drift class detection:** Implement an unsupervised or semi-supervised approach for out-of-distribution detection — candidate methods include autoencoder reconstruction error thresholding, one-class SVM on CNN feature embeddings, and OpenMax
- [ ] **Continued manual Drift labeling:** Expand the manually labeled `Drift` subset from the public test data to enable supervised training; the `ManualDriftClassLabeling.ipynb` workflow supports this
- [ ] **Class-weighted loss:** Replace undersampling with a class-weighted `CategoricalCrossentropy` loss to allow all 8,278 training samples to be used while correcting for class imbalance in the gradient signal
- [ ] **Conservative augmentation strategies:** Re-test augmentation limited to subtle brightness variation, contrast jitter, and low-amplitude Gaussian noise — excluding any spatial transformations (flips, rotations, shear)
- [ ] **Systematic hyperparameter tuning:** Apply Keras Tuner (random search or Hyperband) over learning rate, dropout rate, batch size, and layer filter counts for the Model 3 architecture
- [ ] **Metadata feature fusion:** Incorporate `lib` (die type) and `window` (year) as categorical embeddings concatenated to CNN feature maps before the dense classification head
- [ ] **Detailed error analysis:** Systematic examination of misclassified samples for each model, with attention to whether errors cluster by die type, year, or resolution — to inform targeted data collection and architecture decisions

---

## 15. Contributors

| Contributor | GitHub Profile |
|---|---|
| Sl4artiB4rtF4rst | [@Sl4artiB4rtF4rst](https://github.com/Sl4artiB4rtF4rst) |
| Aadip Thapaliya | [@Aadip-Thapaliya](https://github.com/Aadip-Thapaliya) |

This project was completed as part of the **OpenCampus SH** course *"Machine Learning with TensorFlow"* (2024/2025 cohort), developed from the [opencampus-sh/ml-project-template](https://github.com/opencampus-sh/ml-project-template).

---

## 16. License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for the full terms and conditions.

The dataset is provided by Valeo via the ChallengeData platform and is subject to the platform's own terms of use. The dataset is **not redistributed** in this repository. Users must register independently at [challengedata.ens.fr](https://challengedata.ens.fr/participants/challenges/157/) to obtain access to the data.

---

## 17. Cover Image

![Project Cover Image](https://github.com/Sl4artiB4rtF4rst/MachineVision_Valeo_ChallangeData/blob/main/CoverImage/cover_image.png)

> *Machine vision imagery of microelectronic conductor tracks as provided in the Valeo ChallengeData dataset. Images depict top-down views of microelectronic structures captured via optical camera or scanning electron microscope (SEM). Classification labels describe the fabrication and structural integrity status of the depicted bridge-like conductive features.*
