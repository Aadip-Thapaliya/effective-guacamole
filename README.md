# Microelectronics Defect Classification — Valeo Challenge

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)

**Repository:** [https://github.com/Sl4artiB4rtF4rst/MachineVision_Valeo_ChallangeData](https://github.com/Sl4artiB4rtF4rst/MachineVision_Valeo_ChallangeData)

---

## Project Overview

This project was developed as part of the **OpenCampus course "Machine Learning with TensorFlow"**. The goal is to classify machine-vision images of microelectronic components provided by the French-Chinese electronics company **Valeo** via the [ChallengeData platform](https://challengedata.ens.fr/participants/challenges/157/).

Each image depicts conductor tracks on a microelectronic chip, and the task is to classify the fabrication status of the depicted structure into one of 6 (or 7, including *Drift*) categories.

---

## Dataset

- **Source:** [ChallengeData ENS — Challenge #157](https://challengedata.ens.fr/participants/challenges/157/)
- **Total images:** 8,278 training images + 1,055 test images
- **Image format:** Grayscale, 8-bit, top-down view (likely optical or scanning electron microscope)
- **Image resolution:** 530×530 px to 1,260×1,260 px

### Features

| Feature | Description |
|---|---|
| `filename` | Image file (grayscale camera/SEM image) |
| `window` | Year of manufacture/inspection — values: `2003`, `2005` |
| `lib` | Die type — values: `Die01`, `Die02`, `Die03`, `Die04` |

### Labels

| Label | Description |
|---|---|
| `GOOD` | Fully functioning structure with all conductive layers present |
| `Flat Loop` | Bridge-like features with an abnormal surface structure |
| `White Lift-off` | Missing layer in conductive track (visible brightness difference); bridge present |
| `Black Lift-off` | Visually near-identical to White Lift-off; difficult to distinguish |
| `Missing` | Bridge-like structure or contacts missing entirely |
| `Short Circuit MOS` | Visually indistinguishable from GOOD; limited trainable signal |
| `Drift` *(test only)* | Does not belong to any of the above classes; penalized more heavily in scoring |

### Class Distribution

Labels are **heavily imbalanced**: approximately 6,500 images are labeled `Missing`, ~1,200 are `GOOD`, and the remaining defect categories each have fewer than 500 samples.

---

## Results Summary

### Model Comparison (Balanced Dataset, 299×299px, No Augmentation)

| Model | Test Accuracy | Test F1 Score (weighted) |
|---|---|---|
| Baseline CNN (1 conv layer) | 0.988 (unbalanced full set) | 0.988 |
| Model 1 — Shallow CNN | 0.767 | 0.766 |
| Model 2 — 2-Layer CNN | 0.849 | 0.850 |
| **Model 3 — 3-Layer CNN** | **0.884** | **0.883** |
| InceptionV3 Feature Extraction | 0.826 | 0.827 |

### Best Model

**Model 3 — 3-Layer CNN** achieved the best performance overall with a weighted F1 score of **0.883** on the balanced test set.

```
Architecture:
  Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Conv2D(128) → MaxPool
  → Flatten → Dense(128) → Dense(64) → Dense(6, softmax)
```

### Classification Report — Model 3

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| 0_GOOD | 0.84 | 0.94 | 0.89 |
| 1_Flat loop | 1.00 | 0.72 | 0.84 |
| 2_White lift-off | 0.76 | 0.93 | 0.84 |
| 3_Black lift-off | 0.85 | 0.85 | 0.85 |
| 4_Missing | 1.00 | 1.00 | 1.00 |
| 5_Short circuit MOS | 0.93 | 0.93 | 0.93 |

---

## Key Findings

- **Simple CNNs outperform transfer learning** at low-to-medium image resolutions. InceptionV3 feature extraction underperformed custom CNNs, likely due to mismatched input size and the grayscale nature of images (pretrained on RGB ImageNet data).
- **Data augmentation hurt performance** across all tested configurations. The images are highly regular and orientation-invariant augmentations (flips, rotations) appear to introduce misleading variation.
- **Higher image resolution** did not meaningfully improve classification accuracy in tested configurations.
- **Dataset balancing** is essential: the raw class imbalance (6,500 vs ~70 samples for rare classes) leads to artificially high accuracy that masks poor minority-class performance.
- **`Missing` and `GOOD`** are the easiest classes to classify. `Flat loop` and visual-noise classes like `Short circuit MOS` are hardest.

---

## Repository Structure

```
MachineVision_Valeo_ChallangeData/
├── 0_LiteratureReview/          # Literature review notes and summaries
├── 1_DatasetCharacteristics/    # EDA notebook and dataset analysis
├── 2_BaselineModel/             # Baseline CNN training and evaluation
├── 3_Model/                     # Final model notebooks and scripts
│   ├── model_definition_evaluation.ipynb
│   ├── model_definition_evaluation_testing.ipynb
│   ├── ManualDriftClassLabeling.ipynb
│   ├── Model Evaluations.ipynb
│   ├── Model_Performance_overview.csv / .xlsx
│   └── models.py
├── 4_Presentation/              # Project presentation slides
├── CoverImage/                  # Cover image for repository
├── Testing/Niklas/              # Experimental / sandbox notebooks
└── machine-learning-with-tensorflow/week-06/
```

---

## Getting Started

### Prerequisites

- Python 3.x
- TensorFlow / Keras 2.x
- scikit-learn
- pandas, numpy, matplotlib, seaborn, Pillow

Install dependencies:

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn Pillow
```

### Data Setup

Download the dataset from [ChallengeData #157](https://challengedata.ens.fr/participants/challenges/157/) and place it as follows:

```
<base_file_path>/
├── input_train/input_train/   # Training images
└── Y_train_eVW9jym.csv        # Labels CSV
```

Update `base_file_path` in the notebooks accordingly.

### Running the Models

1. Open `3_Model/model_definition_evaluation.ipynb`
2. Set your `base_file_path` to point to your local data directory
3. Adjust hyperparameters at the top of the notebook (image size, augmentation, balancing)
4. Run all cells to train and evaluate all four model configurations

---

## Methodology

### Data Preprocessing

- Images are loaded via `ImageDataGenerator.flow_from_dataframe()`
- Label strings are mapped to numbered classes for compatibility
- **Dataset balancing:** undersampling to the size of the smallest class (~71 samples/class)
- Train/Validation/Test split: **64% / 16% / 20%**

### Models Trained

- **Model 1:** 1 Conv layer, 1 Dense layer (baseline-equivalent)
- **Model 2:** 2 Conv layers, 1 Dense layer
- **Model 3:** 3 Conv layers, 2 Dense layers (best performer)
- **InceptionV3 (Feature Extraction):** Frozen pretrained base + GlobalAveragePooling + Dense head

### Training Setup

- Optimizer: Adam
- Loss: Categorical Crossentropy
- Early stopping on validation loss (patience = 7, `restore_best_weights=True`)
- Metrics: Accuracy, Precision, Recall, F1-Score (weighted)

### Evaluation Rationale

In automated microelectronics inspection, **false negatives on defective parts are more costly than false positives** (shipping a faulty component is worse than discarding a good one). This motivates:

- Emphasis on **recall for defect classes**
- Emphasis on **precision for the GOOD class**
- Using **F1-Score** as primary balanced trade-off metric

---

## Literature Review

| Source | Key Takeaway |
|---|---|
| [CNN Review for Industrial Defect Detection](https://doi.org/) | Transfer learning from ImageNet models widely used; provides overview of architectures and benchmarks |
| [Defect Detection in Li-Ion Battery Electrodes](https://doi.org/) | Fine-tuned CNNs achieve F1 ≈ 0.99 with ~3,200 images; supports viability of our approach |
| [Neural Networks for SEM Image Recognition](https://doi.org/) | 85–95% accuracy on SEM-style images with 10 classes; Inception-v3 fastest and most accurate |

---

## Next Steps

- [ ] **Transfer learning with fine-tuning** (unfreeze top layers of InceptionV3/EfficientNet)
- [ ] **Alternative pretrained architectures** (EfficientNet, ResNet, MobileNet)
- [ ] **Drift class handling** — manual labeling of a subset of test data, or unsupervised anomaly detection approach
- [ ] **Class weighting** as alternative to dataset balancing (avoids data loss)
- [ ] **Systematic hyperparameter tuning** (grid search / random search)
- [ ] **Refined augmentation** (exclude flips, test subtle brightness/contrast shifts only)

---

## Contributors

- [@Sl4artiB4rtF4rst](https://github.com/Sl4artiB4rtF4rst)
- [@Aadip-Thapaliya](https://github.com/Aadip-Thapaliya)

---

## License

This project is licensed under the **Apache 2.0 License** — see the [LICENSE](LICENSE) file for details.
