# CSA-ViT: Cross-Scale Attention Vision Transformer for Multi-Class Brain Tumor Classification Using Multi-Sequence MRI

Research implementation accompanying **CSA-ViT (Cross-Scale Attention Vision Transformer)** for multi-class brain tumor classification using multi-sequence magnetic resonance imaging (MRI).

The proposed framework integrates **hierarchical multi-scale feature learning, cross-scale attention, adaptive MRI sequence fusion, and Transformer-based global contextual modeling** for brain tumor classification.

---

## Overview

Accurate brain tumor classification from MRI is challenging because tumors exhibit substantial variations in size, morphology, texture, and spatial appearance. Different MRI sequences also provide complementary information about tumor characteristics and surrounding tissues.

CSA-ViT addresses these challenges through the following components:

- Lightweight CNN stem for local feature extraction
- Hierarchical multi-scale patch embedding
- Hierarchical Cross-Scale Attention Module (HCSAM)
- Adaptive Sequence Attention Module (SAM)
- Transformer-based global contextual modeling
- Attention-based feature aggregation
- Confidence-aware predictive uncertainty analysis

The framework jointly exploits spatial information at multiple resolutions and complementary information across **T1, T1CE, T2, and FLAIR** MRI sequences.

---

# Architecture

The overall CSA-ViT pipeline is:

```text
                         Multi-Sequence MRI
        ┌──────────────┬──────────────┬──────────────┬──────────────┐
        │     T1       │    T1CE      │      T2      │    FLAIR     │
        └──────┬───────┴──────┬───────┴──────┬───────┴──────┬───────┘
               │              │              │              │
               └──────────────┴──────────────┴──────────────┘
                                      │
                               MRI Preprocessing
                                      │
                              Lightweight CNN Stem
                                      │
                    ┌─────────────────┼─────────────────┐
                    │                 │                 │
                8 × 8 patches    16 × 16 patches    32 × 32 patches
                    │                 │                 │
               784 tokens        196 tokens         49 tokens
                    │                 │                 │
                    └─────────────────┼─────────────────┘
                                      │
                    Hierarchical Cross-Scale Attention
                                  HCSAM
                                      │
                                      ↓
                         Adaptive Sequence Attention
                                     SAM
                                      │
                                      ↓
                              Transformer Encoder
                                      │
                                      ↓
                         Attention-Based Aggregation
                                      │
                                      ↓
                              Classification Head
                                      │
                   ┌──────────┬──────┼──────────┬──────────┐
                   │          │      │          │
                No-tumor   Glioma  Meningioma  Pituitary
```

---

# Key Components

## 1. MRI Preprocessing

The preprocessing pipeline consists of:

1. Bias-field correction
2. Skull stripping
3. Intensity standardization
4. Resizing to `224 × 224` pixels

The four MRI sequences used by the framework are:

- T1
- T1CE
- T2
- FLAIR

---

## 2. Lightweight CNN Stem

The CNN stem consists of two consecutive convolutional layers.

### Configuration

- Kernel size: `3 × 3`
- Stride: `1`
- Padding: spatial-resolution preserving
- Batch Normalization
- GELU activation
- CNN channels: `64 → 128`

The CNN stem extracts local structural information before hierarchical multi-scale token generation.

---

## 3. Hierarchical Multi-Scale Patch Embedding

CSA-ViT uses three spatial resolutions:

| Patch Size | Token Grid | Number of Tokens |
|---|---:|---:|
| `8 × 8` | `28 × 28` | `784` |
| `16 × 16` | `14 × 14` | `196` |
| `32 × 32` | `7 × 7` | `49` |

The three resolutions capture:

- Fine-grained local characteristics
- Intermediate regional structures
- Broader anatomical and contextual information

The resulting representations are denoted as fine-scale, intermediate-scale, and coarse-scale feature representations.

---

## 4. Hierarchical Cross-Scale Attention Module (HCSAM)

HCSAM explicitly models interactions between adjacent spatial resolutions.

```text
Fine-scale representation
          Es
          │
          ↓
   Cross-Scale Attention
          │
          ↓
Intermediate representation
          Em
          │
          ↓
   Cross-Scale Attention
          │
          ↓
Coarse-scale representation
          El
```

For the `Es → Em` interaction:

```text
Query  = Es
Key    = Em
Value  = Em
```

The query resolution is preserved during the cross-scale interaction.

This enables fine-grained representations to incorporate complementary regional and contextual information from the adjacent scale.

The same hierarchical interaction is subsequently applied between intermediate- and coarse-scale representations.

### HCSAM configuration

- Embedding dimension: `256`
- Attention heads: `8`
- Head dimension: `32`
- HCSAM blocks: `6`

---

## 5. Adaptive Sequence Attention Module (SAM)

The four MRI sequences are processed to obtain modality-specific representations:

```text
T1       ─┐
T1CE     ─┤
T2       ─┤ → Modality-specific representations
FLAIR    ─┘
                 │
                 ↓
        Common 256-dimensional space
                 │
                 ↓
       Adaptive sequence weighting
                 │
                 ↓
          Fused representation
```

Before adaptive fusion, modality-specific representations are projected into a common embedding dimension of `256`.

The sequence attention module computes learned attention scores for the four MRI sequences and converts them into normalized modality coefficients.

The resulting coefficients are used for weighted sequence aggregation.

This allows the network to adaptively emphasize diagnostically informative MRI sequences rather than assigning equal importance to all modalities.

> **Note:** SAM is an adaptive sequence-fusion module and should not be interpreted as a conventional multi-head self-attention layer.

---

## 6. Transformer Encoder

The fused representation is processed using stacked Transformer encoder layers.

| Parameter | Configuration |
|---|---:|
| Embedding dimension | `256` |
| Attention heads | `8` |
| Transformer layers | `6` |
| MLP ratio | `4` |
| Dropout | `0.1` |
| Activation | GELU |

The Transformer encoder models long-range contextual relationships among the feature tokens.

---

## 7. Attention-Based Feature Aggregation

Instead of relying only on conventional average pooling, CSA-ViT uses learned token-importance scores to obtain the global representation.

The aggregated representation is subsequently passed through a multilayer perceptron classification head.

The classifier predicts one of four diagnostic categories:

1. No-tumor
2. Glioma
3. Meningioma
4. Pituitary tumor

---

## 8. Confidence-Aware Predictive Uncertainty

Predictive uncertainty is estimated from the class-probability distribution using normalized predictive entropy.

Lower entropy indicates a more concentrated class-probability distribution and therefore greater predictive confidence.

Higher entropy indicates greater ambiguity among the predicted classes.

The framework additionally evaluates calibration using:

- Expected Calibration Error (ECE)
- Negative Log-Likelihood (NLL)
- Brier score
- Normalized predictive entropy

---

# Dataset

The study uses a clinically collected multi-sequence MRI dataset comprising examinations from **200 patients**.

The classification task contains four diagnostic categories:

- No-tumor
- Glioma
- Meningioma
- Pituitary tumor

A total of **1,800 selected MRI samples** are used, with:

| Class | Samples |
|---|---:|
| No-tumor | 450 |
| Glioma | 450 |
| Meningioma | 450 |
| Pituitary | 450 |
| **Total** | **1,800** |

The samples represent selected MRI images obtained from patient examinations and therefore do not represent 1,800 independent patients.

The clinical MRI dataset was obtained from **Adichunchanagiri Research Centre** with institutional permission for research use.

---

# Data Availability

The clinical MRI dataset is **not included in this repository** because it was obtained under institutional permission.

Researchers wishing to reproduce the experiments must obtain appropriate authorization and comply with applicable ethical, privacy, and data-use requirements.

No patient-identifying information should be uploaded to this repository.

---

# Data Organization

The implementation uses CSV files to define the dataset splits.

The expected dataset organization is:

```text
data/
├── images/
│   ├── sample_0001/
│   │   ├── T1.png
│   │   ├── T1CE.png
│   │   ├── T2.png
│   │   └── FLAIR.png
│   │
│   └── ...
│
├── train.csv
├── val.csv
└── test.csv
```

The exact CSV fields should correspond to the implementation in:

```text
data/dataset.py
```

Patient identifiers should be retained in the dataset metadata to ensure patient-level partitioning.

---

# Patient-Level Data Partitioning

Data partitioning is performed at the **patient level** to prevent information leakage.

Images originating from the same patient must remain within the same subset.

The experimental protocol uses a **70:15:15 patient-level split**:

```text
                    200 Patients
                         │
                         ↓
              Patient-Level Partitioning
                         │
              ┌──────────┼──────────┐
              ↓          ↓          ↓
           Training   Validation   Test
             70%         15%        15%
              │           │          │
           1,260         270        270
           samples      samples    samples
```

All MRI sequences belonging to a patient are kept within the same partition.

The internal held-out test set contains **270 samples**, distributed as:

| Class | Test Samples |
|---|---:|
| No-tumor | 67 |
| Glioma | 68 |
| Meningioma | 67 |
| Pituitary | 68 |
| **Total** | **270** |

---

# Experimental Configuration

| Parameter | Configuration |
|---|---|
| Input size | `224 × 224` |
| MRI sequences | T1, T1CE, T2, FLAIR |
| Patch sizes | `8 × 8`, `16 × 16`, `32 × 32` |
| CNN stem | 2 convolutional layers |
| CNN channels | `64 → 128` |
| Embedding dimension | `256` |
| HCSAM blocks | `6` |
| Attention heads | `8` |
| Transformer layers | `6` |
| Dropout | `0.1` |
| Activation | GELU |
| Label smoothing | `0.1` |
| Optimizer | AdamW |
| Learning rate | `1 × 10⁻⁴` |
| Weight decay | `1 × 10⁻⁴` |
| AdamW β values | `(0.9, 0.999)` |
| Batch size | `32` |
| Maximum epochs | `100` |
| Scheduler | Cosine Annealing |
| Loss | Label-smoothed cross-entropy |
| GPU | NVIDIA RTX 4090, 24 GB |
| Repeated runs | `5` |
| Random seeds | `42, 52, 62, 72, 82` |

---

# Repository Structure

```text
CSA-ViT/
│
├── README.md
├── requirements.txt
├── train.py
├── test.py
├── evaluate.py
│
├── configs/
│   └── csavit_config.yaml
│
├── models/
│   ├── __init__.py
│   ├── cnn_stem.py
│   ├── multiscale_patch_embedding.py
│   ├── hcsam.py
│   ├── sequence_attention.py
│   ├── transformer_encoder.py
│   ├── attention_pooling.py
│   └── csavit.py
│
├── data/
│   ├── __init__.py
│   ├── dataset.py
│   └── preprocessing.py
│
├── utils/
│   ├── __init__.py
│   ├── metrics.py
│   ├── uncertainty.py
│   └── visualization.py
│
└── experiments/
    ├── __init__.py
    ├── baseline_vit.py
    ├── ablation.py
    └── repeated_runs.py
```

---

# Installation

Clone the repository:

```bash
git clone <REPOSITORY_URL>
cd CSA-ViT
```

Create a virtual environment:

```bash
python -m venv csavit-env
```

### Windows

```bash
csavit-env\Scripts\activate
```

### Linux/macOS

```bash
source csavit-env/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Training

Training requires patient-level training and validation CSV files.

Example:

```bash
python train.py \
    --data_dir data/images \
    --train_csv data/train.csv \
    --val_csv data/val.csv \
    --batch_size 32 \
    --epochs 100 \
    --learning_rate 1e-4 \
    --weight_decay 1e-4 \
    --label_smoothing 0.1 \
    --output_dir results/training
```

The best model is saved as:

```text
results/training/checkpoints/csavit_best.pth
```

The final model is saved as:

```text
results/training/checkpoints/csavit_final.pth
```

Training curves are saved under:

```text
results/training/figures/
```

---

# Testing

The trained model can be evaluated on the held-out test set using:

```bash
python test.py \
    --checkpoint results/training/checkpoints/csavit_best.pth \
    --data_dir data/images \
    --csv_file data/test.csv \
    --batch_size 32 \
    --output_dir results/test
```

The test procedure reports:

- Accuracy
- Precision
- Recall
- F1-score
- Macro one-vs-rest AUC
- Confusion matrix
- Class-wise performance

---

# General Evaluation

The general evaluation script can be executed using:

```bash
python evaluate.py \
    --checkpoint results/training/checkpoints/csavit_best.pth \
    --data_dir data/images \
    --csv_file data/test.csv \
    --batch_size 32 \
    --output_dir results/evaluation
```

Expected outputs include:

```text
results/evaluation/
├── classification_report.csv
├── predictions_uncertainty.csv
├── confusion_matrix.png
├── confusion_matrix_normalized.png
└── uncertainty.png
```

---

# Evaluation Metrics

The implementation calculates:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1-score
- Macro one-vs-rest AUC

For the four-class classification task, AUC is calculated using a **one-vs-rest strategy** and macro-averaged across the four diagnostic categories.

---

# Repeated Runs

The stability of CSA-ViT is evaluated using multiple random initialization seeds.

The experimental seeds are:

```text
42
52
62
72
82
```

The repeated-run implementation is located at:

```text
experiments/repeated_runs.py
```

Results are summarized using:

```text
Mean ± Standard Deviation
```

The reported five-run performance of CSA-ViT is:

| Metric | Mean ± SD |
|---|---:|
| Accuracy | `99.42 ± 0.18%` |
| Precision | `99.41 ± 0.19%` |
| Recall | `99.40 ± 0.20%` |
| F1-score | `99.41 ± 0.19%` |
| AUC | `99.82 ± 0.18%` |

These values correspond to the repeated experimental evaluation described in the associated manuscript.

---

# Held-Out Internal Test Evaluation

In addition to the repeated-run performance analysis, the manuscript reports class-wise performance on a fixed **270-sample internal held-out test set**.

The test set contains:

- 67 No-tumor
- 68 Glioma
- 67 Meningioma
- 68 Pituitary

The reported overall held-out test accuracy is:

```text
97.78%
```

Class-wise performance is reported separately in the associated manuscript.

The repeated-run results and the held-out class-wise evaluation serve different purposes and should not be interpreted as identical evaluation protocols.

---

# Calibration and Uncertainty

The held-out test predictions are additionally evaluated using:

| Metric | Value |
|---|---:|
| Expected Calibration Error (ECE) | `0.009` |
| Negative Log-Likelihood (NLL) | `0.021` |
| Brier Score | `0.008` |
| Normalized Predictive Entropy | `0.038` |

Predictive uncertainty is estimated from the class-probability distribution using normalized entropy.

The repository provides utilities for generating uncertainty visualizations and storing prediction-level uncertainty estimates.

---

# External Cross-Dataset Evaluation

The associated manuscript additionally evaluates CSA-ViT on an external **Kaggle Brain Tumor MRI Dataset containing 2,406 images**.

No fine-tuning is performed on the external dataset.

Because the external images are single 2-D MRI images and do not provide matched T1, T1CE, T2, and FLAIR sequences, the available image is replicated across the four modality branches to enable evaluation using the trained four-sequence architecture.

The reported external cross-dataset performance is:

| Metric | External Performance |
|---|---:|
| Macro Precision | `87.42%` |
| Macro Recall | `87.00%` |
| Macro F1-score | `87.18%` |

This experiment is intended as a **cross-dataset robustness assessment** rather than matched multi-sequence external validation.

---

# Ablation Study

The contribution of the major CSA-ViT components can be evaluated using:

```text
experiments/ablation.py
```

The ablation sequence is:

```text
Baseline Vision Transformer
          ↓
+ Multi-Scale Patch Embedding
          ↓
+ HCSAM
          ↓
+ SAM
          ↓
Complete CSA-ViT
```

The reported five-run ablation results are:

| Model Configuration | Accuracy (%) |
|---|---:|
| Vision Transformer | `98.34 ± 0.26` |
| + Multi-Scale Patch Embedding | `98.56 ± 0.28` |
| + HCSAM | `98.94 ± 0.33` |
| + SAM | `99.16 ± 0.27` |
| Complete CSA-ViT | `99.42 ± 0.18` |

These experiments quantify the incremental contribution of hierarchical multi-scale representation learning, cross-scale attention, and adaptive sequence fusion.

---

# Baseline Comparison

The repository provides an experimental framework for comparison with representative CNN- and Transformer-based architectures, including:

- VGG19
- ResNet50
- EfficientNetV2
- Vision Transformer
- Swin Transformer
- CSA-ViT

The corresponding experimental code is located in:

```text
experiments/baseline_vit.py
```

---

# Visualization

The repository provides visualization utilities for:

### Training Convergence

Training and validation behavior can be visualized using:

- Accuracy
- Loss
- Precision
- Recall
- F1-score

### Confusion Matrix

Confusion matrices can be generated for classification analysis, including normalized and unnormalized versions.

### HCSAM Attention Visualization

Attention visualization utilities are provided for examining the behavior of the hierarchical cross-scale attention mechanism.

### Predictive Uncertainty

Uncertainty visualizations can be generated using the predicted class-probability distributions and normalized predictive entropy.

The visualization utilities are available in:

```text
utils/visualization.py
```

---

# Computational Complexity

The reported computational characteristics of CSA-ViT are:

| Measure | Value |
|---|---:|
| Trainable parameters | `41.3M` |
| FLOPs | `12.8G` |
| Forward-pass time | `7.4 ms/image` |
| Input resolution | `224 × 224` |
| Batch size | `32` |
| Precision | FP32 |
| GPU | NVIDIA RTX 4090 |

The reported forward-pass time corresponds to model inference under the specified experimental configuration. Data loading and preprocessing are excluded from the reported forward-pass measurement.

---

# Reproducibility

The repository specifies the principal model and training parameters required to reproduce the proposed methodology, including:

- Input resolution
- MRI sequences
- Patch sizes
- Embedding dimension
- Attention heads
- HCSAM depth
- Transformer depth
- Dropout
- Optimizer
- Learning rate
- Weight decay
- Label smoothing
- Batch size
- Scheduler
- Random seeds

The principal implementation components are organized into:

```text
models/
data/
utils/
experiments/
configs/
```

Exact reproduction of the clinical experiments requires authorized access to the corresponding clinical MRI dataset.

---

# Computational Environment

The experiments described in the associated manuscript were performed using:

```text
GPU:
NVIDIA RTX 4090
24 GB VRAM

Framework:
PyTorch 2.1.0

Input:
224 × 224

Batch Size:
32

Precision:
FP32
```

A CUDA-enabled NVIDIA GPU is recommended for full training.

---

# Reported Results

The associated manuscript reports the following five-run performance for CSA-ViT:

| Metric | Performance |
|---|---:|
| Accuracy | `99.42 ± 0.18%` |
| Precision | `99.41 ± 0.19%` |
| Recall | `99.40 ± 0.20%` |
| F1-score | `99.41 ± 0.19%` |
| AUC | `99.82 ± 0.18%` |

The manuscript additionally reports:

- Five-run performance analysis
- CNN and Transformer baseline comparisons
- Component-wise ablation experiments
- Class-wise held-out test evaluation
- Cross-dataset robustness evaluation
- HCSAM attention visualization
- Predictive uncertainty analysis
- Calibration analysis
- Computational complexity analysis

These values are **reported experimental results from the associated manuscript** and correspond to the experimental protocol and dataset described there. They should not be interpreted as guaranteed results on different datasets, data splits, preprocessing pipelines, or hardware configurations.

---

# Ethical and Data-Use Statement

The clinical MRI data used in this research were obtained from **Adichunchanagiri Research Centre** with institutional permission for research use.

The clinical dataset is not distributed through this repository.

Users working with clinical MRI data are responsible for obtaining the necessary institutional authorization and complying with applicable ethical, privacy, and data-use requirements.

Patient-identifying information must not be included in the repository.

---

# Limitations

The current study has several limitations.

First, the evaluation is based on an available clinical dataset containing four diagnostic categories. Independent external validation using larger and more diverse cohorts is required to further assess generalization across institutions and imaging protocols.

Second, the current framework operates on selected 2-D MRI images rather than complete 3-D volumetric examinations.

Third, the uncertainty analysis provides sample-level predictive uncertainty derived from the classification probability distribution and does not constitute a spatially resolved uncertainty estimate.

Fourth, the external cross-dataset evaluation uses single 2-D images rather than matched four-sequence MRI examinations and therefore represents a robustness assessment rather than a direct matched-modality external validation.

Future work will investigate:

- Cross-institutional validation
- 3-D volumetric MRI analysis
- Domain adaptation
- Missing-modality robustness
- Computational optimization

---

# Citation

If you use CSA-ViT or this implementation in academic work, please cite the associated publication.

```bibtex
@article{csavit2026,
  title   = {CSA-ViT: Cross-Scale Attention Vision Transformer
             for Multi-Class Brain Tumor Classification Using
             Multi-Sequence MRI},
  author  = {Indrakumar, K},
  year    = {2026}
}
```

---

# License

The source code is provided for research and academic purposes.

The clinical MRI data are **not included in this repository** and remain subject to the permissions and restrictions of the data provider.

Before public redistribution of the source code, add the selected open-source license as a `LICENSE` file in the repository and ensure that its terms are consistent with the intended use of the code.

---

# Contact

For questions regarding the CSA-ViT implementation or research methodology, please refer to the corresponding author information in the associated publication.

---

# Acknowledgements

The authors acknowledge the institutional support and clinical research resources that enabled the collection and analysis of the MRI data used in this study.
