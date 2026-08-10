
# CSA-ViT: Cross-Scale Attention Vision Transformer for Multi-Class Brain Tumor Classification Using Multi-Sequence MRI

Research implementation accompanying the **CSA-ViT (Cross-Scale Attention Vision Transformer)** for multi-class brain tumor classification using multi-sequence magnetic resonance imaging (MRI).

The proposed framework integrates hierarchical multi-scale feature learning, cross-scale attention, adaptive MRI sequence fusion, and Transformer-based global contextual modeling.

---

## Overview

Accurate brain tumor classification from MRI is challenging because tumors exhibit variations in size, morphology, texture, and spatial appearance. In addition, different MRI sequences provide complementary information about tumor and surrounding tissues.

CSA-ViT addresses these challenges through the following components:

- Lightweight CNN stem for local feature extraction
- Hierarchical multi-scale patch embedding
- Hierarchical Cross-Scale Attention Module (HCSAM)
- Adaptive Sequence Attention Module (SAM)
- Transformer-based global contextual modeling
- Attention-based feature aggregation
- Confidence-aware uncertainty analysis

The proposed framework is designed to jointly exploit spatial information at multiple resolutions and complementary information across T1, T1CE, T2, and FLAIR MRI sequences.

---

## Architecture

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
                  ┌────────────────┼────────────────┐
                  │                │                │
              8 × 8 patches    16 × 16 patches   32 × 32 patches
                  │                │                │
             784 tokens        196 tokens         49 tokens
                  │                │                │
                  └────────────────┼────────────────┘
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
          ┌────────────┬───────────┼───────────┬─────────────┐
          │            │           │           │
       Healthy       Glioma    Meningioma   Pituitary

````

---

## Key Components

### 1. MRI Preprocessing

The preprocessing pipeline consists of:

1. Bias-field correction
2. Skull stripping
3. Intensity standardization
4. Resizing to `224 × 224` pixels

Intensity standardization is performed using:

$$
\hat{I}(p)=
\frac{I(p)-E[I]}
{\sqrt{\operatorname{Var}(I)+\epsilon}}
$$

---

### 2. Lightweight CNN Stem

The CNN stem consists of two consecutive convolutional layers with:

- Kernel size: `3 × 3`
- Stride: `1`
- Padding preserving spatial resolution
- Batch Normalization
- GELU activation

The CNN stem extracts local structural information before multi-scale token generation.

---

### 3. Hierarchical Multi-Scale Patch Embedding

Three patch resolutions are used:

| Patch SizeToken GridNumber of Tokens |           |       |
| ------------------------------------ | --------- | ----- |
| `8 × 8`                              | `28 × 28` | `784` |
| `16 × 16`                            | `14 × 14` | `196` |
| `32 × 32`                            | `7 × 7`   | `49`  |

The three resolutions capture fine-grained local characteristics, intermediate regional structures, and broader anatomical context.

---

### 4. Hierarchical Cross-Scale Attention Module (HCSAM)

HCSAM explicitly performs cross-attention between adjacent resolutions:

```
Fine-scale representation
          Es
          │
          ↓
     Cross-Attention
          │
          ↓
Intermediate representation
          Em
          │
          ↓
     Cross-Attention
          │
          ↓
Coarse-scale representation
          El

```

For the `Es → Em` interaction:

```
Query  = Es
Key    = Em
Value  = Em

```

The query resolution is preserved during cross-scale interaction.

This enables fine-grained features to incorporate complementary regional and contextual information.

---

### 5. Adaptive Sequence Attention Module (SAM)

The four MRI sequences are processed to obtain modality-specific representations:

```
T1       ─┐
T1CE     ─┤
T2       ─┤ → Common 256-dimensional embedding
FLAIR    ─┘

```

Before adaptive fusion, the modality-specific representations are projected to a common embedding dimension of `256`.

Learned attention coefficients are then used for weighted aggregation:

$$
F_{SAM} = \sum_{i=1}^{M} \beta_i F_i
$$

This allows the network to adaptively emphasize diagnostically informative MRI sequences.

---

### 6. Transformer Encoder

The fused representation is processed using stacked Transformer encoder layers.

| ParameterConfiguration |      |
| ---------------------- | ---- |
| Embedding dimension    | 256  |
| Attention heads        | 8    |
| Transformer layers     | 6    |
| MLP ratio              | 4    |
| Dropout                | 0.1  |
| Activation             | GELU |

The Transformer encoder models long-range contextual relationships among feature tokens.

---

### 7. Attention-Based Feature Aggregation

Instead of conventional average pooling, learned token importance scores are used to obtain the global representation.

The resulting representation is passed through a multilayer perceptron classification head.

The classifier predicts one of four classes:

**Healthy, Glioma, Meningioma, and Pituitary Tumor.**

---

### 8. Confidence-Aware Uncertainty

Predictive uncertainty is calculated from the class-probability distribution using normalized entropy:

$$
U(x)=
-\frac{1}{\log C}
\sum_{c=1}^{C}
p_c(x)\log(p_c(x)+\epsilon)
$$

where (C=4).

Lower entropy indicates greater confidence, whereas higher entropy indicates greater ambiguity in the predicted class distribution.

**Important:** The implementation provides **sample-level predictive uncertainty**. It does not interpret classification entropy as a spatial uncertainty map.

---

# Dataset

The study uses a clinically collected multi-sequence MRI dataset comprising examinations from **250 patients**.

The four diagnostic categories are:

- Healthy
- Glioma
- Meningioma
- Pituitary tumor

The four-class classification experiments use **1,800 selected MRI samples**, with **450 samples per diagnostic category**.

The samples represent selected MRI images obtained from patient examinations and do not represent 1,800 independent patients.

The clinical MRI dataset was obtained from **Adichunchanagiri Research Centre** with institutional permission for research use.

## Data Availability

The clinical MRI dataset is **not included in this repository** because it was obtained under institutional permission.

Researchers wishing to reproduce the experiments must obtain appropriate authorization and comply with applicable ethical and data-use requirements.

No patient-identifying information should be uploaded to this repository.

---

# Data Organization

The implementation uses CSV files to define the dataset splits.

The dataset should be organized according to the following structure expected by the implementation:

```
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

```
data/dataset.py

```

Patient identifiers should be retained in the dataset metadata to ensure patient-level partitioning.

---

# Patient-Level Data Partitioning

Data partitioning is performed at the **patient level** to prevent information leakage.

Images originating from the same patient must remain within the same subset.

```
250 Patients
     │
     ↓
Patient-Level Partitioning
     │
 ┌───┼────┐
 ↓   ↓    ↓
Train Val Test

```

The exact train/validation/test ratio should correspond to the experimental split used in the associated manuscript.

---

# Experimental Configuration

| ParameterConfiguration |                               |
| ---------------------- | ----------------------------- |
| Input size             | `224 × 224`                   |
| MRI sequences          | T1, T1CE, T2, FLAIR           |
| Patch sizes            | `8 × 8`, `16 × 16`, `32 × 32` |
| CNN stem               | 2 convolutional layers        |
| CNN channels           | `64 → 128`                    |
| Embedding dimension    | `256`                         |
| HCSAM blocks           | `6`                           |
| Attention heads        | `8`                           |
| Transformer layers     | `6`                           |
| Dropout                | `0.1`                         |
| Activation             | GELU                          |
| Label smoothing        | `0.1`                         |
| Optimizer              | AdamW                         |
| Learning rate          | `1 × 10⁻⁴`                    |
| Weight decay           | `1 × 10⁻⁴`                    |
| AdamW β values         | `(0.9, 0.999)`                |
| Batch size             | `32`                          |
| Maximum epochs         | `100`                         |
| Scheduler              | Cosine Annealing              |
| Loss                   | Label-smoothed cross-entropy  |
| GPU                    | NVIDIA RTX 4090, 24 GB        |
| Repeated runs          | `5`                           |

---

# Repository Structure

```
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
├── experiments/
│   ├── __init__.py
│   ├── baseline_vit.py
│   ├── ablation.py
│   └── repeated_runs.py
│
└── results/
    ├── figures/
    └── tables/

```

---

# Installation

Clone the repository:

```
git clone <REPOSITORY_URL>
cd CSA-ViT

```

Create a virtual environment:

```
python -m venv csavit-env

```

### Windows

```
csavit-env\Scripts\activate

```

### Linux/macOS

```
source csavit-env/bin/activate

```

Install dependencies:

```
pip install -r requirements.txt

```

---

# Training

Training requires patient-level training and validation CSV files.

```
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

```
results/training/checkpoints/csavit_best.pth

```

The final model is saved as:

```
results/training/checkpoints/csavit_final.pth

```

Training curves are saved under:

```
results/training/figures/

```

---

# Testing

The trained model can be evaluated on the held-out test set using:

```
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

```
python evaluate.py \
    --checkpoint results/training/checkpoints/csavit_best.pth \
    --data_dir data/images \
    --csv_file data/test.csv \
    --batch_size 32 \
    --output_dir results/evaluation

```

Expected outputs include:

```
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

The stability of CSA-ViT can be evaluated using multiple random initialization seeds.

The experimental seeds used in the study are:

```
42
52
62
72
82

```

The repeated-run implementation is located at:

```
experiments/repeated_runs.py

```

Results can be summarized as:

```
Mean ± Standard Deviation

```

---

# Ablation Study

The contribution of the major CSA-ViT components can be evaluated using:

```
experiments/ablation.py

```

The intended sequence is:

```
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

```
experiments/baseline_vit.py

```

---

# Visualization

The repository provides visualization utilities for:

### Training convergence

```
Figure 4(a-e)

```

including:

- Accuracy
- Loss
- Precision
- Recall
- F1-score

### Confusion matrix

```
Figure 5

```

### HCSAM attention visualization

```
Figures 6-7

```

### Predictive uncertainty

```
Figure 8

```

The visualization utilities are available in:

```
utils/visualization.py

```

---

# Reproducibility

The repository specifies the principal model and training parameters required to reproduce the proposed methodology, including:

- input resolution,
- MRI sequences,
- patch sizes,
- embedding dimension,
- attention heads,
- HCSAM depth,
- Transformer depth,
- dropout,
- optimizer,
- learning rate,
- weight decay,
- label smoothing,
- batch size,
- scheduler, and
- random seeds.

Exact reproduction of the clinical experiments requires authorized access to the corresponding clinical MRI dataset.

---

# Computational Environment

The experiments described in the associated manuscript were performed using:

```
GPU:
NVIDIA RTX 4090
24 GB VRAM

Framework:
PyTorch 2.2

Input:
224 × 224

Batch Size:
32

```

A CUDA-enabled NVIDIA GPU is recommended for full training.

---

# Reported Results

The associated manuscript reports the following performance for CSA-ViT:

| MetricPerformance |        |
| ----------------- | ------ |
| Accuracy          | 99.42% |
| Precision         | 99.41% |
| Recall            | 99.40% |
| F1-score          | 99.41% |

The manuscript also reports:

- five-run performance analysis,
- CNN and Transformer baseline comparisons,
- ablation experiments,
- attention visualization,
- uncertainty analysis, and
- computational complexity analysis.

These values are **reported experimental results from the associated manuscript** and correspond to the experimental protocol and dataset described there. They should not be interpreted as guaranteed results on different datasets, data splits, preprocessing pipelines, or hardware configurations.

---

# Ethical and Data-Use Statement

The clinical MRI data used in this research were obtained from Adichunchanagiri Research Centre with institutional permission for research use.

The clinical dataset is not distributed through this repository.

Users working with clinical MRI data are responsible for obtaining the necessary institutional authorization and complying with applicable ethical, privacy, and data-use requirements.

Patient-identifying information must not be included in the repository.

---

# Limitations

The current study has several limitations.

First, the evaluation is based on the available clinical dataset and four diagnostic categories. Independent external validation using larger and diverse cohorts is required to assess generalization across institutions and imaging protocols.

Second, the current framework operates on selected 2D MRI images rather than complete 3D volumetric examinations.

Third, the uncertainty analysis provides sample-level predictive uncertainty derived from the classification probability distribution and does not constitute a spatially resolved uncertainty estimate.

Future work will investigate:

- cross-institutional validation,
- 3D volumetric MRI analysis,
- domain adaptation,
- missing-modality robustness, and
- computational optimization.

---

# Citation

If you use CSA-ViT or this implementation in academic work, please cite the associated publication.

```
@article{csavit2026,
  title   = {CSA-ViT: Cross-Scale Attention Vision Transformer
             for Multi-Class Brain Tumor Classification Using
             Multi-Sequence MRI},
  author  = {Indrakumar, K. and Ravikumar, M.},
  year    = {2026}
}

```

The citation should be updated with the final journal/conference bibliographic information after publication.

---

# License

The source code is provided for research and academic purposes.

The clinical MRI data are not included in this repository and remain subject to the permissions and restrictions of the data provider.

Before public redistribution of the source code, add the selected open-source license as a `LICENSE` file in the repository and ensure that its terms are consistent with the intended use of the code.

---

# Contact

For questions regarding the CSA-ViT implementation or research methodology, please refer to the corresponding author information in the associated publication.

---

# Acknowledgements

The authors acknowledge the institutional support and clinical research resources that enabled the collection and analysis of the MRI data used in this study.

```
```
