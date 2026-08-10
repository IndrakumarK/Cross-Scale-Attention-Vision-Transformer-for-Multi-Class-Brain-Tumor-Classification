# ============================================================
# CSA-ViT Configuration File
# Cross-Scale Attention Vision Transformer
# Multi-Class Brain Tumor Classification Using Multi-Sequence MRI
# ============================================================

# ------------------------------------------------------------
# 1. REPRODUCIBILITY
# ------------------------------------------------------------

seed: 42


# ------------------------------------------------------------
# 2. DATASET CONFIGURATION
# ------------------------------------------------------------

data:

  # Input MRI image size
  image_size: 224

  # Number of diagnostic categories
  num_classes: 4

  # MRI sequences used by the proposed framework
  modalities:
    - T1
    - T1CE
    - T2
    - FLAIR

  # Diagnostic categories
  class_names:
    - healthy
    - glioma
    - meningioma
    - pituitary

  # ----------------------------------------------------------
  # Dataset information
  # ----------------------------------------------------------

  # Clinically collected dataset
  number_of_patients: 250

  # Total selected MRI samples
  number_of_samples: 1800

  # Samples per diagnostic category
  samples_per_class: 450

  # ----------------------------------------------------------
  # Patient-level data partitioning
  # ----------------------------------------------------------
  #
  # The manuscript states that partitioning was performed at
  # the patient level to prevent information leakage.
  #
  # No numerical train/validation/test ratio is specified
  # in the manuscript, therefore no ratio is assumed here.
  #

  split_strategy: patient_level

  patient_column: patient_id

  split_column: split

  allowed_splits:
    - train
    - val
    - test

  # ----------------------------------------------------------
  # DataLoader
  # ----------------------------------------------------------

  batch_size: 32

  num_workers: 4

  shuffle_training: true

  pin_memory: true


# ------------------------------------------------------------
# 3. MRI PREPROCESSING
# ------------------------------------------------------------

preprocessing:

  # Bias-field correction
  bias_field_correction: true

  # Skull stripping
  skull_stripping: true

  # Intensity standardization
  intensity_standardization: true

  # ----------------------------------------------------------
  # Intensity normalization
  #
  # I_hat(p) =
  # [I(p) - E(I)] / sqrt(Var(I) + epsilon)
  # ----------------------------------------------------------

  normalization:

    method: z_score

    epsilon: 1.0e-6

  # ----------------------------------------------------------
  # Image resizing
  # ----------------------------------------------------------

  resize:

    height: 224

    width: 224

  # ----------------------------------------------------------
  # MRI registration
  #
  # The supplied manuscript does not explicitly state that
  # registration was performed.
  # ----------------------------------------------------------

  registration: false

  # ----------------------------------------------------------
  # Data augmentation
  #
  # The supplied manuscript does not specify an augmentation
  # pipeline, so it is not claimed here.
  # ----------------------------------------------------------

  augmentation:

    enabled: false


# ------------------------------------------------------------
# 4. CNN STEM
# ------------------------------------------------------------

cnn_stem:

  # Number of convolutional layers
  num_layers: 2

  # Convolution kernel
  kernel_size: 3

  # Stride
  stride: 1

  # Padding
  padding: 1

  # Channel progression
  channels:
    - 64
    - 128

  # Normalization
  normalization: batch_norm

  # Activation
  activation: GELU

  # Spatial resolution is preserved
  preserve_spatial_resolution: true


# ------------------------------------------------------------
# 5. HIERARCHICAL MULTI-SCALE PATCH EMBEDDING
# ------------------------------------------------------------

patch_embedding:

  # Three patch resolutions
  patch_sizes:
    - 8
    - 16
    - 32

  # Common token embedding dimension
  embedding_dim: 256

  # Positional encoding
  positional_encoding: true

  # ----------------------------------------------------------
  # Token counts for 224 × 224 input
  # ----------------------------------------------------------

  token_counts:

    patch_8x8: 784

    patch_16x16: 196

    patch_32x32: 49

  # ----------------------------------------------------------
  # Corresponding token grids
  # ----------------------------------------------------------

  token_grids:

    patch_8x8:
      - 28
      - 28

    patch_16x16:
      - 14
      - 14

    patch_32x32:
      - 7
      - 7


# ------------------------------------------------------------
# 6. HIERARCHICAL CROSS-SCALE ATTENTION MODULE (HCSAM)
# ------------------------------------------------------------

hcsam:

  # Number of HCSAM blocks
  num_blocks: 6

  # Embedding dimension
  embedding_dim: 256

  # Number of attention heads
  num_heads: 8

  # Dimension per attention head
  head_dimension: 32

  # ----------------------------------------------------------
  # Adjacent-scale cross-attention
  # ----------------------------------------------------------

  adjacent_scale_attention: true

  # ----------------------------------------------------------
  # Small -> Medium
  #
  # Es supplies Query
  # Em supplies Key and Value
  # ----------------------------------------------------------

  small_to_medium:

    enabled: true

    query_source: small

    key_source: medium

    value_source: medium

  # ----------------------------------------------------------
  # Medium -> Large
  #
  # Em supplies Query
  # El supplies Key and Value
  # ----------------------------------------------------------

  medium_to_large:

    enabled: true

    query_source: medium

    key_source: large

    value_source: large

  # ----------------------------------------------------------
  # Preserve query resolution
  # ----------------------------------------------------------

  preserve_query_resolution: true

  # Residual connection
  residual_connection: true

  # Layer normalization
  layer_normalization: true

  # Learnable residual scaling parameter gamma
  learnable_gamma: true

  # Attention dropout
  dropout: 0.1

  # Output projection
  output_projection: true


# ------------------------------------------------------------
# 7. SEQUENCE ATTENTION MODULE (SAM)
# ------------------------------------------------------------

sam:

  enabled: true

  # Number of MRI sequences
  num_modalities: 4

  # MRI sequences
  modalities:
    - T1
    - T1CE
    - T2
    - FLAIR

  # ----------------------------------------------------------
  # Common feature dimension
  #
  # All modality-specific representations are projected to
  # 256 dimensions before weighted aggregation.
  # ----------------------------------------------------------

  common_embedding_dim: 256

  # Output dimension
  output_dim: 256

  # ----------------------------------------------------------
  # Attention scoring
  #
  # u_i = w^T tanh(WF_i + b)
  # ----------------------------------------------------------

  scoring:

    activation: tanh

    normalization: softmax

  # Adaptive weighted fusion
  weighted_fusion: true

  # Modality weights are learned dynamically
  dynamic_modality_weighting: true


# ------------------------------------------------------------
# 8. TRANSFORMER ENCODER
# ------------------------------------------------------------

transformer:

  # Number of encoder layers
  num_layers: 6

  # Feature embedding dimension
  embedding_dim: 256

  # Number of attention heads
  num_heads: 8

  # Dimension per attention head
  head_dimension: 32

  # Transformer feed-forward dimension
  feedforward_dim: 1024

  # Dropout
  dropout: 0.1

  # Activation
  activation: GELU

  # Layer normalization
  layer_normalization: true

  # Pre-normalization
  norm_first: true

  # Batch-first tensor representation
  batch_first: true


# ------------------------------------------------------------
# 9. ATTENTION-BASED FEATURE AGGREGATION
# ------------------------------------------------------------

attention_pooling:

  enabled: true

  # Token relevance scoring
  scoring: linear

  # Token embedding dimension
  embedding_dim: 256

  # Normalize token importance scores
  normalization: softmax

  # Weighted token aggregation
  weighted_pooling: true


# ------------------------------------------------------------
# 10. CLASSIFICATION HEAD
# ------------------------------------------------------------

classifier:

  # Four-class classification
  num_classes: 4

  # Input feature dimension
  input_dim: 256

  # Hidden layer dimension
  hidden_dim: 256

  # Activation
  activation: GELU

  # Dropout
  dropout: 0.1

  # Final activation
  output_activation: softmax

  # Diagnostic categories
  classes:
    - healthy
    - glioma
    - meningioma
    - pituitary


# ------------------------------------------------------------
# 11. OPTIMIZATION
# ------------------------------------------------------------

training:

  # Maximum number of epochs
  epochs: 100

  # Optimizer
  optimizer: AdamW

  # Initial learning rate
  learning_rate: 1.0e-4

  # AdamW beta parameters
  betas:

    beta1: 0.9

    beta2: 0.999

  # Weight decay
  weight_decay: 1.0e-4

  # Label smoothing
  label_smoothing: 0.1

  # Early stopping
  early_stopping: true

  early_stopping_patience: 15

  # Checkpoint directory
  checkpoint_dir: checkpoints


# ------------------------------------------------------------
# 12. LEARNING RATE SCHEDULER
# ------------------------------------------------------------

scheduler:

  # Cosine annealing
  name: cosine_annealing

  # Number of epochs
  T_max: 100

  # Minimum learning rate
  eta_min: 1.0e-6


# ------------------------------------------------------------
# 13. LOSS FUNCTION
# ------------------------------------------------------------

loss:

  name: label_smoothed_cross_entropy

  # Four diagnostic classes
  num_classes: 4

  # Label smoothing coefficient
  smoothing: 0.1


# ------------------------------------------------------------
# 14. UNCERTAINTY ANALYSIS
# ------------------------------------------------------------

uncertainty:

  enabled: true

  # Normalized predictive entropy
  method: normalized_entropy

  # Number of classes
  num_classes: 4

  # Numerical stability
  epsilon: 1.0e-8

  # ----------------------------------------------------------
  # Important:
  # This produces a sample-level uncertainty value from the
  # class-probability distribution. It should not be described
  # as a spatial uncertainty map unless a separate spatial
  # mapping procedure is actually implemented.
  # ----------------------------------------------------------

  spatial_mapping: false


# ------------------------------------------------------------
# 15. EVALUATION
# ------------------------------------------------------------

evaluation:

  # Multi-class evaluation
  num_classes: 4

  # ----------------------------------------------------------
  # Metrics
  # ----------------------------------------------------------

  metrics:

    - accuracy
    - precision
    - recall
    - f1
    - auc

  # ----------------------------------------------------------
  # AUC calculation
  # ----------------------------------------------------------

  auc:

    enabled: true

    strategy: one_vs_rest

    averaging: macro

  # Confusion matrix
  confusion_matrix: true

  # Classification report
  classification_report: true


# ------------------------------------------------------------
# 16. REPEATED EXPERIMENTS
# ------------------------------------------------------------

repeated_runs:

  enabled: true

  number_of_runs: 5

  # Random initialization seeds
  seeds:
    - 42
    - 52
    - 62
    - 72
    - 82

  # Report mean and standard deviation
  reporting:

    mean: true

    standard_deviation: true


# ------------------------------------------------------------
# 17. ATTENTION VISUALIZATION
# ------------------------------------------------------------

visualization:

  # HCSAM cross-scale attention
  cross_scale_attention: true

  # Token importance
  token_attention: true

  # MRI sequence attention weights
  modality_attention: true

  # Failure-case attention visualization
  failure_cases: true

  # Predictive uncertainty visualization
  uncertainty: true

  # Output directory
  output_directory: results/figures


# ------------------------------------------------------------
# 18. COMPUTATIONAL COMPLEXITY
# ------------------------------------------------------------

complexity:

  # Input used for computational measurements
  input_height: 224

  input_width: 224

  # Hardware
  gpu: NVIDIA RTX 4090

  gpu_memory_gb: 24

  # GPU warm-up before inference timing
  warmup: true

  # Data loading excluded from timing
  exclude_data_loading: true

  # Preprocessing excluded from timing
  exclude_preprocessing: true


# ------------------------------------------------------------
# 19. OUTPUT DIRECTORIES
# ------------------------------------------------------------

output:

  checkpoint_directory: checkpoints

  figure_directory: results/figures

  table_directory: results/tables

  log_directory: results/logs


# ------------------------------------------------------------
# 20. REPOSITORY DATA MANIFEST
# ------------------------------------------------------------
#
# Expected CSV columns:
#
# path,label,patient_id,split
#
# Example:
#
# data/patient001.npz,0,patient001,train
# data/patient002.npz,1,patient002,val
# data/patient003.npz,2,patient003,test
#
# The split column must be assigned at the PATIENT level.
# The same patient must not appear in more than one split.
#
# Labels:
# 0 = Healthy
# 1 = Glioma
# 2 = Meningioma
# 3 = Pituitary
#
# Each NPZ sample should contain:
# T1, T1CE, T2, FLAIR
#
# ============================================================