"""
Visualization Utilities for CSA-ViT
====================================

Visualization functions used for:

1. Confusion matrix
2. Training/validation convergence curves
3. HCSAM attention maps
4. Sample-level predictive uncertainty

Important:
The uncertainty visualization represents SAMPLE-LEVEL entropy.
It does not create a spatial uncertainty map from classification
entropy.
"""

import os

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Class Names
# ============================================================

CLASS_NAMES = [
    "Healthy",
    "Glioma",
    "Meningioma",
    "Pituitary Tumor"
]


# ============================================================
# Create Output Directory
# ============================================================

def _create_output_directory(
    save_path
):
    """
    Create parent directory if necessary.
    """

    if save_path is None:
        return

    directory = os.path.dirname(
        save_path
    )

    if directory:
        os.makedirs(
            directory,
            exist_ok=True
        )


# ============================================================
# Confusion Matrix
# ============================================================

def plot_confusion_matrix(
    cm,
    class_names=None,
    normalize=False,
    save_path=None,
    title="CSA-ViT Confusion Matrix",
    figsize=(7, 6)
):
    """
    Plot a confusion matrix.

    Parameters
    ----------
    cm : array-like
        Confusion matrix.

    class_names : list, optional
        Class names.

    normalize : bool
        If True, normalize each row.

    save_path : str, optional
        Output file path.

    title : str
        Figure title.

    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """

    cm = np.asarray(
        cm,
        dtype=float
    )

    if class_names is None:

        class_names = CLASS_NAMES

    if normalize:

        row_sums = cm.sum(
            axis=1,
            keepdims=True
        )

        cm_display = np.divide(
            cm,
            row_sums,
            out=np.zeros_like(cm),
            where=row_sums != 0
        )

    else:

        cm_display = cm

    fig, ax = plt.subplots(
        figsize=figsize
    )

    image = ax.imshow(
        cm_display,
        interpolation="nearest"
    )

    ax.set_title(
        title
    )

    ax.set_xlabel(
        "Predicted Class"
    )

    ax.set_ylabel(
        "True Class"
    )

    ax.set_xticks(
        np.arange(
            len(class_names)
        )
    )

    ax.set_yticks(
        np.arange(
            len(class_names)
        )
    )

    ax.set_xticklabels(
        class_names,
        rotation=30,
        ha="right"
    )

    ax.set_yticklabels(
        class_names
    )

    # --------------------------------------------------------
    # Annotate cells
    # --------------------------------------------------------

    threshold = (
        cm_display.max() / 2.0
        if cm_display.size > 0
        else 0.0
    )

    for i in range(
        cm_display.shape[0]
    ):

        for j in range(
            cm_display.shape[1]
        ):

            if normalize:

                text = (
                    f"{cm_display[i, j]:.3f}"
                )

            else:

                text = (
                    f"{int(cm_display[i, j])}"
                )

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color=(
                    "white"
                    if cm_display[i, j] > threshold
                    else "black"
                )
            )

    fig.colorbar(
        image,
        ax=ax
    )

    fig.tight_layout()

    if save_path is not None:

        _create_output_directory(
            save_path
        )

        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    return fig


# ============================================================
# Training Curves
# ============================================================

def plot_training_curves(
    history,
    save_directory=None,
    show=False
):
    """
    Plot training and validation convergence curves.

    Expected history dictionary:

        {
            "train_loss": [...],
            "val_loss": [...],
            "train_accuracy": [...],
            "val_accuracy": [...],
            "train_precision": [...],
            "val_precision": [...],
            "train_recall": [...],
            "val_recall": [...],
            "train_f1": [...],
            "val_f1": [...]
        }

    The function creates five separate figures corresponding
    to Figure 4(a-e) in the manuscript.
    """

    if save_directory is not None:

        os.makedirs(
            save_directory,
            exist_ok=True
        )

    figures = {}

    # --------------------------------------------------------
    # Epochs
    # --------------------------------------------------------

    if "train_loss" in history:

        epochs = range(
            1,
            len(
                history["train_loss"]
            ) + 1
        )

    elif "train_accuracy" in history:

        epochs = range(
            1,
            len(
                history["train_accuracy"]
            ) + 1
        )

    else:

        raise ValueError(
            "Training history does not contain "
            "recognized metrics."
        )

    # ========================================================
    # Accuracy
    # ========================================================

    if (
        "train_accuracy" in history
        and
        "val_accuracy" in history
    ):

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        ax.plot(
            epochs,
            history["train_accuracy"],
            label="Training Accuracy"
        )

        ax.plot(
            epochs,
            history["val_accuracy"],
            label="Validation Accuracy"
        )

        ax.set_xlabel(
            "Epoch"
        )

        ax.set_ylabel(
            "Accuracy"
        )

        ax.set_title(
            "Training and Validation Accuracy"
        )

        ax.legend()

        ax.grid(
            True,
            alpha=0.3
        )

        fig.tight_layout()

        figures["accuracy"] = fig

        if save_directory:

            fig.savefig(
                os.path.join(
                    save_directory,
                    "accuracy_curve.png"
                ),
                dpi=300,
                bbox_inches="tight"
            )

    # ========================================================
    # Loss
    # ========================================================

    if (
        "train_loss" in history
        and
        "val_loss" in history
    ):

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        ax.plot(
            range(
                1,
                len(
                    history["train_loss"]
                ) + 1
            ),
            history["train_loss"],
            label="Training Loss"
        )

        ax.plot(
            range(
                1,
                len(
                    history["val_loss"]
                ) + 1
            ),
            history["val_loss"],
            label="Validation Loss"
        )

        ax.set_xlabel(
            "Epoch"
        )

        ax.set_ylabel(
            "Loss"
        )

        ax.set_title(
            "Training and Validation Loss"
        )

        ax.legend()

        ax.grid(
            True,
            alpha=0.3
        )

        fig.tight_layout()

        figures["loss"] = fig

        if save_directory:

            fig.savefig(
                os.path.join(
                    save_directory,
                    "loss_curve.png"
                ),
                dpi=300,
                bbox_inches="tight"
            )

    # ========================================================
    # Precision
    # ========================================================

    if (
        "train_precision" in history
        and
        "val_precision" in history
    ):

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        ax.plot(
            range(
                1,
                len(
                    history["train_precision"]
                ) + 1
            ),
            history["train_precision"],
            label="Training Precision"
        )

        ax.plot(
            range(
                1,
                len(
                    history["val_precision"]
                ) + 1
            ),
            history["val_precision"],
            label="Validation Precision"
        )

        ax.set_xlabel(
            "Epoch"
        )

        ax.set_ylabel(
            "Precision"
        )

        ax.set_title(
            "Training and Validation Precision"
        )

        ax.legend()

        ax.grid(
            True,
            alpha=0.3
        )

        fig.tight_layout()

        figures["precision"] = fig

        if save_directory:

            fig.savefig(
                os.path.join(
                    save_directory,
                    "precision_curve.png"
                ),
                dpi=300,
                bbox_inches="tight"
            )

    # ========================================================
    # Recall
    # ========================================================

    if (
        "train_recall" in history
        and
        "val_recall" in history
    ):

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        ax.plot(
            range(
                1,
                len(
                    history["train_recall"]
                ) + 1
            ),
            history["train_recall"],
            label="Training Recall"
        )

        ax.plot(
            range(
                1,
                len(
                    history["val_recall"]
                ) + 1
            ),
            history["val_recall"],
            label="Validation Recall"
        )

        ax.set_xlabel(
            "Epoch"
        )

        ax.set_ylabel(
            "Recall"
        )

        ax.set_title(
            "Training and Validation Recall"
        )

        ax.legend()

        ax.grid(
            True,
            alpha=0.3
        )

        fig.tight_layout()

        figures["recall"] = fig

        if save_directory:

            fig.savefig(
                os.path.join(
                    save_directory,
                    "recall_curve.png"
                ),
                dpi=300,
                bbox_inches="tight"
            )

    # ========================================================
    # F1-score
    # ========================================================

    if (
        "train_f1" in history
        and
        "val_f1" in history
    ):

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        ax.plot(
            range(
                1,
                len(
                    history["train_f1"]
                ) + 1
            ),
            history["train_f1"],
            label="Training F1-score"
        )

        ax.plot(
            range(
                1,
                len(
                    history["val_f1"]
                ) + 1
            ),
            history["val_f1"],
            label="Validation F1-score"
        )

        ax.set_xlabel(
            "Epoch"
        )

        ax.set_ylabel(
            "F1-score"
        )

        ax.set_title(
            "Training and Validation F1-score"
        )

        ax.legend()

        ax.grid(
            True,
            alpha=0.3
        )

        fig.tight_layout()

        figures["f1"] = fig

        if save_directory:

            fig.savefig(
                os.path.join(
                    save_directory,
                    "f1_curve.png"
                ),
                dpi=300,
                bbox_inches="tight"
            )

    if show:

        plt.show()

    return figures


# ============================================================
# HCSAM Attention Map
# ============================================================

def plot_attention_map(
    attention,
    query_size,
    key_size,
    query_index=0,
    head_index=0,
    save_path=None,
    title="HCSAM Cross-Scale Attention"
):
    """
    Visualize one HCSAM cross-attention map.

    Parameters
    ----------
    attention : torch.Tensor or numpy.ndarray

        Expected shape:

            [B, H, Nq, Nk]

        Example for Es -> Em:

            [B, 8, 784, 196]

    query_size : tuple

        Spatial grid of query tokens.

        For Es:

            (28, 28)

        For Em:

            (14, 14)

    key_size : tuple

        Spatial grid of key/value tokens.

        For Em:

            (14, 14)

        For El:

            (7, 7)

    query_index : int
        Query token to visualize.

    head_index : int
        Attention head.

    IMPORTANT:
    This function visualizes attention weights, not uncertainty.
    """

    if hasattr(
        attention,
        "detach"
    ):

        attention = (
            attention
            .detach()
            .cpu()
            .numpy()
        )

    attention = np.asarray(
        attention
    )

    if attention.ndim != 4:

        raise ValueError(
            "Attention must have shape "
            "[B, H, Nq, Nk]."
        )

    batch_index = 0

    if head_index >= attention.shape[1]:

        raise ValueError(
            "head_index exceeds the number "
            "of attention heads."
        )

    if query_index >= attention.shape[2]:

        raise ValueError(
            "query_index exceeds the number "
            "of query tokens."
        )

    # --------------------------------------------------------
    # Select one query attention vector
    # --------------------------------------------------------

    weights = attention[
        batch_index,
        head_index,
        query_index
    ]

    # --------------------------------------------------------
    # Reshape key tokens into their spatial grid
    # --------------------------------------------------------

    expected_key_tokens = (
        key_size[0] *
        key_size[1]
    )

    if len(weights) != expected_key_tokens:

        raise ValueError(
            "key_size does not match "
            "the number of key tokens."
        )

    attention_map = weights.reshape(
        key_size
    )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    image = ax.imshow(
        attention_map,
        interpolation="bilinear"
    )

    ax.set_title(
        title
    )

    ax.set_xlabel(
        "Key token position"
    )

    ax.set_ylabel(
        "Key token position"
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Attention Weight"
    )

    fig.tight_layout()

    if save_path is not None:

        _create_output_directory(
            save_path
        )

        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    return fig


# ============================================================
# Sample-Level Uncertainty Plot
# ============================================================

def plot_uncertainty(
    uncertainty,
    sample_indices=None,
    save_path=None,
    title="Predictive Uncertainty Analysis"
):
    """
    Plot sample-level normalized entropy.

    Parameters
    ----------
    uncertainty : array-like

        One uncertainty value per MRI sample.

        Values should normally be in [0, 1].

    sample_indices : array-like, optional

        Sample identifiers or indices.

    save_path : str, optional
        Output file path.

    title : str
        Figure title.

    IMPORTANT:
    This is a sample-level uncertainty plot. It is not a
    spatial uncertainty map.
    """

    uncertainty = np.asarray(
        uncertainty,
        dtype=float
    )

    if uncertainty.ndim != 1:

        raise ValueError(
            "Uncertainty must be a one-dimensional "
            "array containing one value per sample."
        )

    if sample_indices is None:

        sample_indices = np.arange(
            len(uncertainty)
        )

    if len(sample_indices) != len(
        uncertainty
    ):

        raise ValueError(
            "sample_indices and uncertainty "
            "must have the same length."
        )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.plot(
        sample_indices,
        uncertainty,
        marker="o",
        markersize=3,
        linewidth=1
    )

    ax.set_xlabel(
        "MRI Sample"
    )

    ax.set_ylabel(
        "Normalized Entropy"
    )

    ax.set_title(
        title
    )

    ax.set_ylim(
        0,
        1
    )

    ax.grid(
        True,
        alpha=0.3
    )

    fig.tight_layout()

    if save_path is not None:

        _create_output_directory(
            save_path
        )

        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    return fig


# ============================================================
# Sequence Attention Weights
# ============================================================

def plot_sequence_attention(
    weights,
    save_path=None,
    title="Adaptive MRI Sequence Attention"
):
    """
    Visualize learned sequence attention weights.

    Parameters
    ----------
    weights : array-like

        Either:

            [4]

        or:

            [N, 4]

        where columns correspond to:

            T1
            T1CE
            T2
            FLAIR

    If multiple samples are supplied, the mean weight of each
    sequence is displayed.
    """

    weights = np.asarray(
        weights,
        dtype=float
    )

    if weights.ndim == 1:

        if len(weights) != 4:

            raise ValueError(
                "Expected four modality weights."
            )

        mean_weights = weights

    elif weights.ndim == 2:

        if weights.shape[1] != 4:

            raise ValueError(
                "Expected weights with shape [N, 4]."
            )

        mean_weights = np.mean(
            weights,
            axis=0
        )

    else:

        raise ValueError(
            "weights must have shape [4] "
            "or [N, 4]."
        )

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.bar(
        CLASS_NAMES[:0] +
        [
            "T1",
            "T1CE",
            "T2",
            "FLAIR"
        ],
        mean_weights
    )

    ax.set_xlabel(
        "MRI Sequence"
    )

    ax.set_ylabel(
        "Attention Weight"
    )

    ax.set_title(
        title
    )

    ax.set_ylim(
        0,
        max(
            1.0,
            float(
                np.max(
                    mean_weights
                )
            ) * 1.15
        )
    )

    fig.tight_layout()

    if save_path is not None:

        _create_output_directory(
            save_path
        )

        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    return fig


# ============================================================
# Simple Test
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Example confusion matrix
    # --------------------------------------------------------

    cm = np.array([
        [448, 1, 1, 0],
        [2, 447, 1, 0],
        [1, 1, 447, 1],
        [0, 1, 1, 448]
    ])

    fig = plot_confusion_matrix(
        cm,
        class_names=CLASS_NAMES,
        normalize=False
    )

    plt.close(
        fig
    )

    # --------------------------------------------------------
    # Example uncertainty
    # --------------------------------------------------------

    uncertainty = np.array([
        0.04,
        0.08,
        0.12,
        0.31,
        0.16,
        0.72,
        0.21,
        0.11
    ])

    fig = plot_uncertainty(
        uncertainty
    )

    plt.close(
        fig
    )

    # --------------------------------------------------------
    # Example sequence attention
    # --------------------------------------------------------

    weights = np.array([
        0.20,
        0.35,
        0.18,
        0.27
    ])

    fig = plot_sequence_attention(
        weights
    )

    plt.close(
        fig
    )

    print(
        "Visualization utilities tested successfully."
    )