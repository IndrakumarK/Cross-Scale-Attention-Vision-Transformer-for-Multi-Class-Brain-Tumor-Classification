"""
Evaluation Metrics for CSA-ViT
==============================

Provides classification metrics used in the CSA-ViT experiments:

    Accuracy
    Precision
    Recall
    F1-score
    Macro AUC

AUC is calculated using a one-vs-rest (OvR) strategy and
macro-averaged across the four diagnostic classes:

    0 - Healthy
    1 - Glioma
    2 - Meningioma
    3 - Pituitary Tumor
"""

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


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
# Basic Classification Metrics
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    y_prob=None,
    num_classes=4
):
    """
    Calculate classification metrics.

    Parameters
    ----------
    y_true : array-like
        Ground-truth class labels.

    y_pred : array-like
        Predicted class labels.

    y_prob : array-like, optional
        Predicted class probabilities.

        Shape:
            [N, C]

        Required for AUC.

    num_classes : int
        Number of diagnostic classes.

    Returns
    -------
    dict
        Classification metrics.
    """

    y_true = np.asarray(
        y_true
    )

    y_pred = np.asarray(
        y_pred
    )

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    # --------------------------------------------------------
    # Macro Precision
    # --------------------------------------------------------

    precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    # --------------------------------------------------------
    # Macro Recall
    # --------------------------------------------------------

    recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    # --------------------------------------------------------
    # Macro F1-score
    # --------------------------------------------------------

    f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    # --------------------------------------------------------
    # Macro One-vs-Rest AUC
    # --------------------------------------------------------

    auc = calculate_auc(
        y_true,
        y_prob,
        num_classes=num_classes
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc": auc
    }


# ============================================================
# AUC
# ============================================================

def calculate_auc(
    y_true,
    y_prob,
    num_classes=4
):
    """
    Calculate one-vs-rest macro AUC.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.

    y_prob : array-like
        Predicted class probabilities.

        Shape:
            [N, C]

    num_classes : int
        Number of classes.

    Returns
    -------
    float
        Macro-averaged one-vs-rest AUC.
    """

    if y_prob is None:
        return np.nan

    y_true = np.asarray(
        y_true
    )

    y_prob = np.asarray(
        y_prob
    )

    if y_prob.ndim != 2:

        raise ValueError(
            "y_prob must have shape [N, C]."
        )

    if y_prob.shape[1] != num_classes:

        raise ValueError(
            f"Expected probability matrix with "
            f"{num_classes} classes, but received "
            f"{y_prob.shape[1]}."
        )

    try:

        auc = roc_auc_score(
            y_true,
            y_prob,
            multi_class="ovr",
            average="macro"
        )

    except ValueError:

        # AUC cannot be calculated if the required
        # class representation is unavailable.
        auc = np.nan

    return auc


# ============================================================
# Per-Class Metrics
# ============================================================

def calculate_classwise_metrics(
    y_true,
    y_pred,
    class_names=None
):
    """
    Calculate precision, recall, F1-score and support
    for each diagnostic class.

    Returns a dictionary suitable for generating
    the classification report in the manuscript.
    """

    if class_names is None:

        class_names = CLASS_NAMES

    precision = precision_score(
        y_true,
        y_pred,
        labels=range(
            len(class_names)
        ),
        average=None,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        labels=range(
            len(class_names)
        ),
        average=None,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        labels=range(
            len(class_names)
        ),
        average=None,
        zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=range(
            len(class_names)
        )
    )

    support = cm.sum(
        axis=1
    )

    results = {}

    for i, class_name in enumerate(
        class_names
    ):

        results[class_name] = {
            "precision": precision[i],
            "recall": recall[i],
            "f1_score": f1[i],
            "support": int(
                support[i]
            )
        }

    return results


# ============================================================
# Confusion Matrix
# ============================================================

def calculate_confusion_matrix(
    y_true,
    y_pred,
    num_classes=4
):
    """
    Calculate the confusion matrix.

    Rows:
        Actual classes

    Columns:
        Predicted classes
    """

    return confusion_matrix(
        y_true,
        y_pred,
        labels=list(
            range(num_classes)
        )
    )


# ============================================================
# Convert Metrics to Percentage
# ============================================================

def metrics_to_percentage(
    metrics
):
    """
    Convert metrics expressed in [0, 1] into percentages.

    Example:

        0.9942 -> 99.42
    """

    converted = {}

    for key, value in metrics.items():

        if value is None:

            converted[key] = None

        elif np.isnan(value):

            converted[key] = np.nan

        else:

            converted[key] = (
                value * 100.0
            )

    return converted


# ============================================================
# Format Mean ± Standard Deviation
# ============================================================

def format_mean_std(
    values,
    decimals=2
):
    """
    Format repeated-run results as:

        mean ± standard deviation

    Parameters
    ----------
    values : array-like
        Results from repeated runs.

    decimals : int
        Number of decimal places.
    """

    values = np.asarray(
        values,
        dtype=float
    )

    mean = np.nanmean(
        values
    )

    std = np.nanstd(
        values,
        ddof=1
    )

    return (
        f"{mean:.{decimals}f} ± "
        f"{std:.{decimals}f}"
    )


# ============================================================
# Simple Test
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Example labels
    # --------------------------------------------------------

    y_true = np.array([
        0, 0, 0,
        1, 1, 1,
        2, 2, 2,
        3, 3, 3
    ])

    y_pred = np.array([
        0, 0, 0,
        1, 1, 1,
        2, 2, 2,
        3, 3, 3
    ])

    # --------------------------------------------------------
    # Example probabilities
    # --------------------------------------------------------

    y_prob = np.zeros(
        (len(y_true), 4)
    )

    for i, label in enumerate(
        y_true
    ):

        y_prob[i, label] = 1.0

    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    metrics = calculate_metrics(
        y_true,
        y_pred,
        y_prob
    )

    print(
        "Overall Metrics:"
    )

    for name, value in metrics.items():

        print(
            f"{name}: {value:.4f}"
        )

    # --------------------------------------------------------
    # Class-wise metrics
    # --------------------------------------------------------

    classwise = calculate_classwise_metrics(
        y_true,
        y_pred
    )

    print(
        "\nClass-wise Metrics:"
    )

    for class_name, values in classwise.items():

        print(
            f"\n{class_name}"
        )

        for metric, value in values.items():

            print(
                f"  {metric}: {value}"
            )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = calculate_confusion_matrix(
        y_true,
        y_pred
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        cm
    )