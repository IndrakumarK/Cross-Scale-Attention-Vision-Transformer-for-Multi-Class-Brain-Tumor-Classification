"""
CSA-ViT Evaluation Script
=========================

Evaluates a trained CSA-ViT model on a predefined dataset split.

Outputs:
    - Accuracy
    - Macro Precision
    - Macro Recall
    - Macro F1-score
    - Macro One-vs-Rest AUC
    - Class-wise classification metrics
    - Confusion matrix
    - Predictive entropy
    - Prediction confidence
    - Sample-level prediction CSV

IMPORTANT:
    The dataset split must already be created at the patient level.
    This script does NOT create train/validation/test splits.

    Predictive entropy represents sample-level predictive uncertainty.
    It does NOT generate a spatial uncertainty map.
"""

import os
import argparse
import csv

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.csavit import CSA_ViT
from data.dataset import BrainMRIDataset

from utils.metrics import (
    calculate_metrics,
    calculate_classwise_metrics,
    calculate_confusion_matrix
)

from utils.uncertainty import (
    predictive_entropy,
    prediction_confidence
)

from utils.visualization import (
    plot_confusion_matrix,
    plot_uncertainty
)


# ============================================================
# Class Names
# ============================================================

CLASS_NAMES = [
    "No-tumor",
    "Glioma",
    "Meningioma",
    "Pituitary"
]


# ============================================================
# Device
# ============================================================

def get_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# Load Model
# ============================================================

def load_model(
    checkpoint_path,
    device,
    num_classes=4
):

    """
    Load trained CSA-ViT model.
    """

    model = CSA_ViT(
        num_classes=num_classes
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    # Support:
    # 1. torch.save(model.state_dict())
    # 2. torch.save({"model_state_dict": ...})

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):

        state_dict = checkpoint["model_state_dict"]

    else:

        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    model.to(device)

    model.eval()

    return model


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate_model(
    model,
    dataloader,
    device
):

    """
    Evaluate CSA-ViT.

    Returns
    -------
    y_true : ndarray
        Ground-truth class labels.

    y_pred : ndarray
        Predicted class labels.

    y_prob : ndarray
        Predicted class probabilities.
    """

    all_labels = []
    all_predictions = []
    all_probabilities = []

    for batch in dataloader:

        # Dataset may return:
        # images, labels
        #
        # or:
        # images, labels, metadata

        if len(batch) == 2:

            images, labels = batch

        else:

            images = batch[0]
            labels = batch[1]

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        logits = model(images)

        # ----------------------------------------------------
        # Probability distribution
        # ----------------------------------------------------

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        predictions = torch.argmax(
            probabilities,
            dim=1
        )

        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_probabilities.append(
            probabilities.cpu()
        )

    y_true = np.asarray(
        all_labels
    )

    y_pred = np.asarray(
        all_predictions
    )

    y_prob = torch.cat(
        all_probabilities,
        dim=0
    ).numpy()

    return (
        y_true,
        y_pred,
        y_prob
    )


# ============================================================
# Save Classification Report
# ============================================================

def save_classification_report(
    classwise_metrics,
    save_path
):

    """
    Save class-wise metrics as CSV.
    """

    os.makedirs(
        os.path.dirname(save_path),
        exist_ok=True
    )

    with open(
        save_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Class",
            "Precision",
            "Recall",
            "F1-score",
            "Support"
        ])

        for class_name, values in (
            classwise_metrics.items()
        ):

            writer.writerow([
                class_name,
                f"{values['precision']:.6f}",
                f"{values['recall']:.6f}",
                f"{values['f1_score']:.6f}",
                values["support"]
            ])


# ============================================================
# Save Sample Predictions
# ============================================================

def save_predictions(
    y_true,
    y_pred,
    y_prob,
    entropy,
    confidence,
    save_path
):

    """
    Save sample-level predictions,
    confidence, and predictive entropy.
    """

    os.makedirs(
        os.path.dirname(save_path),
        exist_ok=True
    )

    with open(
        save_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Sample",
            "True Class",
            "Predicted Class",
            "Confidence",
            "Entropy",
            "Correct"
        ])

        for i in range(len(y_true)):

            true_class = CLASS_NAMES[
                int(y_true[i])
            ]

            predicted_class_name = CLASS_NAMES[
                int(y_pred[i])
            ]

            writer.writerow([
                i,
                true_class,
                predicted_class_name,
                f"{confidence[i]:.6f}",
                f"{entropy[i]:.6f}",
                int(y_true[i] == y_pred[i])
            ])


# ============================================================
# Print Metrics
# ============================================================

def print_metrics(metrics):

    print("\n" + "=" * 60)

    print(
        "CSA-ViT Evaluation Results"
    )

    print("=" * 60)

    print(
        f"Accuracy  : "
        f"{metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Precision : "
        f"{metrics['precision'] * 100:.2f}%"
    )

    print(
        f"Recall    : "
        f"{metrics['recall'] * 100:.2f}%"
    )

    print(
        f"F1-score  : "
        f"{metrics['f1_score'] * 100:.2f}%"
    )

    if not np.isnan(metrics["auc"]):

        print(
            f"Macro AUC : "
            f"{metrics['auc'] * 100:.2f}%"
        )

    else:

        print(
            "Macro AUC : N/A"
        )

    print("=" * 60)


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Evaluate a trained CSA-ViT model."
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained CSA-ViT checkpoint."
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing the evaluation images."
    )

    parser.add_argument(
        "--csv_file",
        type=str,
        required=True,
        help="CSV file containing image paths and labels."
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/evaluation"
    )

    args = parser.parse_args()

    # ========================================================
    # Device
    # ========================================================

    device = get_device()

    print(
        f"Using device: {device}"
    )

    # ========================================================
    # Dataset
    # ========================================================

    dataset = BrainMRIDataset(
        data_dir=args.data_dir,
        csv_file=args.csv_file,
        image_size=224,
        train=False
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    print(
        f"Number of evaluation samples: "
        f"{len(dataset)}"
    )

    # ========================================================
    # Model
    # ========================================================

    model = load_model(
        checkpoint_path=args.checkpoint,
        device=device,
        num_classes=4
    )

    # ========================================================
    # Evaluation
    # ========================================================

    y_true, y_pred, y_prob = evaluate_model(
        model,
        dataloader,
        device
    )

    # ========================================================
    # Classification Metrics
    # ========================================================

    metrics = calculate_metrics(
        y_true,
        y_pred,
        y_prob,
        num_classes=4
    )

    print_metrics(
        metrics
    )

    # ========================================================
    # Class-wise Metrics
    # ========================================================

    classwise = calculate_classwise_metrics(
        y_true,
        y_pred,
        class_names=CLASS_NAMES
    )

    print(
        "\nClass-wise Results:"
    )

    for class_name, values in classwise.items():

        print(
            f"\n{class_name}"
        )

        print(
            f"  Precision: "
            f"{values['precision']:.4f}"
        )

        print(
            f"  Recall: "
            f"{values['recall']:.4f}"
        )

        print(
            f"  F1-score: "
            f"{values['f1_score']:.4f}"
        )

        print(
            f"  Support: "
            f"{values['support']}"
        )

    # ========================================================
    # Output Directory
    # ========================================================

    os.makedirs(
        args.output_dir,
        exist_ok=True
    )

    # ========================================================
    # Save Classification Report
    # ========================================================

    save_classification_report(
        classwise,
        os.path.join(
            args.output_dir,
            "classification_report.csv"
        )
    )

    # ========================================================
    # Confusion Matrix
    # ========================================================

    cm = calculate_confusion_matrix(
        y_true,
        y_pred,
        num_classes=4
    )

    plot_confusion_matrix(
        cm,
        class_names=CLASS_NAMES,
        normalize=False,
        save_path=os.path.join(
            args.output_dir,
            "confusion_matrix.png"
        )
    )

    # ========================================================
    # Normalized Confusion Matrix
    # ========================================================

    plot_confusion_matrix(
        cm,
        class_names=CLASS_NAMES,
        normalize=True,
        save_path=os.path.join(
            args.output_dir,
            "confusion_matrix_normalized.png"
        ),
        title="Normalized CSA-ViT Confusion Matrix"
    )

    # ========================================================
    # Predictive Uncertainty
    # ========================================================

    probability_tensor = torch.tensor(
        y_prob,
        dtype=torch.float32
    )

    entropy = predictive_entropy(
        probability_tensor
    ).numpy()

    confidence = prediction_confidence(
        probability_tensor
    ).numpy()

    # ========================================================
    # Save Predictions and Uncertainty
    # ========================================================

    save_predictions(
        y_true,
        y_pred,
        y_prob,
        entropy,
        confidence,
        os.path.join(
            args.output_dir,
            "predictions_uncertainty.csv"
        )
    )

    # ========================================================
    # Uncertainty Plot
    # ========================================================

    plot_uncertainty(
        entropy,
        sample_indices=np.arange(
            len(entropy)
        ),
        save_path=os.path.join(
            args.output_dir,
            "uncertainty.png"
        )
    )

    # ========================================================
    # Summary
    # ========================================================

    print(
        "\nEvaluation completed."
    )

    print(
        f"Results saved to: "
        f"{args.output_dir}"
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    main()
