"""
CSA-ViT Test Script
===================

Runs the trained CSA-ViT model on the test dataset.

Reports:
    - Accuracy
    - Precision
    - Recall
    - F1-score
    - Macro One-vs-Rest AUC
    - Confusion matrix
    - Class-wise results

This script is intended for the final held-out test split.
Patient-level splitting should be performed before creating
the test CSV.

Classes:
    0 - Healthy
    1 - Glioma
    2 - Meningioma
    3 - Pituitary Tumor
"""

import os
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.csavit import CSAViT
from data.dataset import BrainMRIDataset

from utils.metrics import (
    calculate_metrics,
    calculate_classwise_metrics,
    calculate_confusion_matrix
)

from utils.visualization import (
    plot_confusion_matrix
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
# Device
# ============================================================

def get_device():

    if torch.cuda.is_available():

        return torch.device(
            "cuda"
        )

    return torch.device(
        "cpu"
    )


# ============================================================
# Load Checkpoint
# ============================================================

def load_checkpoint(
    model,
    checkpoint_path,
    device
):
    """
    Load a trained CSA-ViT checkpoint.

    Supports either:

        model.state_dict()

    or:

        {
            "model_state_dict": ...
        }
    """

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    if (
        isinstance(
            checkpoint,
            dict
        )
        and
        "model_state_dict" in checkpoint
    ):

        state_dict = checkpoint[
            "model_state_dict"
        ]

    else:

        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    return model


# ============================================================
# Test Function
# ============================================================

@torch.no_grad()
def run_test(
    model,
    dataloader,
    device
):
    """
    Run inference on the test dataset.

    Returns
    -------
    y_true
    y_pred
    y_prob
    """

    model.eval()

    y_true = []
    y_pred = []
    y_prob = []

    for batch in dataloader:

        # ----------------------------------------------------
        # Expected dataset output:
        #
        # images, labels
        #
        # If metadata is returned:
        #
        # images, labels, metadata
        # ----------------------------------------------------

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

        output = model(
            images
        )

        # ----------------------------------------------------
        # Support both:
        #
        # logits tensor
        #
        # and model output dictionary
        # ----------------------------------------------------

        if isinstance(
            output,
            dict
        ):

            logits = output[
                "logits"
            ]

        else:

            logits = output

        # ----------------------------------------------------
        # Probabilities
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
        # Store
        # ----------------------------------------------------

        y_true.extend(
            labels.cpu().numpy()
        )

        y_pred.extend(
            predictions.cpu().numpy()
        )

        y_prob.append(
            probabilities.cpu()
        )

    y_true = np.asarray(
        y_true
    )

    y_pred = np.asarray(
        y_pred
    )

    y_prob = torch.cat(
        y_prob,
        dim=0
    ).numpy()

    return (
        y_true,
        y_pred,
        y_prob
    )


# ============================================================
# Print Overall Results
# ============================================================

def print_overall_results(
    metrics
):

    print(
        "\n"
        + "=" * 65
    )

    print(
        "CSA-ViT TEST-SET RESULTS"
    )

    print(
        "=" * 65
    )

    print(
        f"Accuracy   : "
        f"{metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Precision  : "
        f"{metrics['precision'] * 100:.2f}%"
    )

    print(
        f"Recall     : "
        f"{metrics['recall'] * 100:.2f}%"
    )

    print(
        f"F1-score   : "
        f"{metrics['f1_score'] * 100:.2f}%"
    )

    if np.isnan(
        metrics["auc"]
    ):

        print(
            "Macro AUC  : N/A"
        )

    else:

        print(
            f"Macro AUC  : "
            f"{metrics['auc'] * 100:.2f}%"
        )

    print(
        "=" * 65
    )


# ============================================================
# Print Class-wise Results
# ============================================================

def print_classwise_results(
    classwise
):

    print(
        "\nClass-wise Classification Results"
    )

    print(
        "-" * 80
    )

    print(
        f"{'Class':<20}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1-score':>12}"
        f"{'Support':>12}"
    )

    print(
        "-" * 80
    )

    for class_name, values in (
        classwise.items()
    ):

        print(
            f"{class_name:<20}"
            f"{values['precision']:>12.4f}"
            f"{values['recall']:>12.4f}"
            f"{values['f1_score']:>12.4f}"
            f"{values['support']:>12}"
        )

    print(
        "-" * 80
    )


# ============================================================
# Save Results
# ============================================================

def save_results(
    metrics,
    classwise,
    save_path
):
    """
    Save test results as a text file.
    """

    directory = os.path.dirname(
        save_path
    )

    if directory:

        os.makedirs(
            directory,
            exist_ok=True
        )

    with open(
        save_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "CSA-ViT Test Results\n"
        )

        file.write(
            "=" * 60 + "\n\n"
        )

        file.write(
            f"Accuracy: "
            f"{metrics['accuracy'] * 100:.4f}%\n"
        )

        file.write(
            f"Precision: "
            f"{metrics['precision'] * 100:.4f}%\n"
        )

        file.write(
            f"Recall: "
            f"{metrics['recall'] * 100:.4f}%\n"
        )

        file.write(
            f"F1-score: "
            f"{metrics['f1_score'] * 100:.4f}%\n"
        )

        file.write(
            f"Macro AUC: "
            f"{metrics['auc'] * 100:.4f}%\n"
        )

        file.write(
            "\nClass-wise Results\n"
        )

        file.write(
            "-" * 60 + "\n"
        )

        for class_name, values in (
            classwise.items()
        ):

            file.write(
                f"\n{class_name}\n"
            )

            file.write(
                f"Precision: "
                f"{values['precision']:.6f}\n"
            )

            file.write(
                f"Recall: "
                f"{values['recall']:.6f}\n"
            )

            file.write(
                f"F1-score: "
                f"{values['f1_score']:.6f}\n"
            )

            file.write(
                f"Support: "
                f"{values['support']}\n"
            )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Test a trained CSA-ViT model."
        )
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
        help="Root directory containing MRI data."
    )

    parser.add_argument(
        "--csv_file",
        type=str,
        required=True,
        help="CSV file describing the test split."
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
        default="results/test"
    )

    args = parser.parse_args()

    # ========================================================
    # Device
    # ========================================================

    device = get_device()

    print(
        f"Device: {device}"
    )

    # ========================================================
    # Dataset
    # ========================================================

    test_dataset = BrainMRIDataset(
        data_dir=args.data_dir,
        csv_file=args.csv_file,
        image_size=224,
        train=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    print(
        f"Test samples: "
        f"{len(test_dataset)}"
    )

    # ========================================================
    # Model
    # ========================================================

    model = CSAViT(
        in_channels=4,
        stem_channels=(64, 128),
        embed_dim=256,
        patch_sizes=(8, 16, 32),
        hcsam_blocks=6,
        heads=8,
        transformer_layers=6,
        dropout=0.1,
        num_classes=4
    )

    # ========================================================
    # Load trained weights
    # ========================================================

    model = load_checkpoint(
        model,
        args.checkpoint,
        device
    )

    model.to(
        device
    )

    model.eval()

    # ========================================================
    # Run test
    # ========================================================

    (
        y_true,
        y_pred,
        y_prob
    ) = run_test(
        model,
        test_loader,
        device
    )

    # ========================================================
    # Calculate metrics
    # ========================================================

    metrics = calculate_metrics(
        y_true,
        y_pred,
        y_prob,
        num_classes=4
    )

    # ========================================================
    # Display overall metrics
    # ========================================================

    print_overall_results(
        metrics
    )

    # ========================================================
    # Class-wise metrics
    # ========================================================

    classwise = calculate_classwise_metrics(
        y_true,
        y_pred,
        class_names=CLASS_NAMES
    )

    print_classwise_results(
        classwise
    )

    # ========================================================
    # Output directory
    # ========================================================

    os.makedirs(
        args.output_dir,
        exist_ok=True
    )

    # ========================================================
    # Save text results
    # ========================================================

    save_results(
        metrics,
        classwise,
        os.path.join(
            args.output_dir,
            "test_results.txt"
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

    np.savetxt(
        os.path.join(
            args.output_dir,
            "confusion_matrix.csv"
        ),
        cm,
        delimiter=",",
        fmt="%d"
    )

    # ========================================================
    # Confusion Matrix Figure
    # ========================================================

    plot_confusion_matrix(
        cm,
        class_names=CLASS_NAMES,
        normalize=False,
        save_path=os.path.join(
            args.output_dir,
            "confusion_matrix.png"
        ),
        title=(
            "CSA-ViT Test-Set Confusion Matrix"
        )
    )

    # ========================================================
    # Final message
    # ========================================================

    print(
        "\nTest evaluation completed."
    )

    print(
        "Results saved to:",
        args.output_dir
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    main()