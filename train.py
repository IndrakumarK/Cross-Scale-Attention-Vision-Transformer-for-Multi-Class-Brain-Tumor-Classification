"""
CSA-ViT Training Script
=======================

Training configuration follows the CSA-ViT manuscript.

Dataset:
    Multi-sequence MRI
    T1, T1CE, T2, FLAIR

Classes:
    0 - Healthy
    1 - Glioma
    2 - Meningioma
    3 - Pituitary Tumor

Model:
    CSA-ViT

Configuration:
    Input size          : 224 x 224
    Embedding dimension : 256
    Patch sizes         : 8, 16, 32
    HCSAM blocks        : 6
    Attention heads     : 8
    Transformer layers  : 6
    Dropout             : 0.1
    Activation          : GELU

Optimization:
    Optimizer           : AdamW
    Learning rate       : 1e-4
    Weight decay        : 1e-4
    Label smoothing     : 0.1
    Scheduler           : Cosine Annealing
    Batch size          : 32
    Maximum epochs      : 100
    Early stopping      : Enabled

IMPORTANT:
Patient-level train/validation/test splitting should be
performed before creating the CSV files used by this script.
"""

import os
import csv
import argparse
import random

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from models.csavit import CSAViT
from data.dataset import BrainMRIDataset

from utils.metrics import (
    calculate_metrics
)

from utils.visualization import (
    plot_training_curves
)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(
    seed=42
):
    """
    Set random seeds for reproducible experiments.
    """

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed(
            seed
        )

        torch.cuda.manual_seed_all(
            seed
        )

    # --------------------------------------------------------
    # Deterministic behavior
    # --------------------------------------------------------

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


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
# Training One Epoch
# ============================================================

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device
):
    """
    Train CSA-ViT for one epoch.
    """

    model.train()

    running_loss = 0.0

    all_labels = []
    all_predictions = []
    all_probabilities = []

    for batch in dataloader:

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
        # Clear gradients
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        output = model(
            images
        )

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
        # Loss
        # ----------------------------------------------------

        loss = criterion(
            logits,
            labels
        )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()

        # ----------------------------------------------------
        # Parameter update
        # ----------------------------------------------------

        optimizer.step()

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        batch_size = (
            images.size(0)
        )

        running_loss += (
            loss.item() *
            batch_size
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        predictions = torch.argmax(
            probabilities,
            dim=1
        )

        all_labels.extend(
            labels.detach()
            .cpu()
            .numpy()
        )

        all_predictions.extend(
            predictions.detach()
            .cpu()
            .numpy()
        )

        all_probabilities.append(
            probabilities.detach()
            .cpu()
        )

    # --------------------------------------------------------
    # Average loss
    # --------------------------------------------------------

    epoch_loss = (
        running_loss /
        len(dataloader.dataset)
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

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

    metrics = calculate_metrics(
        y_true,
        y_pred,
        y_prob,
        num_classes=4
    )

    return (
        epoch_loss,
        metrics
    )


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def validate(
    model,
    dataloader,
    criterion,
    device
):
    """
    Validate CSA-ViT.
    """

    model.eval()

    running_loss = 0.0

    all_labels = []
    all_predictions = []
    all_probabilities = []

    for batch in dataloader:

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
        # Forward
        # ----------------------------------------------------

        output = model(
            images
        )

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
        # Loss
        # ----------------------------------------------------

        loss = criterion(
            logits,
            labels
        )

        batch_size = (
            images.size(0)
        )

        running_loss += (
            loss.item() *
            batch_size
        )

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

        all_labels.extend(
            labels.cpu()
            .numpy()
        )

        all_predictions.extend(
            predictions.cpu()
            .numpy()
        )

        all_probabilities.append(
            probabilities.cpu()
        )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    epoch_loss = (
        running_loss /
        len(dataloader.dataset)
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

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

    metrics = calculate_metrics(
        y_true,
        y_pred,
        y_prob,
        num_classes=4
    )

    return (
        epoch_loss,
        metrics
    )


# ============================================================
# Save Checkpoint
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    metrics,
    path
):
    """
    Save the best CSA-ViT model checkpoint.
    """

    directory = os.path.dirname(
        path
    )

    if directory:

        os.makedirs(
            directory,
            exist_ok=True
        )

    torch.save(
        {
            "epoch": epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "scheduler_state_dict":
                scheduler.state_dict(),

            "metrics":
                metrics
        },
        path
    )


# ============================================================
# Save Training History
# ============================================================

def save_history(
    history,
    path
):
    """
    Save training history as CSV.
    """

    directory = os.path.dirname(
        path
    )

    if directory:

        os.makedirs(
            directory,
            exist_ok=True
        )

    number_epochs = len(
        history["train_loss"]
    )

    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "train_accuracy",
        "val_accuracy",
        "train_precision",
        "val_precision",
        "train_recall",
        "val_recall",
        "train_f1",
        "val_f1",
        "train_auc",
        "val_auc"
    ]

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for i in range(
            number_epochs
        ):

            writer.writerow(
                {
                    "epoch":
                        i + 1,

                    "train_loss":
                        history["train_loss"][i],

                    "val_loss":
                        history["val_loss"][i],

                    "train_accuracy":
                        history["train_accuracy"][i],

                    "val_accuracy":
                        history["val_accuracy"][i],

                    "train_precision":
                        history["train_precision"][i],

                    "val_precision":
                        history["val_precision"][i],

                    "train_recall":
                        history["train_recall"][i],

                    "val_recall":
                        history["val_recall"][i],

                    "train_f1":
                        history["train_f1"][i],

                    "val_f1":
                        history["val_f1"][i],

                    "train_auc":
                        history["train_auc"][i],

                    "val_auc":
                        history["val_auc"][i]
                }
            )


# ============================================================
# Main Training Function
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Train CSA-ViT on multi-sequence "
            "brain MRI data."
        )
    )

    # --------------------------------------------------------
    # Dataset arguments
    # --------------------------------------------------------

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root directory containing MRI images."
    )

    parser.add_argument(
        "--train_csv",
        type=str,
        required=True,
        help="Patient-level training split CSV."
    )

    parser.add_argument(
        "--val_csv",
        type=str,
        required=True,
        help="Patient-level validation split CSV."
    )

    # --------------------------------------------------------
    # Training arguments
    # --------------------------------------------------------

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.1
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=15
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/training"
    )

    args = parser.parse_args()

    # ========================================================
    # Reproducibility
    # ========================================================

    set_seed(
        args.seed
    )

    # ========================================================
    # Device
    # ========================================================

    device = get_device()

    print(
        "\nDevice:",
        device
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # ========================================================
    # Output directory
    # ========================================================

    os.makedirs(
        args.output_dir,
        exist_ok=True
    )

    checkpoint_dir = os.path.join(
        args.output_dir,
        "checkpoints"
    )

    os.makedirs(
        checkpoint_dir,
        exist_ok=True
    )

    # ========================================================
    # Training Dataset
    # ========================================================

    train_dataset = BrainMRIDataset(
        data_dir=args.data_dir,
        csv_file=args.train_csv,
        image_size=224,
        train=True
    )

    # ========================================================
    # Validation Dataset
    # ========================================================

    val_dataset = BrainMRIDataset(
        data_dir=args.data_dir,
        csv_file=args.val_csv,
        image_size=224,
        train=False
    )

    # ========================================================
    # DataLoaders
    # ========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )

    print(
        f"Training samples: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation samples: "
        f"{len(val_dataset)}"
    )

    # ========================================================
    # CSA-ViT Model
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

    model = model.to(
        device
    )

    # ========================================================
    # Loss
    # ========================================================

    criterion = nn.CrossEntropyLoss(
        label_smoothing=args.label_smoothing
    )

    # ========================================================
    # AdamW Optimizer
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )

    # ========================================================
    # Cosine Annealing Scheduler
    # ========================================================

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )

    # ========================================================
    # Training History
    # ========================================================

    history = {
        "train_loss": [],
        "val_loss": [],

        "train_accuracy": [],
        "val_accuracy": [],

        "train_precision": [],
        "val_precision": [],

        "train_recall": [],
        "val_recall": [],

        "train_f1": [],
        "val_f1": [],

        "train_auc": [],
        "val_auc": []
    }

    # ========================================================
    # Early Stopping
    # ========================================================

    best_val_accuracy = -np.inf

    best_epoch = 0

    epochs_without_improvement = 0

    # ========================================================
    # Training Loop
    # ========================================================

    for epoch in range(
        args.epochs
    ):

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"Epoch "
            f"{epoch + 1}/{args.epochs}"
        )

        print(
            "=" * 70
        )

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        train_loss, train_metrics = (
            train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device
            )
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_loss, val_metrics = (
            validate(
                model,
                val_loader,
                criterion,
                device
            )
        )

        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        scheduler.step()

        # ----------------------------------------------------
        # Store history
        # ----------------------------------------------------

        history["train_loss"].append(
            train_loss
        )

        history["val_loss"].append(
            val_loss
        )

        history["train_accuracy"].append(
            train_metrics["accuracy"]
        )

        history["val_accuracy"].append(
            val_metrics["accuracy"]
        )

        history["train_precision"].append(
            train_metrics["precision"]
        )

        history["val_precision"].append(
            val_metrics["precision"]
        )

        history["train_recall"].append(
            train_metrics["recall"]
        )

        history["val_recall"].append(
            val_metrics["recall"]
        )

        history["train_f1"].append(
            train_metrics["f1_score"]
        )

        history["val_f1"].append(
            val_metrics["f1_score"]
        )

        history["train_auc"].append(
            train_metrics["auc"]
        )

        history["val_auc"].append(
            val_metrics["auc"]
        )

        # ----------------------------------------------------
        # Current learning rate
        # ----------------------------------------------------

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        # ----------------------------------------------------
        # Display results
        # ----------------------------------------------------

        print(
            f"Learning Rate : "
            f"{current_lr:.8f}"
        )

        print(
            f"Train Loss    : "
            f"{train_loss:.5f}"
        )

        print(
            f"Val Loss      : "
            f"{val_loss:.5f}"
        )

        print(
            f"Train Accuracy: "
            f"{train_metrics['accuracy'] * 100:.2f}%"
        )

        print(
            f"Val Accuracy  : "
            f"{val_metrics['accuracy'] * 100:.2f}%"
        )

        print(
            f"Train F1      : "
            f"{train_metrics['f1_score'] * 100:.2f}%"
        )

        print(
            f"Val F1        : "
            f"{val_metrics['f1_score'] * 100:.2f}%"
        )

        # ----------------------------------------------------
        # Checkpoint
        # ----------------------------------------------------

        if (
            val_metrics["accuracy"]
            >
            best_val_accuracy
        ):

            best_val_accuracy = (
                val_metrics["accuracy"]
            )

            best_epoch = (
                epoch + 1
            )

            epochs_without_improvement = 0

            checkpoint_path = os.path.join(
                checkpoint_dir,
                "csavit_best.pth"
            )

            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch + 1,
                val_metrics,
                checkpoint_path
            )

            print(
                "\nBest model saved."
            )

        else:

            epochs_without_improvement += 1

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= args.patience
        ):

            print(
                "\nEarly stopping triggered."
            )

            break

    # ========================================================
    # Save Final Model
    # ========================================================

    final_checkpoint = os.path.join(
        checkpoint_dir,
        "csavit_final.pth"
    )

    save_checkpoint(
        model,
        optimizer,
        scheduler,
        epoch + 1,
        val_metrics,
        final_checkpoint
    )

    # ========================================================
    # Save History
    # ========================================================

    history_path = os.path.join(
        args.output_dir,
        "training_history.csv"
    )

    save_history(
        history,
        history_path
    )

    # ========================================================
    # Training Curves
    # ========================================================

    plot_training_curves(
        history,
        save_directory=os.path.join(
            args.output_dir,
            "figures"
        ),
        show=False
    )

    # ========================================================
    # Final Information
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "Training completed."
    )

    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy * 100:.2f}%"
    )

    print(
        f"Best epoch: "
        f"{best_epoch}"
    )

    print(
        f"Best checkpoint: "
        f"{os.path.join(checkpoint_dir, 'csavit_best.pth')}"
    )

    print(
        f"Training history: "
        f"{history_path}"
    )

    print(
        "=" * 70
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    main()