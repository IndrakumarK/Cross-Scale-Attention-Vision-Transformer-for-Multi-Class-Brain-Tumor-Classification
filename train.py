
import argparse
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from models.csavit import CSAViT
from data.dataset import BrainMRIDataset


CLASS_NAMES = [
    "No-tumor",
    "Glioma",
    "Meningioma",
    "Pituitary",
]

NUM_CLASSES = 4
SEEDS = [42, 52, 62, 72, 82]


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # The manuscript evaluates repeated independent runs.
    # Deterministic settings improve reproducibility.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_logits(output):
    """Support models returning either logits or {'logits': logits}."""
    if isinstance(output, dict):
        return output["logits"]
    return output


def compute_metrics(y_true, y_pred, y_prob):
    """Compute manuscript-reported classification metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "f1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
    }

    try:
        metrics["auc"] = roc_auc_score(
            y_true,
            y_prob,
            multi_class="ovr",
            average="macro",
            labels=np.arange(NUM_CLASSES),
        )
    except ValueError:
        # AUC is undefined if a validation split does not contain
        # all four classes. Training can continue, but this should
        # not occur for the manuscript's intended split.
        metrics["auc"] = float("nan")

    return metrics


def run_epoch(model, loader, criterion, optimizer, device, training):
    """Run one training or validation epoch."""
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    y_true = []
    y_pred = []
    y_prob = []

    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = extract_logits(model(images))
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)

            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

            y_true.extend(labels.detach().cpu().numpy().tolist())
            y_pred.extend(predictions.detach().cpu().numpy().tolist())
            y_prob.append(probabilities.detach().cpu())

    epoch_loss = total_loss / len(loader.dataset)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = torch.cat(y_prob, dim=0).numpy()

    metrics = compute_metrics(y_true, y_pred, y_prob)

    return epoch_loss, metrics, y_true, y_pred, y_prob


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    seed,
    val_loss,
    val_metrics,
):
    """Save the checkpoint corresponding to the lowest validation loss."""
    torch.save(
        {
            "epoch": epoch,
            "seed": seed,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_loss,
            "val_metrics": val_metrics,
            "class_names": CLASS_NAMES,
        },
        path,
    )


def save_history(history, path):
    """Save epoch-wise training/validation history."""
    fieldnames = [
        "epoch",
        "learning_rate",
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
        "val_auc",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def build_model():
    """
    Construct CSA-ViT using the architecture reported in the manuscript.

    The exact CSAViT implementation belongs in models/csavit.py.
    """
    model = CSAViT(
        in_channels=4,
        stem_channels=(64, 128),
        embed_dim=256,
        patch_sizes=(8, 16, 32),
        hcsam_blocks=6,
        heads=8,
        transformer_layers=6,
        dropout=0.1,
        num_classes=NUM_CLASSES,
    )
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train CSA-ViT on patient-level multi-sequence MRI data."
    )

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        choices=SEEDS,
        help="One of the five manuscript seeds: 42, 52, 62, 72, 82.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/csavit",
    )

    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    output_dir = Path(args.output_dir) / f"seed_{args.seed}"
    checkpoint_dir = output_dir / "checkpoints"
    figure_dir = output_dir / "figures"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("CSA-ViT TRAINING")
    print("=" * 72)
    print(f"Device       : {device}")
    if torch.cuda.is_available():
        print(f"GPU          : {torch.cuda.get_device_name(0)}")
    print(f"Seed         : {args.seed}")
    print(f"Batch size   : {args.batch_size}")
    print(f"Max epochs   : {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Weight decay : {args.weight_decay}")
    print(f"Patience     : {args.patience}")
    print("=" * 72)

    # Patient-level split manifests are supplied explicitly.
    # The same patient must never occur in both train and validation CSVs.
    train_dataset = BrainMRIDataset(
        data_dir=args.data_dir,
        csv_file=args.train_csv,
        image_size=224,
        train=True,
    )

    val_dataset = BrainMRIDataset(
        data_dir=args.data_dir,
        csv_file=args.val_csv,
        image_size=224,
        train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    print(f"Training samples  : {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    model = build_model().to(device)

    criterion = nn.CrossEntropyLoss(
        label_smoothing=args.label_smoothing
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    history = []
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics, _, _, _ = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            training=True,
        )

        val_loss, val_metrics, _, _, _ = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer=None,
            device=device,
            training=False,
        )

        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "learning_rate": current_lr,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_metrics["accuracy"],
            "val_accuracy": val_metrics["accuracy"],
            "train_precision": train_metrics["precision"],
            "val_precision": val_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "val_recall": val_metrics["recall"],
            "train_f1": train_metrics["f1"],
            "val_f1": val_metrics["f1"],
            "train_auc": train_metrics["auc"],
            "val_auc": val_metrics["auc"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"LR {current_lr:.7f} | "
            f"Train Loss {train_loss:.4f} | "
            f"Val Loss {val_loss:.4f} | "
            f"Train Acc {train_metrics['accuracy'] * 100:.2f}% | "
            f"Val Acc {val_metrics['accuracy'] * 100:.2f}% | "
            f"Val F1 {val_metrics['f1'] * 100:.2f}%"
        )

        # Manuscript protocol:
        # early stopping is based on validation loss, and the checkpoint
        # with the lowest validation loss is retained.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            best_path = checkpoint_dir / "csavit_best.pth"

            save_checkpoint(
                best_path,
                model,
                optimizer,
                scheduler,
                epoch,
                args.seed,
                val_loss,
                val_metrics,
            )

            print(
                f"  -> Best checkpoint saved "
                f"(validation loss = {val_loss:.6f})"
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(
                f"  -> Early stopping at epoch {epoch}; "
                f"best epoch = {best_epoch}"
            )
            break

    history_path = output_dir / "training_history.csv"
    save_history(history, history_path)

    config = {
        "seed": args.seed,
        "input_size": [224, 224],
        "modalities": ["T1", "T1CE", "T2", "FLAIR"],
        "patch_sizes": [8, 16, 32],
        "cnn_stem_channels": [64, 128],
        "embed_dim": 256,
        "hcsam_blocks": 6,
        "attention_heads": 8,
        "transformer_layers": 6,
        "dropout": 0.1,
        "optimizer": "AdamW",
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "betas": [0.9, 0.999],
        "label_smoothing": args.label_smoothing,
        "scheduler": "CosineAnnealingLR",
        "batch_size": args.batch_size,
        "max_epochs": args.epochs,
        "early_stopping_patience": args.patience,
        "best_checkpoint_selection": "lowest_validation_loss",
        "device": str(device),
    }

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("=" * 72)
    print("Training completed")
    print(f"Best epoch       : {best_epoch}")
    print(f"Best val loss    : {best_val_loss:.6f}")
    print(f"Best checkpoint  : {checkpoint_dir / 'csavit_best.pth'}")
    print(f"History          : {history_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
