"""
CSA-ViT Repeated Runs
=====================

Runs CSA-ViT five times using different random initialization seeds.

Seeds reported in the manuscript:
    42
    52
    62
    72
    82

For each run:
    - Same patient-level train/validation/test partition
    - Different random initialization seed
    - Same model configuration
    - Same optimizer and training settings

Outputs:
    results/tables/repeated_runs.csv
    results/tables/repeated_runs_summary.csv

The script does NOT hard-code the reported manuscript results.
All metrics are calculated from the actual predictions.
"""

import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
import yaml

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import (
    load_manifest,
    MultiSequenceNPZDataset
)

from models.csavit import CSAViT


# ============================================================
# Reproducibility
# ============================================================

def seed_everything(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Build CSA-ViT
# ============================================================

def build_model(cfg):

    model = CSAViT(
        in_channels=1,

        stem_channels=tuple(
            cfg["cnn_stem"]["channels"]
        ),

        embed_dim=cfg[
            "patch_embedding"
        ]["embedding_dim"],

        patch_sizes=tuple(
            cfg[
                "patch_embedding"
            ]["patch_sizes"]
        ),

        hcsam_blocks=cfg[
            "hcsam"
        ]["num_blocks"],

        heads=cfg[
            "hcsam"
        ]["num_heads"],

        transformer_layers=cfg[
            "transformer"
        ]["num_layers"],

        dropout=cfg[
            "transformer"
        ]["dropout"],

        num_classes=cfg[
            "data"
        ]["num_classes"]
    )

    return model


# ============================================================
# Train Model
# ============================================================

def train_model(
    model,
    train_loader,
    val_loader,
    cfg,
    device
):

    optimizer = AdamW(
        model.parameters(),

        lr=cfg[
            "training"
        ]["learning_rate"],

        betas=(
            cfg[
                "training"
            ]["betas"]["beta1"],

            cfg[
                "training"
            ]["betas"]["beta2"]
        ),

        weight_decay=cfg[
            "training"
        ]["weight_decay"]
    )

    scheduler = CosineAnnealingLR(
        optimizer,

        T_max=cfg[
            "scheduler"
        ]["T_max"],

        eta_min=cfg[
            "scheduler"
        ]["eta_min"]
    )

    criterion = torch.nn.CrossEntropyLoss(
        label_smoothing=cfg[
            "training"
        ]["label_smoothing"]
    )

    best_val_loss = float("inf")

    best_state = None

    patience = 0

    for epoch in range(
        cfg[
            "training"
        ]["epochs"]
    ):

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        model.train()

        train_loss = 0.0

        train_samples = 0

        for x, y in tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}"
        ):

            x = x.to(
                device,
                non_blocking=True
            )

            y = y.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            output = model(x)

            logits = output["logits"]

            loss = criterion(
                logits,
                y
            )

            loss.backward()

            optimizer.step()

            train_loss += (
                loss.item() *
                x.size(0)
            )

            train_samples += x.size(0)

        train_loss /= max(
            train_samples,
            1
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_loss = evaluate_loss(
            model,
            val_loader,
            criterion,
            device
        )

        scheduler.step()

        print(
            f"Epoch {epoch + 1:03d} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )

        # ----------------------------------------------------
        # Save best model in memory
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_state = {
                key: value.detach().cpu().clone()
                for key, value
                in model.state_dict().items()
            }

            patience = 0

        else:

            patience += 1

            if patience >= cfg[
                "training"
            ]["early_stopping_patience"]:

                print(
                    "Early stopping."
                )

                break

    # --------------------------------------------------------
    # Restore best validation model
    # --------------------------------------------------------

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

    return model


# ============================================================
# Validation Loss
# ============================================================

def evaluate_loss(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    total_loss = 0.0

    total_samples = 0

    with torch.no_grad():

        for x, y in loader:

            x = x.to(
                device,
                non_blocking=True
            )

            y = y.to(
                device,
                non_blocking=True
            )

            output = model(x)

            loss = criterion(
                output["logits"],
                y
            )

            total_loss += (
                loss.item() *
                x.size(0)
            )

            total_samples += x.size(0)

    return total_loss / max(
        total_samples,
        1
    )


# ============================================================
# Test Prediction
# ============================================================

def predict(
    model,
    test_loader,
    device
):

    model.eval()

    all_labels = []

    all_predictions = []

    all_probabilities = []

    with torch.no_grad():

        for x, y in test_loader:

            x = x.to(
                device,
                non_blocking=True
            )

            output = model(x)

            logits = output["logits"]

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            predictions = torch.argmax(
                probabilities,
                dim=1
            )

            all_labels.extend(
                y.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_probabilities.append(
                probabilities.cpu().numpy()
            )

    labels = np.asarray(
        all_labels
    )

    predictions = np.asarray(
        all_predictions
    )

    probabilities = np.concatenate(
        all_probabilities,
        axis=0
    )

    return (
        labels,
        predictions,
        probabilities
    )


# ============================================================
# Calculate Metrics
# ============================================================

def calculate_metrics(
    labels,
    predictions,
    probabilities
):

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision = precision_score(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )

    # --------------------------------------------------------
    # One-vs-rest macro AUC
    # --------------------------------------------------------

    try:

        auc = roc_auc_score(
            labels,
            probabilities,
            multi_class="ovr",
            average="macro"
        )

    except ValueError:

        auc = np.nan

    return {
        "Accuracy": accuracy * 100.0,
        "Precision": precision * 100.0,
        "Recall": recall * 100.0,
        "F1-score": f1 * 100.0,
        "AUC": auc * 100.0
        if not np.isnan(auc)
        else np.nan
    }


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run five repeated CSA-ViT experiments."
        )
    )

    parser.add_argument(
        "--config",
        default="configs/csavit_config.yaml"
    )

    parser.add_argument(
        "--manifest",
        required=True
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Load configuration
    # --------------------------------------------------------

    with open(
        args.config,
        "r",
        encoding="utf-8"
    ) as file:

        cfg = yaml.safe_load(
            file
        )

    # --------------------------------------------------------
    # Seeds from configuration
    # --------------------------------------------------------

    seeds = cfg[
        "repeated_runs"
    ]["seeds"]

    print(
        "\nRepeated-run seeds:",
        seeds
    )

    # --------------------------------------------------------
    # Load manifest
    # --------------------------------------------------------

    dataframe = load_manifest(
        args.manifest
    )

    # --------------------------------------------------------
    # Patient-level partitions
    #
    # IMPORTANT:
    # The split assignments are fixed across all five runs.
    # Only the random seed changes.
    # --------------------------------------------------------

    train_df = dataframe[
        dataframe["split"] == "train"
    ].reset_index(
        drop=True
    )

    val_df = dataframe[
        dataframe["split"] == "val"
    ].reset_index(
        drop=True
    )

    test_df = dataframe[
        dataframe["split"] == "test"
    ].reset_index(
        drop=True
    )

    print(
        f"\nTrain samples: {len(train_df)}"
    )

    print(
        f"Validation samples: {len(val_df)}"
    )

    print(
        f"Test samples: {len(test_df)}"
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nDevice: {device}"
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results = []

    # ========================================================
    # Five Repeated Runs
    # ========================================================

    for run_number, seed in enumerate(
        seeds,
        start=1
    ):

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"Repeated Run {run_number}"
        )

        print(
            f"Random Seed: {seed}"
        )

        print(
            "=" * 70
        )

        # ----------------------------------------------------
        # Set seed
        # ----------------------------------------------------

        seed_everything(
            seed
        )

        # ----------------------------------------------------
        # Datasets
        # ----------------------------------------------------

        train_dataset = MultiSequenceNPZDataset(
            train_df,
            cfg[
                "data"
            ]["image_size"]
        )

        val_dataset = MultiSequenceNPZDataset(
            val_df,
            cfg[
                "data"
            ]["image_size"]
        )

        test_dataset = MultiSequenceNPZDataset(
            test_df,
            cfg[
                "data"
            ]["image_size"]
        )

        # ----------------------------------------------------
        # DataLoaders
        # ----------------------------------------------------

        train_loader = DataLoader(
            train_dataset,

            batch_size=cfg[
                "data"
            ]["batch_size"],

            shuffle=True,

            num_workers=cfg[
                "data"
            ]["num_workers"],

            pin_memory=cfg[
                "data"
            ]["pin_memory"]
        )

        val_loader = DataLoader(
            val_dataset,

            batch_size=cfg[
                "data"
            ]["batch_size"],

            shuffle=False,

            num_workers=cfg[
                "data"
            ]["num_workers"],

            pin_memory=cfg[
                "data"
            ]["pin_memory"]
        )

        test_loader = DataLoader(
            test_dataset,

            batch_size=cfg[
                "data"
            ]["batch_size"],

            shuffle=False,

            num_workers=cfg[
                "data"
            ]["num_workers"],

            pin_memory=cfg[
                "data"
            ]["pin_memory"]
        )

        # ----------------------------------------------------
        # Build model
        # ----------------------------------------------------

        model = build_model(
            cfg
        ).to(device)

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        model = train_model(
            model,
            train_loader,
            val_loader,
            cfg,
            device
        )

        # ----------------------------------------------------
        # Test
        # ----------------------------------------------------

        labels, predictions, probabilities = predict(
            model,
            test_loader,
            device
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        metrics = calculate_metrics(
            labels,
            predictions,
            probabilities
        )

        result = {
            "Run": run_number,
            "Seed": seed,
            **metrics
        }

        results.append(
            result
        )

        print(
            "\nRun Results:"
        )

        for metric, value in metrics.items():

            print(
                f"{metric}: "
                f"{value:.4f}"
            )

    # ========================================================
    # Save Individual Run Results
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    os.makedirs(
        "results/tables",
        exist_ok=True
    )

    individual_file = (
        "results/tables/"
        "repeated_runs.csv"
    )

    results_df.to_csv(
        individual_file,
        index=False
    )

    # ========================================================
    # Calculate Mean and Standard Deviation
    # ========================================================

    metric_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "AUC"
    ]

    summary = []

    for metric in metric_columns:

        values = results_df[
            metric
        ].dropna()

        summary.append({
            "Metric": metric,
            "Mean": values.mean(),
            "Standard Deviation": (
                values.std(
                    ddof=1
                )
            )
        })

    summary_df = pd.DataFrame(
        summary
    )

    summary_file = (
        "results/tables/"
        "repeated_runs_summary.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False
    )

    # ========================================================
    # Display Final Summary
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL FIVE-RUN SUMMARY"
    )

    print(
        "=" * 70
    )

    for _, row in summary_df.iterrows():

        print(
            f"{row['Metric']:<12} "
            f"{row['Mean']:.4f} ± "
            f"{row['Standard Deviation']:.4f}"
        )

    print(
        "\nIndividual results saved to:"
    )

    print(
        individual_file
    )

    print(
        "\nSummary saved to:"
    )

    print(
        summary_file
    )


if __name__ == "__main__":
    main()