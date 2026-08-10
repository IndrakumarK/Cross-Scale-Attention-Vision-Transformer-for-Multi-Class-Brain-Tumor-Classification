"""
CSA-ViT Ablation Study
=======================

Configurations corresponding to the manuscript's ablation study:

1. Baseline Vision Transformer
2. Baseline ViT + Multi-Scale Patch Embedding
3. Previous + HCSAM
4. Previous + Sequence Attention Module

The script trains each configuration using the same patient-level
data split supplied through the manifest.

IMPORTANT:
Reported manuscript accuracies are NOT hard-coded. They must be
obtained by actually running the experiments.
"""

import argparse
import copy
import os
import random

import numpy as np
import pandas as pd
import torch
import yaml

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
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Build CSA-ViT Configuration
# ============================================================

def build_configuration(base_config, configuration):

    cfg = copy.deepcopy(base_config)

    if configuration == "baseline_vit":

        # ----------------------------------------------------
        # Baseline Vision Transformer
        #
        # Single-scale patch representation.
        # HCSAM disabled.
        # SAM disabled.
        # ----------------------------------------------------

        cfg["patch_embedding"]["patch_sizes"] = [16]

        cfg["hcsam"]["num_blocks"] = 0

        cfg["sam"]["enabled"] = False

        return cfg

    elif configuration == "multiscale":

        # ----------------------------------------------------
        # Baseline ViT + Multi-Scale Patch Embedding
        #
        # Multi-scale patches enabled.
        # HCSAM disabled.
        # SAM disabled.
        # ----------------------------------------------------

        cfg["patch_embedding"]["patch_sizes"] = [
            8,
            16,
            32
        ]

        cfg["hcsam"]["num_blocks"] = 0

        cfg["sam"]["enabled"] = False

        return cfg

    elif configuration == "hcsam":

        # ----------------------------------------------------
        # Multi-Scale Patch Embedding + HCSAM
        #
        # SAM disabled.
        # ----------------------------------------------------

        cfg["patch_embedding"]["patch_sizes"] = [
            8,
            16,
            32
        ]

        cfg["hcsam"]["num_blocks"] = 6

        cfg["sam"]["enabled"] = False

        return cfg

    elif configuration == "csavit":

        # ----------------------------------------------------
        # Complete CSA-ViT
        #
        # Multi-scale patch embedding
        # + HCSAM
        # + SAM
        # ----------------------------------------------------

        cfg["patch_embedding"]["patch_sizes"] = [
            8,
            16,
            32
        ]

        cfg["hcsam"]["num_blocks"] = 6

        cfg["sam"]["enabled"] = True

        return cfg

    else:

        raise ValueError(
            f"Unknown configuration: {configuration}"
        )


# ============================================================
# Build Model
# ============================================================

def build_model(cfg, configuration):

    # --------------------------------------------------------
    # The current CSAViT implementation contains the complete
    # architecture. For the ablation experiments, architectural
    # switches should be implemented in the model itself before
    # executing these configurations.
    #
    # This function therefore validates the configuration and
    # constructs the corresponding model.
    # --------------------------------------------------------

    if configuration != "csavit":

        raise NotImplementedError(
            "Baseline/multiscale/HCSAM ablations require "
            "explicit architecture switches in CSAViT. "
            "Do not silently substitute the complete model."
        )

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
# Train One Configuration
# ============================================================

def train_configuration(
    cfg,
    configuration,
    train_loader,
    val_loader,
    train_size,
    val_size,
    device
):

    model = build_model(
        cfg,
        configuration
    ).to(device)

    optimizer = AdamW(
        model.parameters(),

        lr=cfg[
            "training"
        ]["learning_rate"],

        betas=tuple(
            cfg[
                "training"
            ]["betas"]
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

    patience = 0

    history = {
        "train_loss": [],
        "val_loss": []
    }

    checkpoint_dir = os.path.join(
        cfg[
            "training"
        ]["checkpoint_dir"],
        "ablation"
    )

    os.makedirs(
        checkpoint_dir,
        exist_ok=True
    )

    checkpoint_path = os.path.join(
        checkpoint_dir,
        f"{configuration}_best.pt"
    )

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    for epoch in range(
        cfg[
            "training"
        ]["epochs"]
    ):

        model.train()

        running_train_loss = 0.0

        for x, y in tqdm(
            train_loader,
            desc=(
                f"{configuration} | "
                f"Epoch {epoch + 1}"
            )
        ):

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            output = model(x)

            loss = criterion(
                output["logits"],
                y
            )

            loss.backward()

            optimizer.step()

            running_train_loss += (
                loss.item() *
                x.size(0)
            )

        train_loss = (
            running_train_loss /
            max(train_size, 1)
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        model.eval()

        running_val_loss = 0.0

        with torch.no_grad():

            for x, y in val_loader:

                x = x.to(device)
                y = y.to(device)

                output = model(x)

                loss = criterion(
                    output["logits"],
                    y
                )

                running_val_loss += (
                    loss.item() *
                    x.size(0)
                )

        val_loss = (
            running_val_loss /
            max(val_size, 1)
        )

        scheduler.step()

        history[
            "train_loss"
        ].append(train_loss)

        history[
            "val_loss"
        ].append(val_loss)

        print(
            f"Epoch {epoch + 1:03d} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            patience = 0

            torch.save(
                model.state_dict(),
                checkpoint_path
            )

        else:

            patience += 1

            if patience >= cfg[
                "training"
            ]["early_stopping_patience"]:

                print(
                    "Early stopping."
                )

                break

    return {
        "configuration": configuration,
        "best_validation_loss": best_val_loss,
        "checkpoint": checkpoint_path,
        "history": history
    }


# ============================================================
# Main Ablation Experiment
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "CSA-ViT ablation study"
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

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--configurations",
        nargs="+",
        default=[
            "baseline_vit",
            "multiscale",
            "hcsam",
            "csavit"
        ]
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

        base_config = yaml.safe_load(
            file
        )

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    seed_everything(
        args.seed
    )

    # --------------------------------------------------------
    # Load manifest
    # --------------------------------------------------------

    dataframe = load_manifest(
        args.manifest
    )

    # --------------------------------------------------------
    # Patient-level splits are supplied by the manifest.
    # --------------------------------------------------------

    train_df = dataframe[
        dataframe["split"] == "train"
    ].reset_index(drop=True)

    val_df = dataframe[
        dataframe["split"] == "val"
    ].reset_index(drop=True)

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset = MultiSequenceNPZDataset(
        train_df,
        base_config[
            "data"
        ]["image_size"]
    )

    val_dataset = MultiSequenceNPZDataset(
        val_df,
        base_config[
            "data"
        ]["image_size"]
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,

        batch_size=base_config[
            "data"
        ]["batch_size"],

        shuffle=True,

        num_workers=base_config[
            "data"
        ]["num_workers"],

        pin_memory=base_config[
            "data"
        ]["pin_memory"]
    )

    val_loader = DataLoader(
        val_dataset,

        batch_size=base_config[
            "data"
        ]["batch_size"],

        shuffle=False,

        num_workers=base_config[
            "data"
        ]["num_workers"],

        pin_memory=base_config[
            "data"
        ]["pin_memory"]
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
    # Run configurations
    # --------------------------------------------------------

    results = []

    for configuration in args.configurations:

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"Running ablation: "
            f"{configuration}"
        )

        print(
            "=" * 70
        )

        cfg = build_configuration(
            base_config,
            configuration
        )

        result = train_configuration(
            cfg,
            configuration,
            train_loader,
            val_loader,
            len(train_dataset),
            len(val_dataset),
            device
        )

        results.append({
            "Configuration": configuration,
            "Best Validation Loss":
                result[
                    "best_validation_loss"
                ],
            "Checkpoint":
                result[
                    "checkpoint"
                ]
        })

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    os.makedirs(
        "results/tables",
        exist_ok=True
    )

    results_df = pd.DataFrame(
        results
    )

    output_file = (
        "results/tables/"
        "ablation_results.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    print(
        "\nAblation experiment completed."
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    print(
        f"\nResults saved to: "
        f"{output_file}"
    )


if __name__ == "__main__":
    main()