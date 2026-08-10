import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .preprocessing import preprocess_sequence


# ============================================================
# MRI Modalities
# ============================================================

MODALITIES = [
    "T1",
    "T1CE",
    "T2",
    "FLAIR"
]


# ============================================================
# Diagnostic Classes
# ============================================================

CLASS_NAMES = [
    "healthy",
    "glioma",
    "meningioma",
    "pituitary"
]


# ============================================================
# Multi-Sequence MRI Dataset
# ============================================================

class MultiSequenceNPZDataset(Dataset):
    """
    Dataset loader for CSA-ViT.

    Each .npz file must contain four MRI sequences:

        T1
        T1CE
        T2
        FLAIR

    Expected manifest columns:

        path
        label
        patient_id
        split

    Labels:

        0 -> Healthy
        1 -> Glioma
        2 -> Meningioma
        3 -> Pituitary
    """

    def __init__(
        self,
        dataframe,
        image_size=224
    ):
        self.df = dataframe.reset_index(drop=True)
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        # ----------------------------------------------------
        # Load multi-sequence MRI sample
        # ----------------------------------------------------

        file_path = row["path"]

        try:
            data = np.load(file_path)
        except Exception as e:
            raise RuntimeError(
                f"Unable to load MRI sample: {file_path}"
            ) from e

        images = []

        # ----------------------------------------------------
        # Process T1, T1CE, T2 and FLAIR
        # ----------------------------------------------------

        for modality in MODALITIES:

            if modality not in data:

                raise KeyError(
                    f"Modality '{modality}' is missing from "
                    f"sample: {file_path}"
                )

            image = data[modality]

            # Ensure numerical array
            image = np.asarray(
                image,
                dtype=np.float32
            )

            # Preprocessing:
            # intensity standardization + resizing
            image = preprocess_sequence(
                image,
                self.image_size
            )

            images.append(image)

        # ----------------------------------------------------
        # Combine modalities
        #
        # Output:
        # [4, 224, 224]
        #
        # Channel order:
        # T1, T1CE, T2, FLAIR
        # ----------------------------------------------------

        image = torch.cat(
            images,
            dim=0
        )

        # ----------------------------------------------------
        # Class label
        # ----------------------------------------------------

        label = int(row["label"])

        if label < 0 or label >= len(CLASS_NAMES):

            raise ValueError(
                f"Invalid class label {label} "
                f"for sample {file_path}"
            )

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return image, label


# ============================================================
# Manifest Loader
# ============================================================

def load_manifest(csv_path):
    """
    Load the patient-level dataset manifest.

    Required columns:

        path
        label
        patient_id
        split
    """

    df = pd.read_csv(csv_path)

    required_columns = {
        "path",
        "label",
        "patient_id",
        "split"
    }

    missing_columns = (
        required_columns -
        set(df.columns)
    )

    if missing_columns:

        raise ValueError(
            "The manifest is missing required columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    # --------------------------------------------------------
    # Validate split names
    # --------------------------------------------------------

    valid_splits = {
        "train",
        "val",
        "test"
    }

    invalid_splits = (
        set(df["split"].unique()) -
        valid_splits
    )

    if invalid_splits:

        raise ValueError(
            f"Invalid split values: {invalid_splits}. "
            f"Allowed values are: "
            f"{sorted(valid_splits)}"
        )

    # --------------------------------------------------------
    # Validate class labels
    # --------------------------------------------------------

    invalid_labels = df[
        ~df["label"].isin(
            range(len(CLASS_NAMES))
        )
    ]

    if len(invalid_labels) > 0:

        raise ValueError(
            "Manifest contains invalid class labels."
        )

    return df


# ============================================================
# Obtain a Dataset Split
# ============================================================

def get_split(
    dataframe,
    split_name
):
    """
    Return one patient-level split.

    split_name:
        train
        val
        test
    """

    valid_splits = {
        "train",
        "val",
        "test"
    }

    if split_name not in valid_splits:

        raise ValueError(
            f"Invalid split '{split_name}'. "
            f"Use train, val, or test."
        )

    split_df = dataframe[
        dataframe["split"] == split_name
    ].copy()

    return split_df.reset_index(
        drop=True
    )


# ============================================================
# Patient Leakage Check
# ============================================================

def check_patient_leakage(dataframe):
    """
    Verify that the same patient does not occur
    in more than one dataset split.

    This is important because the manuscript specifies
    patient-level partitioning to prevent information leakage.
    """

    patient_splits = (
        dataframe
        .groupby("patient_id")["split"]
        .nunique()
    )

    leakage = patient_splits[
        patient_splits > 1
    ]

    if len(leakage) > 0:

        raise ValueError(
            "Patient-level data leakage detected. "
            "The following patient IDs occur in multiple splits: "
            + ", ".join(
                map(str, leakage.index.tolist())
            )
        )

    return True


# ============================================================
# Dataset Summary
# ============================================================

def dataset_summary(dataframe):
    """
    Print a summary of samples and patients in
    train, validation and test partitions.
    """

    print("\nDataset Summary")
    print("=" * 60)

    for split in ["train", "val", "test"]:

        subset = dataframe[
            dataframe["split"] == split
        ]

        samples = len(subset)

        patients = subset[
            "patient_id"
        ].nunique()

        print(
            f"{split.upper():<10} "
            f"Samples: {samples:<6} "
            f"Patients: {patients}"
        )

    print("=" * 60)

    print("\nClass distribution:")

    for split in ["train", "val", "test"]:

        subset = dataframe[
            dataframe["split"] == split
        ]

        print(f"\n{split.upper()}")

        counts = (
            subset["label"]
            .value_counts()
            .sort_index()
        )

        for label, count in counts.items():

            print(
                f"  {label}: "
                f"{CLASS_NAMES[int(label)]} "
                f"= {count}"
            )