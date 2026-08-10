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
# Dataset Information
# ============================================================

EXPECTED_PATIENTS = 200
EXPECTED_SAMPLES = 1800
EXPECTED_SAMPLES_PER_CLASS = 450


# ============================================================
# Multi-Sequence MRI Dataset
# ============================================================

class MultiSequenceNPZDataset(Dataset):
    """
    Dataset loader for the CSA-ViT framework.

    Each NPZ file must contain four MRI sequences:

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

    The dataset manifest must assign each patient to exactly
    one split: train, val, or test.
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
        # Process MRI modalities
        # ----------------------------------------------------

        for modality in MODALITIES:

            if modality not in data:

                raise KeyError(
                    f"Modality '{modality}' is missing from "
                    f"sample: {file_path}"
                )

            image = data[modality]

            image = np.asarray(
                image,
                dtype=np.float32
            )

            # Intensity standardization and resizing
            image = preprocess_sequence(
                image,
                self.image_size
            )

            images.append(image)

        # ----------------------------------------------------
        # Combine four MRI modalities
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
        # Validate label
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
    Load and validate the patient-level dataset manifest.

    Required columns:

        path
        label
        patient_id
        split

    The manifest is expected to contain the selected
    1,800 MRI samples obtained from 200 patients.
    """

    df = pd.read_csv(csv_path)

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

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
    # Check missing values
    # --------------------------------------------------------

    if df[
        ["path", "label", "patient_id", "split"]
    ].isnull().any().any():

        raise ValueError(
            "The manifest contains missing values in "
            "path, label, patient_id, or split."
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

    # --------------------------------------------------------
    # Validate expected dataset size
    # --------------------------------------------------------

    number_of_patients = (
        df["patient_id"].nunique()
    )

    number_of_samples = len(df)

    if number_of_patients != EXPECTED_PATIENTS:

        raise ValueError(
            f"Expected {EXPECTED_PATIENTS} patients, "
            f"but manifest contains "
            f"{number_of_patients} unique patients."
        )

    if number_of_samples != EXPECTED_SAMPLES:

        raise ValueError(
            f"Expected {EXPECTED_SAMPLES} MRI samples, "
            f"but manifest contains "
            f"{number_of_samples} samples."
        )

    # --------------------------------------------------------
    # Validate overall class distribution
    # --------------------------------------------------------

    class_counts = (
        df["label"]
        .value_counts()
        .sort_index()
    )

    for label in range(len(CLASS_NAMES)):

        count = int(
            class_counts.get(label, 0)
        )

        if count != EXPECTED_SAMPLES_PER_CLASS:

            raise ValueError(
                f"Class '{CLASS_NAMES[label]}' contains "
                f"{count} samples; expected "
                f"{EXPECTED_SAMPLES_PER_CLASS}."
            )

    # --------------------------------------------------------
    # Check patient leakage
    # --------------------------------------------------------

    check_patient_leakage(df)

    return df


# ============================================================
# Obtain a Dataset Split
# ============================================================

def get_split(
    dataframe,
    split_name
):
    """
    Return one patient-level dataset split.

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

    if len(split_df) == 0:

        raise ValueError(
            f"The '{split_name}' split contains no samples."
        )

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

    This check is critical for preventing information
    leakage when multiple MRI samples originate from
    the same patient.
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

        leakage_ids = (
            leakage.index
            .tolist()
        )

        raise ValueError(
            "Patient-level data leakage detected. "
            "The following patient IDs occur in multiple "
            f"splits: {leakage_ids}"
        )

    return True


# ============================================================
# Dataset Summary
# ============================================================

def dataset_summary(dataframe):
    """
    Print the number of MRI samples and unique patients
    in the train, validation, and test partitions.
    """

    print("\nDataset Summary")
    print("=" * 65)

    total_samples = 0
    total_patients = set()

    for split in [
        "train",
        "val",
        "test"
    ]:

        subset = dataframe[
            dataframe["split"] == split
        ]

        samples = len(subset)

        patients = (
            subset["patient_id"]
            .nunique()
        )

        total_samples += samples

        total_patients.update(
            subset["patient_id"]
            .unique()
        )

        print(
            f"{split.upper():<10} "
            f"Samples: {samples:<6} "
            f"Patients: {patients}"
        )

    print("-" * 65)

    print(
        f"{'TOTAL':<10} "
        f"Samples: {total_samples:<6} "
        f"Patients: {len(total_patients)}"
    )

    print("=" * 65)

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    print("\nClass Distribution")
    print("=" * 65)

    for split in [
        "train",
        "val",
        "test"
    ]:

        subset = dataframe[
            dataframe["split"] == split
        ]

        print(f"\n{split.upper()}")

        counts = (
            subset["label"]
            .value_counts()
            .sort_index()
        )

        for label in range(
            len(CLASS_NAMES)
        ):

            count = int(
                counts.get(label, 0)
            )

            print(
                f"  {CLASS_NAMES[label]:<12} "
                f"= {count}"
            )

    print("=" * 65)


# ============================================================
# Validate Dataset Manifest
# ============================================================

def validate_manifest(dataframe):
    """
    Perform all dataset-level validation checks.
    """

    # Patient leakage
    check_patient_leakage(dataframe)

    # Expected patient count
    patients = dataframe[
        "patient_id"
    ].nunique()

    if patients != EXPECTED_PATIENTS:

        raise ValueError(
            f"Dataset contains {patients} patients. "
            f"Expected {EXPECTED_PATIENTS}."
        )

    # Expected sample count
    samples = len(dataframe)

    if samples != EXPECTED_SAMPLES:

        raise ValueError(
            f"Dataset contains {samples} samples. "
            f"Expected {EXPECTED_SAMPLES}."
        )

    print(
        "\nManifest validation successful."
    )

    print(
        f"Patients: {patients}"
    )

    print(
        f"Samples: {samples}"
    )

    print(
        "Patient-level leakage: None"
    )

    return True
