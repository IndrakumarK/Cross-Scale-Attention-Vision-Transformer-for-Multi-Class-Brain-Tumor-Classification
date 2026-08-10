import numpy as np
import torch
import torch.nn.functional as F


# ============================================================
# CSA-ViT MRI PREPROCESSING
# ============================================================
#
# Manuscript preprocessing pipeline:
#
# 1. Bias-field correction
# 2. Skull stripping
# 3. Intensity standardization
# 4. Resizing to 224 x 224
#
# IMPORTANT:
# Bias-field correction and skull stripping are normally
# performed before this Python preprocessing stage using the
# corresponding MRI preprocessing pipeline.
#
# This file implements:
#   - intensity standardization
#   - spatial resizing
#
# ============================================================


# ------------------------------------------------------------
# 1. Intensity Standardization
# ------------------------------------------------------------

def intensity_standardize(
    image,
    epsilon=1.0e-6
):
    """
    Perform z-score intensity standardization.

    Equation used in the manuscript:

        I_hat(p) =
        [I(p) - E(I)] /
        sqrt(Var(I) + epsilon)

    Parameters
    ----------
    image : numpy.ndarray
        Input MRI image.

    epsilon : float
        Small constant used for numerical stability.

    Returns
    -------
    numpy.ndarray
        Standardized MRI image.
    """

    # Convert to floating-point representation
    image = np.asarray(
        image,
        dtype=np.float32
    )

    # Calculate image mean
    mean = np.mean(image)

    # Calculate image variance
    variance = np.var(image)

    # Z-score normalization
    standardized = (
        image - mean
    ) / np.sqrt(
        variance + epsilon
    )

    return standardized.astype(
        np.float32
    )


# ------------------------------------------------------------
# 2. Convert MRI Image to Tensor
# ------------------------------------------------------------

def to_tensor(image):
    """
    Convert a 2-D MRI image from NumPy format
    to a PyTorch tensor.

    Output shape:

        [1, H, W]
    """

    image = np.asarray(
        image,
        dtype=np.float32
    )

    tensor = torch.from_numpy(
        image
    ).float()

    # Add channel dimension
    tensor = tensor.unsqueeze(0)

    return tensor


# ------------------------------------------------------------
# 3. Resize MRI Image
# ------------------------------------------------------------

def resize_tensor(
    image,
    size=224
):
    """
    Resize a 2-D MRI image to 224 x 224.

    Bilinear interpolation is used for image data.

    Parameters
    ----------
    image : numpy.ndarray or torch.Tensor
        Input 2-D MRI image.

    size : int
        Target spatial size.

    Returns
    -------
    torch.Tensor
        Resized image with shape [1, size, size].
    """

    # --------------------------------------------------------
    # Convert NumPy image to tensor
    # --------------------------------------------------------

    if isinstance(
        image,
        np.ndarray
    ):

        image = to_tensor(
            image
        )

    elif isinstance(
        image,
        torch.Tensor
    ):

        image = image.float()

        # Expected input:
        # [H, W]
        # or
        # [1, H, W]

        if image.ndim == 2:

            image = image.unsqueeze(0)

    else:

        raise TypeError(
            "Image must be a NumPy array "
            "or PyTorch tensor."
        )

    # --------------------------------------------------------
    # Add batch dimension
    #
    # [1, H, W] -> [1, 1, H, W]
    # --------------------------------------------------------

    image = image.unsqueeze(0)

    # --------------------------------------------------------
    # Bilinear resizing
    # --------------------------------------------------------

    image = F.interpolate(
        image,
        size=(
            size,
            size
        ),
        mode="bilinear",
        align_corners=False
    )

    # --------------------------------------------------------
    # Remove batch dimension
    #
    # [1, 1, H, W] -> [1, H, W]
    # --------------------------------------------------------

    image = image.squeeze(0)

    return image


# ------------------------------------------------------------
# 4. Complete Sequence Preprocessing
# ------------------------------------------------------------

def preprocess_sequence(
    image,
    size=224,
    epsilon=1.0e-6
):
    """
    Preprocess one MRI sequence.

    Processing implemented here:

        MRI image
            ?
        Intensity standardization
            ?
        Resize to 224 × 224
            ?
        PyTorch tensor

    Parameters
    ----------
    image : numpy.ndarray
        Input MRI sequence.

    size : int
        Target image size.

    epsilon : float
        Numerical stability constant.

    Returns
    -------
    torch.Tensor
        Tensor with shape:

            [1, 224, 224]
    """

    # Step 1:
    # Intensity standardization
    image = intensity_standardize(
        image,
        epsilon=epsilon
    )

    # Step 2:
    # Spatial resizing
    image = resize_tensor(
        image,
        size=size
    )

    return image


# ------------------------------------------------------------
# 5. Preprocess All Four MRI Modalities
# ------------------------------------------------------------

def preprocess_multisequence(
    data,
    size=224,
    epsilon=1.0e-6
):
    """
    Preprocess T1, T1CE, T2 and FLAIR sequences.

    Parameters
    ----------
    data : dictionary-like object
        Must contain:

            T1
            T1CE
            T2
            FLAIR

    Returns
    -------
    torch.Tensor

        Shape:

            [4, 224, 224]

        Channel order:

            0 -> T1
            1 -> T1CE
            2 -> T2
            3 -> FLAIR
    """

    modalities = [
        "T1",
        "T1CE",
        "T2",
        "FLAIR"
    ]

    processed = []

    for modality in modalities:

        if modality not in data:

            raise KeyError(
                f"Required MRI modality "
                f"'{modality}' was not found."
            )

        image = preprocess_sequence(
            data[modality],
            size=size,
            epsilon=epsilon
        )

        processed.append(
            image
        )

    # Concatenate the four modalities
    # [1,H,W] x 4 -> [4,H,W]

    return torch.cat(
        processed,
        dim=0
    )


# ------------------------------------------------------------
# 6. Basic Validation
# ------------------------------------------------------------

def validate_preprocessed_image(
    image,
    expected_size=224
):
    """
    Validate the final preprocessed MRI tensor.

    Expected shape:

        [1, 224, 224]
        or
        [4, 224, 224]
    """

    if not isinstance(
        image,
        torch.Tensor
    ):

        raise TypeError(
            "Expected a PyTorch tensor."
        )

    if image.ndim != 3:

        raise ValueError(
            "Expected tensor with shape "
            "[C, H, W]."
        )

    if (
        image.shape[-2] != expected_size
        or
        image.shape[-1] != expected_size
    ):

        raise ValueError(
            f"Expected spatial size "
            f"{expected_size} x {expected_size}, "
            f"but received "
            f"{tuple(image.shape)}."
        )

    return True