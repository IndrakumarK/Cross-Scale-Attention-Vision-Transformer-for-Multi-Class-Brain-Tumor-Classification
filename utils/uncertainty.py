"""
Uncertainty Analysis for CSA-ViT
================================

Implements the confidence-aware uncertainty analysis described
in Section 3.9 of the CSA-ViT manuscript.

The model produces a class-probability distribution:

    p(x) = [p1, p2, ..., pC]

Predictive uncertainty is quantified using normalized entropy:

    U(x) = -1/log(C) * sum_c p_c(x) log(p_c(x) + epsilon)

where:

    C = 4 diagnostic classes
    epsilon = small numerical-stability constant

Interpretation:

    U ˜ 0
        Confident prediction

    U ˜ 1
        High prediction ambiguity

IMPORTANT:
The entropy value is SAMPLE-LEVEL uncertainty.
It is not a spatial uncertainty map.
"""

import numpy as np
import torch


# ============================================================
# Predictive Entropy
# ============================================================

def predictive_entropy(
    probabilities,
    epsilon=1e-8,
    normalize=True
):
    """
    Calculate predictive entropy.

    Parameters
    ----------
    probabilities : torch.Tensor or numpy.ndarray

        Class probabilities.

        Shape can be:

            [C]
            [B, C]

    epsilon : float
        Small value used to avoid log(0).

    normalize : bool
        If True, entropy is normalized to [0, 1].

    Returns
    -------
    torch.Tensor or numpy.ndarray

        Entropy for each input sample.

        Shape:

            [B]
    """

    if isinstance(
        probabilities,
        np.ndarray
    ):

        probabilities = np.asarray(
            probabilities,
            dtype=np.float64
        )

        if probabilities.ndim == 1:

            probabilities = probabilities[
                np.newaxis,
                :
            ]

        if probabilities.ndim != 2:

            raise ValueError(
                "Probability array must have "
                "shape [B, C] or [C]."
            )

        # ----------------------------------------------------
        # Numerical stabilization
        # ----------------------------------------------------

        probabilities = np.clip(
            probabilities,
            epsilon,
            1.0
        )

        # ----------------------------------------------------
        # Normalize probabilities
        # ----------------------------------------------------

        probabilities = (
            probabilities /
            probabilities.sum(
                axis=1,
                keepdims=True
            )
        )

        # ----------------------------------------------------
        # Entropy
        # ----------------------------------------------------

        entropy = -np.sum(
            probabilities *
            np.log(probabilities),
            axis=1
        )

        # ----------------------------------------------------
        # Normalization
        # ----------------------------------------------------

        if normalize:

            num_classes = (
                probabilities.shape[1]
            )

            entropy = (
                entropy /
                np.log(num_classes)
            )

        return entropy

    # ========================================================
    # PyTorch implementation
    # ========================================================

    if not torch.is_tensor(
        probabilities
    ):

        raise TypeError(
            "probabilities must be a "
            "PyTorch tensor or NumPy array."
        )

    if probabilities.ndim == 1:

        probabilities = probabilities.unsqueeze(
            0
        )

    if probabilities.ndim != 2:

        raise ValueError(
            "Probability tensor must have "
            "shape [B, C] or [C]."
        )

    # --------------------------------------------------------
    # Numerical stabilization
    # --------------------------------------------------------

    probabilities = torch.clamp(
        probabilities,
        min=epsilon,
        max=1.0
    )

    # --------------------------------------------------------
    # Normalize probabilities
    # --------------------------------------------------------

    probabilities = (
        probabilities /
        probabilities.sum(
            dim=1,
            keepdim=True
        )
    )

    # --------------------------------------------------------
    # Entropy
    # --------------------------------------------------------

    entropy = -torch.sum(
        probabilities *
        torch.log(probabilities),
        dim=1
    )

    # --------------------------------------------------------
    # Normalize to [0, 1]
    # --------------------------------------------------------

    if normalize:

        num_classes = (
            probabilities.shape[1]
        )

        entropy = (
            entropy /
            torch.log(
                torch.tensor(
                    float(num_classes),
                    device=probabilities.device
                )
            )
        )

    return entropy


# ============================================================
# Entropy from Logits
# ============================================================

def entropy_from_logits(
    logits,
    normalize=True
):
    """
    Calculate predictive entropy directly from model logits.

    Parameters
    ----------
    logits : torch.Tensor

        Shape:

            [B, C]

    normalize : bool
        Whether to normalize entropy by log(C).

    Returns
    -------
    entropy : torch.Tensor

        Sample-level predictive uncertainty.
    """

    if logits.ndim != 2:

        raise ValueError(
            "Logits must have shape [B, C]."
        )

    probabilities = torch.softmax(
        logits,
        dim=1
    )

    return predictive_entropy(
        probabilities,
        normalize=normalize
    )


# ============================================================
# Confidence
# ============================================================

def prediction_confidence(
    probabilities
):
    """
    Calculate maximum predicted class probability.

    Confidence is:

        max_c p_c(x)

    Returns
    -------
    torch.Tensor or numpy.ndarray

        Maximum class probability for each sample.
    """

    if torch.is_tensor(
        probabilities
    ):

        if probabilities.ndim == 1:

            probabilities = probabilities.unsqueeze(
                0
            )

        return torch.max(
            probabilities,
            dim=1
        ).values

    probabilities = np.asarray(
        probabilities
    )

    if probabilities.ndim == 1:

        probabilities = probabilities[
            np.newaxis,
            :
        ]

    return np.max(
        probabilities,
        axis=1
    )


# ============================================================
# Predicted Class
# ============================================================

def predicted_class(
    probabilities
):
    """
    Return the class with the highest predicted probability.
    """

    if torch.is_tensor(
        probabilities
    ):

        if probabilities.ndim == 1:

            probabilities = probabilities.unsqueeze(
                0
            )

        return torch.argmax(
            probabilities,
            dim=1
        )

    probabilities = np.asarray(
        probabilities
    )

    if probabilities.ndim == 1:

        probabilities = probabilities[
            np.newaxis,
            :
        ]

    return np.argmax(
        probabilities,
        axis=1
    )


# ============================================================
# Uncertainty Summary
# ============================================================

def uncertainty_summary(
    probabilities,
    class_names=None
):
    """
    Generate a sample-level uncertainty summary.

    Returns a list containing:

        predicted class
        confidence
        normalized entropy
    """

    if class_names is None:

        class_names = [
            "Healthy",
            "Glioma",
            "Meningioma",
            "Pituitary Tumor"
        ]

    entropy = predictive_entropy(
        probabilities
    )

    confidence = prediction_confidence(
        probabilities
    )

    classes = predicted_class(
        probabilities
    )

    # --------------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------------

    if torch.is_tensor(
        entropy
    ):

        entropy = (
            entropy.detach()
            .cpu()
            .numpy()
        )

        confidence = (
            confidence.detach()
            .cpu()
            .numpy()
        )

        classes = (
            classes.detach()
            .cpu()
            .numpy()
        )

    summary = []

    for i in range(
        len(entropy)
    ):

        class_index = int(
            classes[i]
        )

        summary.append({
            "predicted_class":
                class_names[
                    class_index
                ],

            "confidence":
                float(
                    confidence[i]
                ),

            "normalized_entropy":
                float(
                    entropy[i]
                )
        })

    return summary


# ============================================================
# Simple Test
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Example probability distributions
    # --------------------------------------------------------

    probabilities = torch.tensor([
        [0.98, 0.01, 0.005, 0.005],
        [0.40, 0.30, 0.20, 0.10],
        [0.25, 0.25, 0.25, 0.25]
    ])

    # --------------------------------------------------------
    # Entropy
    # --------------------------------------------------------

    entropy = predictive_entropy(
        probabilities
    )

    print(
        "Normalized entropy:"
    )

    print(
        entropy
    )

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confidence = prediction_confidence(
        probabilities
    )

    print(
        "\nConfidence:"
    )

    print(
        confidence
    )

    # --------------------------------------------------------
    # Predicted classes
    # --------------------------------------------------------

    classes = predicted_class(
        probabilities
    )

    print(
        "\nPredicted classes:"
    )

    print(
        classes
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = uncertainty_summary(
        probabilities
    )

    print(
        "\nUncertainty summary:"
    )

    for item in summary:

        print(
            item
        )