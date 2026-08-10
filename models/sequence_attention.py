"""
Adaptive Sequence Attention Module (SAM)
=========================================

Implements Section 3.6 of the CSA-ViT manuscript.

Four MRI sequences:

    T1
    T1CE
    T2
    FLAIR

Each modality is first processed by:

    CNN Stem
        ?
    Multi-Scale Patch Embedding
        ?
    HCSAM
        ?
    Scale aggregation
        ?
    256-dimensional modality representation

SAM then learns an adaptive importance weight for each
MRI sequence and performs weighted aggregation.

Equations:

    u_i = w^T tanh(W F_i + b)

    beta_i = exp(u_i) / sum_j exp(u_j)

    F_SAM = sum_i beta_i F_i

The four modality representations have the same dimension
(256), ensuring dimensional compatibility for weighted fusion.
"""

import torch
import torch.nn as nn


# ============================================================
# Sequence Attention Module
# ============================================================

class SequenceAttention(nn.Module):
    """
    Adaptive MRI sequence attention and fusion.

    Parameters
    ----------
    embedding_dim : int
        Common feature dimension.

        Default:
            256

    num_modalities : int
        Number of MRI sequences.

        Default:
            4

    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        embedding_dim=256,
        num_modalities=4,
        dropout=0.1
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_modalities = num_modalities

        # ----------------------------------------------------
        # Attention scoring function
        #
        # u_i = w^T tanh(WF_i + b)
        # ----------------------------------------------------

        self.feature_projection = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.score_projection = nn.Linear(
            embedding_dim,
            1
        )

        self.activation = nn.Tanh()

        self.dropout = nn.Dropout(
            dropout
        )

        # ----------------------------------------------------
        # Modality names
        # ----------------------------------------------------

        self.modality_names = [
            "T1",
            "T1CE",
            "T2",
            "FLAIR"
        ]

        # ----------------------------------------------------
        # Weight initialization
        # ----------------------------------------------------

        self._initialize_weights()

    # ========================================================
    # Initialization
    # ========================================================

    def _initialize_weights(self):

        nn.init.xavier_uniform_(
            self.feature_projection.weight
        )

        nn.init.zeros_(
            self.feature_projection.bias
        )

        nn.init.xavier_uniform_(
            self.score_projection.weight
        )

        nn.init.zeros_(
            self.score_projection.bias
        )

    # ========================================================
    # Aggregate HCSAM Scales
    # ========================================================

    def _aggregate_scales(
        self,
        modality_features
    ):
        """
        Convert the three HCSAM scale representations of one
        MRI modality into a single 256-dimensional feature.

        Input:

            E_s = [B, 784, 256]
            E_m = [B, 196, 256]
            E_l = [B, 49, 256]

        The token representations are concatenated and
        aggregated over the token dimension.

        Output:

            [B, 256]
        """

        if not isinstance(
            modality_features,
            (tuple, list)
        ):

            raise TypeError(
                "Each modality must contain the three "
                "HCSAM scale representations."
            )

        if len(
            modality_features
        ) != 3:

            raise ValueError(
                "Each modality must contain exactly "
                "three scale representations."
            )

        E_s, E_m, E_l = (
            modality_features
        )

        # ----------------------------------------------------
        # Validate dimensions
        # ----------------------------------------------------

        for name, feature in [
            ("small", E_s),
            ("medium", E_m),
            ("large", E_l)
        ]:

            if feature.ndim != 3:

                raise ValueError(
                    f"{name}-scale feature must have "
                    "shape [B, N, D]."
                )

            if feature.shape[-1] != self.embedding_dim:

                raise ValueError(
                    f"{name}-scale feature must have "
                    f"embedding dimension "
                    f"{self.embedding_dim}."
                )

        # ----------------------------------------------------
        # Concatenate tokens from all three scales
        #
        # 784 + 196 + 49 = 1029 tokens
        #
        # [B, 1029, 256]
        # ----------------------------------------------------

        tokens = torch.cat(
            [
                E_s,
                E_m,
                E_l
            ],
            dim=1
        )

        # ----------------------------------------------------
        # Global scale aggregation
        #
        # [B, 1029, 256]
        #          ?
        # [B, 256]
        # ----------------------------------------------------

        modality_feature = tokens.mean(
            dim=1
        )

        return modality_feature

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        modality_features
    ):
        """
        Parameters
        ----------
        modality_features : list or tuple

            Four modality representations:

                [
                    (Es_T1, Em_T1, El_T1),
                    (Es_T1CE, Em_T1CE, El_T1CE),
                    (Es_T2, Em_T2, El_T2),
                    (Es_FLAIR, Em_FLAIR, El_FLAIR)
                ]

        Returns
        -------
        fused_feature : torch.Tensor

            Adaptive fused representation.

            Shape:
                [B, 256]

        modality_weights : torch.Tensor

            Learned modality weights.

            Shape:
                [B, 4]
        """

        if not isinstance(
            modality_features,
            (tuple, list)
        ):

            raise TypeError(
                "SequenceAttention expects a list or tuple "
                "of modality representations."
            )

        if len(
            modality_features
        ) != self.num_modalities:

            raise ValueError(
                f"Expected {self.num_modalities} MRI "
                f"modalities, but received "
                f"{len(modality_features)}."
            )

        # ====================================================
        # Aggregate each modality
        # ====================================================

        modality_vectors = []

        for modality in modality_features:

            feature = self._aggregate_scales(
                modality
            )

            modality_vectors.append(
                feature
            )

        # ----------------------------------------------------
        # Stack modalities
        #
        # [B, 4, 256]
        # ----------------------------------------------------

        modality_matrix = torch.stack(
            modality_vectors,
            dim=1
        )

        # ====================================================
        # Adaptive attention scores
        #
        # u_i = w^T tanh(WF_i + b)
        # ====================================================

        projected = self.feature_projection(
            modality_matrix
        )

        projected = self.activation(
            projected
        )

        projected = self.dropout(
            projected
        )

        scores = self.score_projection(
            projected
        )

        # ----------------------------------------------------
        # [B, 4, 1]
        # ----------------------------------------------------

        # ====================================================
        # Softmax modality weights
        #
        # beta_i = softmax(u_i)
        # ====================================================

        modality_weights = torch.softmax(
            scores,
            dim=1
        )

        # ----------------------------------------------------
        # Remove final singleton dimension
        #
        # [B, 4, 1]
        #      ?
        # [B, 4]
        # ----------------------------------------------------

        modality_weights = (
            modality_weights.squeeze(
                -1
            )
        )

        # ====================================================
        # Weighted modality fusion
        #
        # F_SAM = sum beta_i F_i
        #
        # [B, 4, 256]
        #          ×
        # [B, 4, 1]
        #
        #        ?
        #
        # [B, 256]
        # ====================================================

        fused_feature = torch.sum(
            modality_matrix *
            modality_weights.unsqueeze(-1),
            dim=1
        )

        return (
            fused_feature,
            modality_weights
        )


# ============================================================
# Simple Test
# ============================================================

if __name__ == "__main__":

    batch_size = 2
    embed_dim = 256

    # --------------------------------------------------------
    # Create dummy HCSAM outputs for four MRI modalities.
    #
    # Each modality:
    #
    #   Es = 784 tokens
    #   Em = 196 tokens
    #   El = 49 tokens
    # --------------------------------------------------------

    modalities = []

    for _ in range(4):

        E_s = torch.randn(
            batch_size,
            784,
            embed_dim
        )

        E_m = torch.randn(
            batch_size,
            196,
            embed_dim
        )

        E_l = torch.randn(
            batch_size,
            49,
            embed_dim
        )

        modalities.append(
            (
                E_s,
                E_m,
                E_l
            )
        )

    # --------------------------------------------------------
    # Create SAM
    # --------------------------------------------------------

    sam = SequenceAttention(
        embedding_dim=256,
        num_modalities=4,
        dropout=0.1
    )

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    with torch.no_grad():

        fused, weights = sam(
            modalities
        )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print(
        "Fused feature shape:",
        fused.shape
    )

    print(
        "Sequence weights shape:",
        weights.shape
    )

    print(
        "\nSequence weights:"
    )

    print(
        weights
    )

    # --------------------------------------------------------
    # Verify weights sum to 1
    # --------------------------------------------------------

    print(
        "\nWeight sums:"
    )

    print(
        weights.sum(
            dim=1
        )
    )