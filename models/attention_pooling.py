"""
Attention-Based Feature Aggregation
====================================

Implements the attention-based feature aggregation described
in Section 3.8 of the CSA-ViT methodology.

Equation (15):

    omega_i = exp(a_i) / sum_j exp(a_j)

Equation (16):

    F_G = sum_i omega_i T_i

Input:
    [B, N, D]

where:
    B = batch size
    N = number of tokens
    D = embedding dimension

Output:
    Global representation [B, D]
"""

import torch
import torch.nn as nn


class AttentionPooling(nn.Module):
    """
    Learnable attention-based token aggregation.

    Instead of conventional average pooling, each token
    receives a learned importance score.
    """

    def __init__(
        self,
        embedding_dim=256
    ):
        super().__init__()

        # ----------------------------------------------------
        # Learnable token relevance function
        #
        # a_i = Linear(T_i)
        # ----------------------------------------------------

        self.score = nn.Linear(
            embedding_dim,
            1
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Transformer token representations.

            Shape:
                [B, N, D]

        Returns
        -------
        global_feature : torch.Tensor
            Weighted global representation.

            Shape:
                [B, D]

        attention_weights : torch.Tensor
            Normalized token importance.

            Shape:
                [B, N, 1]
        """

        # ----------------------------------------------------
        # Validate input
        # ----------------------------------------------------

        if x.ndim != 3:

            raise ValueError(
                "AttentionPooling expects input "
                "with shape [B, N, D]."
            )

        # ----------------------------------------------------
        # Compute token relevance scores
        #
        # [B, N, D]
        #       ?
        # [B, N, 1]
        # ----------------------------------------------------

        scores = self.score(x)

        # ----------------------------------------------------
        # Softmax normalization across tokens
        #
        # This implements Equation (15).
        # ----------------------------------------------------

        attention_weights = torch.softmax(
            scores,
            dim=1
        )

        # ----------------------------------------------------
        # Weighted token aggregation
        #
        # F_G = sum_i omega_i T_i
        #
        # [B, N, 1] × [B, N, D]
        #       ?
        # [B, N, D]
        #
        # Sum over N:
        #       ?
        # [B, D]
        # ----------------------------------------------------

        global_feature = torch.sum(
            attention_weights * x,
            dim=1
        )

        return (
            global_feature,
            attention_weights
        )


# ============================================================
# Simple Test
# ============================================================

if __name__ == "__main__":

    batch_size = 2
    num_tokens = 196
    embedding_dim = 256

    x = torch.randn(
        batch_size,
        num_tokens,
        embedding_dim
    )

    pooling = AttentionPooling(
        embedding_dim=embedding_dim
    )

    global_feature, attention = pooling(
        x
    )

    print(
        "Input:",
        x.shape
    )

    print(
        "Global feature:",
        global_feature.shape
    )

    print(
        "Attention weights:",
        attention.shape
    )

    # Verify that attention weights sum to 1
    attention_sum = attention.sum(
        dim=1
    )

    print(
        "Attention sum:",
        attention_sum
    )