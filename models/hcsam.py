"""
Hierarchical Cross-Scale Attention Module (HCSAM)
==================================================

CSA-ViT implementation corresponding to Section 3.5.

Cross-scale interactions:

    E_s -> E_m
    E_m -> E_l

where:

    E_s : fine-scale tokens
          784 tokens for 8x8 patches

    E_m : medium-scale tokens
          196 tokens for 16x16 patches

    E_l : coarse-scale tokens
          49 tokens for 32x32 patches

The finer-scale representation provides the Query, while the
adjacent coarser-scale representation provides Key and Value.

The query resolution is preserved.

Equations:

    A_sm = Softmax(
        (E_s W_Q)(E_m W_K)^T / sqrt(d_k)
    )

    Z_sm = A_sm(E_m W_V)

    F_HCSA = LN(E_s + gamma Z_sm)

The same principle is applied to E_m -> E_l.
"""

import math

import torch
import torch.nn as nn


# ============================================================
# Cross-Scale Attention
# ============================================================

class CrossScaleAttention(nn.Module):
    """
    Multi-head cross-attention between two spatial scales.

    Query:
        Finer-scale representation

    Key/Value:
        Adjacent coarser-scale representation

    Example:

        Query  = Es [B, 784, 256]
        Key    = Em [B, 196, 256]
        Value  = Em [B, 196, 256]

    Output:

        [B, 784, 256]

    The query token count is preserved.
    """

    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        dropout=0.1
    ):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                "embed_dim must be divisible by num_heads."
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # ----------------------------------------------------
        # Query projection
        # ----------------------------------------------------

        self.q_proj = nn.Linear(
            embed_dim,
            embed_dim
        )

        # ----------------------------------------------------
        # Key projection
        # ----------------------------------------------------

        self.k_proj = nn.Linear(
            embed_dim,
            embed_dim
        )

        # ----------------------------------------------------
        # Value projection
        # ----------------------------------------------------

        self.v_proj = nn.Linear(
            embed_dim,
            embed_dim
        )

        # ----------------------------------------------------
        # Output projection
        # ----------------------------------------------------

        self.out_proj = nn.Linear(
            embed_dim,
            embed_dim
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.scale = math.sqrt(
            self.head_dim
        )

        self._initialize_weights()

    # ========================================================
    # Weight Initialization
    # ========================================================

    def _initialize_weights(self):

        nn.init.xavier_uniform_(
            self.q_proj.weight
        )

        nn.init.xavier_uniform_(
            self.k_proj.weight
        )

        nn.init.xavier_uniform_(
            self.v_proj.weight
        )

        nn.init.xavier_uniform_(
            self.out_proj.weight
        )

        nn.init.zeros_(
            self.q_proj.bias
        )

        nn.init.zeros_(
            self.k_proj.bias
        )

        nn.init.zeros_(
            self.v_proj.bias
        )

        nn.init.zeros_(
            self.out_proj.bias
        )

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        query,
        key_value,
        return_attention=False
    ):
        """
        Parameters
        ----------
        query : torch.Tensor

            Fine-scale representation.

            Shape:
                [B, Nq, D]

        key_value : torch.Tensor

            Coarser-scale representation.

            Shape:
                [B, Nk, D]

        return_attention : bool

            If True, return attention maps.

        Returns
        -------
        output : torch.Tensor

            Shape:
                [B, Nq, D]

        attention : torch.Tensor, optional

            Shape:
                [B, H, Nq, Nk]
        """

        if query.ndim != 3:
            raise ValueError(
                "Query must have shape [B, N, D]."
            )

        if key_value.ndim != 3:
            raise ValueError(
                "Key/value must have shape [B, N, D]."
            )

        if query.shape[-1] != self.embed_dim:
            raise ValueError(
                "Query embedding dimension does not "
                "match embed_dim."
            )

        if key_value.shape[-1] != self.embed_dim:
            raise ValueError(
                "Key/value embedding dimension does "
                "not match embed_dim."
            )

        # ----------------------------------------------------
        # Dimensions
        # ----------------------------------------------------

        batch_size = query.shape[0]

        query_tokens = query.shape[1]

        key_tokens = key_value.shape[1]

        # ----------------------------------------------------
        # Linear projections
        # ----------------------------------------------------

        q = self.q_proj(
            query
        )

        k = self.k_proj(
            key_value
        )

        v = self.v_proj(
            key_value
        )

        # ----------------------------------------------------
        # Split into attention heads
        #
        # [B, N, D]
        #
        # ->
        #
        # [B, H, N, Dh]
        # ----------------------------------------------------

        q = q.reshape(
            batch_size,
            query_tokens,
            self.num_heads,
            self.head_dim
        ).transpose(
            1,
            2
        )

        k = k.reshape(
            batch_size,
            key_tokens,
            self.num_heads,
            self.head_dim
        ).transpose(
            1,
            2
        )

        v = v.reshape(
            batch_size,
            key_tokens,
            self.num_heads,
            self.head_dim
        ).transpose(
            1,
            2
        )

        # ----------------------------------------------------
        # Attention scores
        #
        # Q K^T / sqrt(d_k)
        #
        # [B,H,Nq,Dh]
        #          ×
        # [B,H,Dh,Nk]
        #
        # ->
        # [B,H,Nq,Nk]
        # ----------------------------------------------------

        attention_scores = torch.matmul(
            q,
            k.transpose(
                -2,
                -1
            )
        ) / self.scale

        # ----------------------------------------------------
        # Softmax
        # ----------------------------------------------------

        attention_weights = torch.softmax(
            attention_scores,
            dim=-1
        )

        attention_weights = self.dropout(
            attention_weights
        )

        # ----------------------------------------------------
        # Weighted Value aggregation
        #
        # A V
        # ----------------------------------------------------

        output = torch.matmul(
            attention_weights,
            v
        )

        # ----------------------------------------------------
        # Merge heads
        #
        # [B,H,Nq,Dh]
        #
        # ->
        #
        # [B,Nq,D]
        # ----------------------------------------------------

        output = output.transpose(
            1,
            2
        ).contiguous()

        output = output.reshape(
            batch_size,
            query_tokens,
            self.embed_dim
        )

        # ----------------------------------------------------
        # Output projection
        # ----------------------------------------------------

        output = self.out_proj(
            output
        )

        if return_attention:

            return (
                output,
                attention_weights
            )

        return output


# ============================================================
# HCSAM Block
# ============================================================

class HCSAMBlock(nn.Module):
    """
    One hierarchical cross-scale attention block.

    It performs:

        Fine -> Medium
        Medium -> Large

    independently and preserves the query resolutions.

    For Es -> Em:

        Query = Es
        Key   = Em
        Value = Em

    For Em -> El:

        Query = Em
        Key   = El
        Value = El
    """

    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        dropout=0.1
    ):
        super().__init__()

        # ----------------------------------------------------
        # Small -> Medium
        # ----------------------------------------------------

        self.small_to_medium = CrossScaleAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # ----------------------------------------------------
        # Medium -> Large
        # ----------------------------------------------------

        self.medium_to_large = CrossScaleAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # ----------------------------------------------------
        # Learnable scaling coefficients
        #
        # gamma_sm
        # gamma_ml
        # ----------------------------------------------------

        self.gamma_sm = nn.Parameter(
            torch.tensor(1.0)
        )

        self.gamma_ml = nn.Parameter(
            torch.tensor(1.0)
        )

        # ----------------------------------------------------
        # Layer normalization
        # ----------------------------------------------------

        self.norm_small = nn.LayerNorm(
            embed_dim
        )

        self.norm_medium = nn.LayerNorm(
            embed_dim
        )

        # ----------------------------------------------------
        # Dropout
        # ----------------------------------------------------

        self.dropout = nn.Dropout(
            dropout
        )

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        E_s,
        E_m,
        E_l,
        return_attention=False
    ):
        """
        Parameters
        ----------
        E_s : torch.Tensor
            Small-scale tokens.

            [B, 784, 256]

        E_m : torch.Tensor
            Medium-scale tokens.

            [B, 196, 256]

        E_l : torch.Tensor
            Large-scale tokens.

            [B, 49, 256]

        Returns
        -------
        refined_small
            [B, 784, 256]

        refined_medium
            [B, 196, 256]

        refined_large
            [B, 49, 256]
        """

        # ====================================================
        # Small -> Medium
        # ====================================================
        #
        # Query = E_s
        # Key   = E_m
        # Value = E_m
        #
        # Output retains 784 query tokens.
        # ====================================================

        if return_attention:

            Z_sm, A_sm = (
                self.small_to_medium(
                    E_s,
                    E_m,
                    return_attention=True
                )
            )

        else:

            Z_sm = self.small_to_medium(
                E_s,
                E_m
            )

            A_sm = None

        refined_small = self.norm_small(
            E_s +
            self.gamma_sm *
            self.dropout(
                Z_sm
            )
        )

        # ====================================================
        # Medium -> Large
        # ====================================================
        #
        # Query = E_m
        # Key   = E_l
        # Value = E_l
        #
        # Output retains 196 query tokens.
        # ====================================================

        if return_attention:

            Z_ml, A_ml = (
                self.medium_to_large(
                    E_m,
                    E_l,
                    return_attention=True
                )
            )

        else:

            Z_ml = self.medium_to_large(
                E_m,
                E_l
            )

            A_ml = None

        refined_medium = self.norm_medium(
            E_m +
            self.gamma_ml *
            self.dropout(
                Z_ml
            )
        )

        # ----------------------------------------------------
        # Large-scale representation is preserved.
        # ----------------------------------------------------

        refined_large = E_l

        if return_attention:

            return (
                refined_small,
                refined_medium,
                refined_large,
                {
                    "small_to_medium": A_sm,
                    "medium_to_large": A_ml
                }
            )

        return (
            refined_small,
            refined_medium,
            refined_large
        )


# ============================================================
# Hierarchical Cross-Scale Attention Module
# ============================================================

class HCSAM(nn.Module):
    """
    Hierarchical Cross-Scale Attention Module.

    Default configuration:

        embedding dimension = 256
        attention heads    = 8
        HCSAM blocks       = 6

    The module explicitly performs adjacent-scale interaction:

        Es -> Em
        Em -> El
    """

    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        num_blocks=6,
        dropout=0.1
    ):
        super().__init__()

        self.embed_dim = embed_dim

        self.num_heads = num_heads

        self.num_blocks = num_blocks

        # ----------------------------------------------------
        # Stack HCSAM blocks
        # ----------------------------------------------------

        self.blocks = nn.ModuleList(
            [
                HCSAMBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=dropout
                )

                for _ in range(
                    num_blocks
                )
            ]
        )

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        multi_scale_tokens,
        return_attention=False
    ):
        """
        Parameters
        ----------
        multi_scale_tokens : tuple/list

            Expected:

                (
                    E_s,
                    E_m,
                    E_l
                )

            where:

                E_s = [B, 784, 256]

                E_m = [B, 196, 256]

                E_l = [B, 49, 256]

        Returns
        -------
        tuple

            Refined:

                (
                    E_s,
                    E_m,
                    E_l
                )
        """

        if not isinstance(
            multi_scale_tokens,
            (tuple, list)
        ):

            raise TypeError(
                "HCSAM expects a tuple or list "
                "containing E_s, E_m and E_l."
            )

        if len(
            multi_scale_tokens
        ) != 3:

            raise ValueError(
                "HCSAM requires exactly three "
                "scale representations."
            )

        E_s, E_m, E_l = (
            multi_scale_tokens
        )

        # ----------------------------------------------------
        # Validate embedding dimensions
        # ----------------------------------------------------

        for name, tensor in [
            ("E_s", E_s),
            ("E_m", E_m),
            ("E_l", E_l)
        ]:

            if tensor.ndim != 3:

                raise ValueError(
                    f"{name} must have shape "
                    "[B, N, D]."
                )

            if tensor.shape[-1] != self.embed_dim:

                raise ValueError(
                    f"{name} must have embedding "
                    f"dimension {self.embed_dim}."
                )

        # ----------------------------------------------------
        # Store attention maps
        # ----------------------------------------------------

        attention_history = []

        # ====================================================
        # Sequential HCSAM blocks
        # ====================================================

        for block in self.blocks:

            if return_attention:

                (
                    E_s,
                    E_m,
                    E_l,
                    attention
                ) = block(
                    E_s,
                    E_m,
                    E_l,
                    return_attention=True
                )

                attention_history.append(
                    attention
                )

            else:

                E_s, E_m, E_l = block(
                    E_s,
                    E_m,
                    E_l
                )

        # ----------------------------------------------------
        # Return
        # ----------------------------------------------------

        if return_attention:

            return (
                E_s,
                E_m,
                E_l,
                attention_history
            )

        return (
            E_s,
            E_m,
            E_l
        )


# ============================================================
# Simple Test
# ============================================================

if __name__ == "__main__":

    batch_size = 2

    embedding_dim = 256

    # --------------------------------------------------------
    # Token representations generated from:
    #
    # 224 / 8  = 28 ? 784
    # 224 / 16 = 14 ? 196
    # 224 / 32 = 7  ? 49
    # --------------------------------------------------------

    E_s = torch.randn(
        batch_size,
        784,
        embedding_dim
    )

    E_m = torch.randn(
        batch_size,
        196,
        embedding_dim
    )

    E_l = torch.randn(
        batch_size,
        49,
        embedding_dim
    )

    # --------------------------------------------------------
    # Create HCSAM
    # --------------------------------------------------------

    hcsam = HCSAM(
        embed_dim=256,
        num_heads=8,
        num_blocks=6,
        dropout=0.1
    )

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    with torch.no_grad():

        (
            E_s_out,
            E_m_out,
            E_l_out,
            attention
        ) = hcsam(
            (
                E_s,
                E_m,
                E_l
            ),
            return_attention=True
        )

    # --------------------------------------------------------
    # Verify dimensions
    # --------------------------------------------------------

    print(
        "Input small:",
        E_s.shape
    )

    print(
        "Output small:",
        E_s_out.shape
    )

    print(
        "Input medium:",
        E_m.shape
    )

    print(
        "Output medium:",
        E_m_out.shape
    )

    print(
        "Input large:",
        E_l.shape
    )

    print(
        "Output large:",
        E_l_out.shape
    )

    # --------------------------------------------------------
    # Attention dimensions
    # --------------------------------------------------------

    print(
        "\nSmall -> Medium attention:",
        attention[0][
            "small_to_medium"
        ].shape
    )

    print(
        "Medium -> Large attention:",
        attention[0][
            "medium_to_large"
        ].shape
    )

    # --------------------------------------------------------
    # Expected:
    #
    # Small -> Medium:
    # [B, 8, 784, 196]
    #
    # Medium -> Large:
    # [B, 8, 196, 49]
    # --------------------------------------------------------