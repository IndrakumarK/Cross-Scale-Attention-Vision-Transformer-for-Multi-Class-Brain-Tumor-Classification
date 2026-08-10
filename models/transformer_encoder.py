"""
Transformer Encoder for CSA-ViT
===============================

Implements Section 3.7 of the CSA-ViT manuscript.

The fused representation from SAM is processed using stacked
Transformer encoder layers to model long-range contextual
relationships among feature tokens.

Configuration used in the manuscript:

    Embedding dimension : 256
    Attention heads     : 8
    Encoder layers      : 6
    Dropout             : 0.1
    Activation          : GELU

The encoder preserves the token sequence dimension.

Input:
    [B, N, 256]

Output:
    [B, N, 256]
"""

import torch
import torch.nn as nn


# ============================================================
# Transformer Encoder Block
# ============================================================

class TransformerEncoderBlock(nn.Module):
    """
    Single Transformer encoder block.

    Implements the pre-normalization form:

        T' = T + MHSA(LN(T))

        T'' = T' + MLP(LN(T'))

    corresponding to Equations (13) and (14) in the manuscript.
    """

    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1
    ):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                "embed_dim must be divisible by num_heads."
            )

        hidden_dim = int(
            embed_dim * mlp_ratio
        )

        # ----------------------------------------------------
        # Layer normalization before self-attention
        # ----------------------------------------------------

        self.norm1 = nn.LayerNorm(
            embed_dim
        )

        # ----------------------------------------------------
        # Multi-head self-attention
        # ----------------------------------------------------

        self.self_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # ----------------------------------------------------
        # Dropout after attention
        # ----------------------------------------------------

        self.dropout1 = nn.Dropout(
            dropout
        )

        # ----------------------------------------------------
        # Layer normalization before MLP
        # ----------------------------------------------------

        self.norm2 = nn.LayerNorm(
            embed_dim
        )

        # ----------------------------------------------------
        # Feed-forward network
        #
        # Linear ? GELU ? Dropout ? Linear
        # ----------------------------------------------------

        self.mlp = nn.Sequential(

            nn.Linear(
                embed_dim,
                hidden_dim
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_dim,
                embed_dim
            ),

            nn.Dropout(
                dropout
            )
        )

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        x,
        return_attention=False
    ):
        """
        Parameters
        ----------
        x : torch.Tensor

            Shape:
                [B, N, D]

        return_attention : bool

            Whether to return self-attention weights.

        Returns
        -------
        x : torch.Tensor

            Shape:
                [B, N, D]

        attention : torch.Tensor, optional

            Self-attention map.
        """

        # ====================================================
        # Multi-Head Self-Attention
        #
        # T' = T + MHSA(LN(T))
        # ====================================================

        normalized_x = self.norm1(
            x
        )

        attention_output, attention_weights = (
            self.self_attention(
                normalized_x,
                normalized_x,
                normalized_x,
                need_weights=return_attention,
                average_attn_weights=False
            )
        )

        x = (
            x +
            self.dropout1(
                attention_output
            )
        )

        # ====================================================
        # Feed-Forward Network
        #
        # T'' = T' + MLP(LN(T'))
        # ====================================================

        x = (
            x +
            self.mlp(
                self.norm2(x)
            )
        )

        if return_attention:

            return (
                x,
                attention_weights
            )

        return x


# ============================================================
# Transformer Encoder
# ============================================================

class TransformerEncoder(nn.Module):
    """
    Stacked Transformer encoder for CSA-ViT.

    Default configuration:

        embed_dim  = 256
        heads      = 8
        layers     = 6
        dropout    = 0.1
    """

    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        num_layers=6,
        mlp_ratio=4.0,
        dropout=0.1
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # ----------------------------------------------------
        # Stacked encoder layers
        # ----------------------------------------------------

        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout
                )
                for _ in range(num_layers)
            ]
        )

        # ----------------------------------------------------
        # Final normalization
        # ----------------------------------------------------

        self.norm = nn.LayerNorm(
            embed_dim
        )

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        x,
        return_attention=False
    ):
        """
        Parameters
        ----------
        x : torch.Tensor

            Input token representation.

            Shape:
                [B, N, 256]

        return_attention : bool

            If True, attention maps from all encoder
            layers are returned.

        Returns
        -------
        x : torch.Tensor

            Contextualized representation.

            Shape:
                [B, N, 256]

        attention_maps : list, optional

            Attention maps from each layer.
        """

        if x.ndim != 3:

            raise ValueError(
                "TransformerEncoder expects input "
                "with shape [B, N, D]."
            )

        if x.shape[-1] != self.embed_dim:

            raise ValueError(
                f"Expected embedding dimension "
                f"{self.embed_dim}, but received "
                f"{x.shape[-1]}."
            )

        attention_maps = []

        # ----------------------------------------------------
        # Sequential Transformer layers
        # ----------------------------------------------------

        for layer in self.layers:

            if return_attention:

                x, attention = layer(
                    x,
                    return_attention=True
                )

                attention_maps.append(
                    attention
                )

            else:

                x = layer(
                    x
                )

        # ----------------------------------------------------
        # Final normalization
        # ----------------------------------------------------

        x = self.norm(
            x
        )

        if return_attention:

            return (
                x,
                attention_maps
            )

        return x


# ============================================================
# Simple Test
# ============================================================

if __name__ == "__main__":

    batch_size = 2
    num_tokens = 1029
    embedding_dim = 256

    # --------------------------------------------------------
    # Example fused token representation
    # --------------------------------------------------------

    x = torch.randn(
        batch_size,
        num_tokens,
        embedding_dim
    )

    # --------------------------------------------------------
    # Create Transformer Encoder
    # --------------------------------------------------------

    model = TransformerEncoder(
        embed_dim=256,
        num_heads=8,
        num_layers=6,
        mlp_ratio=4.0,
        dropout=0.1
    )

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    with torch.no_grad():

        output, attention_maps = model(
            x,
            return_attention=True
        )

    # --------------------------------------------------------
    # Display dimensions
    # --------------------------------------------------------

    print(
        "Input shape:",
        x.shape
    )

    print(
        "Output shape:",
        output.shape
    )

    print(
        "Number of attention layers:",
        len(attention_maps)
    )

    print(
        "Attention shape from first layer:",
        attention_maps[0].shape
    )

    # --------------------------------------------------------
    # Parameter count
    # --------------------------------------------------------

    parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        "Trainable parameters:",
        f"{parameters:,}"
    )