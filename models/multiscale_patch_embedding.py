"""
Hierarchical Multi-Scale Patch Embedding
=========================================

Implements Section 3.4 of the CSA-ViT manuscript.

For a 224 x 224 feature map:

    Patch size    Token grid    Tokens
    -----------------------------------
    8 x 8         28 x 28       784
    16 x 16       14 x 14       196
    32 x 32       7 x 7         49

Each patch scale is projected into the common embedding
dimension of 256 so that the resulting representations can
be processed by HCSAM.

Output:

    E_s = [B, 784, 256]
    E_m = [B, 196, 256]
    E_l = [B, 49, 256]
"""

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    Patch embedding for one spatial resolution.

    A convolution with kernel_size = patch_size and
    stride = patch_size performs non-overlapping patch
    extraction and linear projection simultaneously.
    """

    def __init__(
        self,
        in_channels,
        embed_dim,
        patch_size
    ):
        super().__init__()

        self.patch_size = patch_size

        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        """
        Input:
            [B, C, H, W]

        Output:
            [B, N, D]
        """

        x = self.projection(x)

        # [B, D, H', W']
        #
        # -> [B, D, N]
        #
        # -> [B, N, D]

        x = x.flatten(
            2
        ).transpose(
            1,
            2
        )

        return x


class MultiScalePatchEmbedding(nn.Module):
    """
    Hierarchical multi-scale patch embedding.

    Default configuration:

        Input channels = 128
        Embedding dim  = 256
        Patch sizes    = 8, 16, 32

    The spatial resolution of the CNN stem is assumed to be
    preserved at 224 x 224.
    """

    def __init__(
        self,
        in_channels=128,
        embed_dim=256,
        patch_sizes=(8, 16, 32),
        image_size=224
    ):
        super().__init__()

        if len(patch_sizes) != 3:
            raise ValueError(
                "MultiScalePatchEmbedding requires exactly "
                "three patch sizes."
            )

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_sizes = patch_sizes
        self.image_size = image_size

        # ----------------------------------------------------
        # Verify that patches divide the input resolution
        # ----------------------------------------------------

        for patch_size in patch_sizes:

            if image_size % patch_size != 0:

                raise ValueError(
                    f"Patch size {patch_size} does not "
                    f"divide image size {image_size}."
                )

        # ----------------------------------------------------
        # Small-scale embedding: 8 x 8
        # ----------------------------------------------------

        self.small_embedding = PatchEmbedding(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_sizes[0]
        )

        # ----------------------------------------------------
        # Medium-scale embedding: 16 x 16
        # ----------------------------------------------------

        self.medium_embedding = PatchEmbedding(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_sizes[1]
        )

        # ----------------------------------------------------
        # Large-scale embedding: 32 x 32
        # ----------------------------------------------------

        self.large_embedding = PatchEmbedding(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_sizes[2]
        )

        # ----------------------------------------------------
        # Learnable positional embeddings
        #
        # One positional embedding for each resolution.
        # ----------------------------------------------------

        self.pos_small = nn.Parameter(
            torch.zeros(
                1,
                self._num_tokens(
                    patch_sizes[0]
                ),
                embed_dim
            )
        )

        self.pos_medium = nn.Parameter(
            torch.zeros(
                1,
                self._num_tokens(
                    patch_sizes[1]
                ),
                embed_dim
            )
        )

        self.pos_large = nn.Parameter(
            torch.zeros(
                1,
                self._num_tokens(
                    patch_sizes[2]
                ),
                embed_dim
            )
        )

        self._initialize_weights()

    # ========================================================
    # Number of Tokens
    # ========================================================

    def _num_tokens(
        self,
        patch_size
    ):

        grid_size = (
            self.image_size //
            patch_size
        )

        return grid_size * grid_size

    # ========================================================
    # Weight Initialization
    # ========================================================

    def _initialize_weights(self):

        nn.init.trunc_normal_(
            self.pos_small,
            std=0.02
        )

        nn.init.trunc_normal_(
            self.pos_medium,
            std=0.02
        )

        nn.init.trunc_normal_(
            self.pos_large,
            std=0.02
        )

    # ========================================================
    # Forward
    # ========================================================

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor

            CNN feature map.

            Expected shape:

                [B, 128, 224, 224]

        Returns
        -------
        tuple

            E_s:
                [B, 784, 256]

            E_m:
                [B, 196, 256]

            E_l:
                [B, 49, 256]
        """

        if x.ndim != 4:

            raise ValueError(
                "Expected input with shape "
                "[B, C, H, W]."
            )

        if x.shape[1] != self.in_channels:

            raise ValueError(
                f"Expected {self.in_channels} input "
                f"channels, but received {x.shape[1]}."
            )

        if x.shape[2] != self.image_size:

            raise ValueError(
                f"Expected spatial height "
                f"{self.image_size}, but received "
                f"{x.shape[2]}."
            )

        if x.shape[3] != self.image_size:

            raise ValueError(
                f"Expected spatial width "
                f"{self.image_size}, but received "
                f"{x.shape[3]}."
            )

        # ====================================================
        # Small scale: 8 x 8
        # ====================================================

        E_s = self.small_embedding(
            x
        )

        E_s = (
            E_s +
            self.pos_small
        )

        # ====================================================
        # Medium scale: 16 x 16
        # ====================================================

        E_m = self.medium_embedding(
            x
        )

        E_m = (
            E_m +
            self.pos_medium
        )

        # ====================================================
        # Large scale: 32 x 32
        # ====================================================

        E_l = self.large_embedding(
            x
        )

        E_l = (
            E_l +
            self.pos_large
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

    # --------------------------------------------------------
    # Example CNN output
    # --------------------------------------------------------

    x = torch.randn(
        2,
        128,
        224,
        224
    )

    # --------------------------------------------------------
    # Create multi-scale embedding
    # --------------------------------------------------------

    model = MultiScalePatchEmbedding(
        in_channels=128,
        embed_dim=256,
        patch_sizes=(8, 16, 32),
        image_size=224
    )

    # --------------------------------------------------------
    # Generate tokens
    # --------------------------------------------------------

    with torch.no_grad():

        E_s, E_m, E_l = model(
            x
        )

    # --------------------------------------------------------
    # Display dimensions
    # --------------------------------------------------------

    print(
        "Input:",
        x.shape
    )

    print(
        "Small-scale tokens:",
        E_s.shape
    )

    print(
        "Medium-scale tokens:",
        E_m.shape
    )

    print(
        "Large-scale tokens:",
        E_l.shape
    )

    # --------------------------------------------------------
    # Expected:
    #
    # Small:
    #   [2, 784, 256]
    #
    # Medium:
    #   [2, 196, 256]
    #
    # Large:
    #   [2, 49, 256]
    # --------------------------------------------------------