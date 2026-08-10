"""
CNN Stem for CSA-ViT
====================

Implements the lightweight CNN stem described in Section 3.3
of the CSA-ViT manuscript.

Architecture:

    Input MRI
        ?
    Conv 3×3, stride 1, padding 1
        ?
    BatchNorm
        ?
    GELU
        ?
    Conv 3×3, stride 1, padding 1
        ?
    BatchNorm
        ?
    GELU
        ?
    Local feature representation

The spatial resolution is preserved.

For a 224×224 input:

    [B, C, 224, 224]
            ?
    [B, 64, 224, 224]
            ?
    [B, 128, 224, 224]
"""

import torch
import torch.nn as nn


class CNNStem(nn.Module):
    """
    Lightweight CNN stem for local MRI feature extraction.

    Parameters
    ----------
    in_channels : int
        Number of input channels.

        For multi-sequence MRI:
            4 = T1, T1CE, T2, FLAIR

        If each modality is processed independently:
            1

    channels : tuple
        Number of output channels of the two convolutional
        layers. The manuscript configuration is:

            64 ? 128

    kernel_size : int
        Convolution kernel size.

    stride : int
        Convolution stride.

    padding : int
        Padding used to preserve spatial resolution.
    """

    def __init__(
        self,
        in_channels=4,
        channels=(64, 128),
        kernel_size=3,
        stride=1,
        padding=1
    ):
        super().__init__()

        if len(channels) != 2:
            raise ValueError(
                "CNNStem requires exactly two convolutional "
                "channel values, e.g. (64, 128)."
            )

        # ----------------------------------------------------
        # First convolution
        # ----------------------------------------------------

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=channels[0],
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(
            channels[0]
        )

        self.act1 = nn.GELU()

        # ----------------------------------------------------
        # Second convolution
        # ----------------------------------------------------

        self.conv2 = nn.Conv2d(
            in_channels=channels[0],
            out_channels=channels[1],
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(
            channels[1]
        )

        self.act2 = nn.GELU()

        # ----------------------------------------------------
        # Weight initialization
        # ----------------------------------------------------

        self._initialize_weights()

    # ========================================================
    # Weight Initialization
    # ========================================================

    def _initialize_weights(self):

        for module in self.modules():

            if isinstance(
                module,
                nn.Conv2d
            ):

                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu"
                )

            elif isinstance(
                module,
                nn.BatchNorm2d
            ):

                nn.init.ones_(
                    module.weight
                )

                nn.init.zeros_(
                    module.bias
                )

    # ========================================================
    # Forward
    # ========================================================

    def forward(self, x):

        # ----------------------------------------------------
        # Input validation
        # ----------------------------------------------------

        if x.ndim != 4:

            raise ValueError(
                "CNNStem expects input with shape "
                "[B, C, H, W]."
            )

        # ----------------------------------------------------
        # First convolution
        # ----------------------------------------------------

        x = self.conv1(x)

        x = self.bn1(x)

        x = self.act1(x)

        # ----------------------------------------------------
        # Second convolution
        # ----------------------------------------------------

        x = self.conv2(x)

        x = self.bn2(x)

        x = self.act2(x)

        return x


# ============================================================
# Simple Test
# ============================================================

if __name__ == "__main__":

    # Four MRI sequences
    x = torch.randn(
        2,
        4,
        224,
        224
    )

    model = CNNStem(
        in_channels=4,
        channels=(64, 128)
    )

    output = model(x)

    print(
        "Input shape:",
        x.shape
    )

    print(
        "Output shape:",
        output.shape
    )

    parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        "Trainable parameters:",
        f"{parameters:,}"
    )