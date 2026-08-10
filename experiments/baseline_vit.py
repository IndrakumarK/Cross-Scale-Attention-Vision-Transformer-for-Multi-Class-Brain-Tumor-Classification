"""
Baseline Vision Transformer for CSA-ViT Ablation Study
=======================================================

This implementation corresponds to the "Baseline Vision Transformer"
configuration in the CSA-ViT ablation study.

Architecture:
    Multi-sequence MRI
        ?
    CNN Stem
        ?
    Single-scale 16×16 Patch Embedding
        ?
    Standard Transformer Encoder
        ?
    Attention Pooling
        ?
    Classification Head

Excluded:
    - Multi-scale patch embedding
    - HCSAM
    - SAM
    - Cross-scale attention

Input:
    [B, 4, 224, 224]

Output:
    Four-class prediction:
        0 = Healthy
        1 = Glioma
        2 = Meningioma
        3 = Pituitary
"""

import torch
import torch.nn as nn


# ============================================================
# Baseline Vision Transformer
# ============================================================

class BaselineViT(nn.Module):

    def __init__(
        self,
        in_channels=4,
        image_size=224,
        patch_size=16,
        stem_channels=(64, 128),
        embed_dim=256,
        num_heads=8,
        num_layers=6,
        mlp_ratio=4.0,
        dropout=0.1,
        num_classes=4
    ):
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # ----------------------------------------------------
        # 1. CNN Stem
        # ----------------------------------------------------

        self.cnn_stem = nn.Sequential(

            nn.Conv2d(
                in_channels,
                stem_channels[0],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                stem_channels[0]
            ),

            nn.GELU(),

            nn.Conv2d(
                stem_channels[0],
                stem_channels[1],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                stem_channels[1]
            ),

            nn.GELU()
        )

        # ----------------------------------------------------
        # 2. Single-scale Patch Embedding
        #
        # 224 / 16 = 14
        #
        # Token grid = 14 × 14
        # Number of tokens = 196
        # ----------------------------------------------------

        self.patch_embedding = nn.Conv2d(
            stem_channels[1],
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        num_patches = (
            image_size // patch_size
        ) ** 2

        # ----------------------------------------------------
        # 3. Positional Embedding
        # ----------------------------------------------------

        self.positional_embedding = nn.Parameter(
            torch.zeros(
                1,
                num_patches,
                embed_dim
            )
        )

        # ----------------------------------------------------
        # 4. Transformer Encoder
        # ----------------------------------------------------

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(
                embed_dim * mlp_ratio
            ),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # ----------------------------------------------------
        # 5. Attention-based Token Pooling
        # ----------------------------------------------------

        self.token_attention = nn.Linear(
            embed_dim,
            1
        )

        # ----------------------------------------------------
        # 6. Classification Head
        # ----------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Linear(
                embed_dim,
                embed_dim
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                embed_dim,
                num_classes
            )
        )

        self._initialize_weights()

    # ========================================================
    # Weight Initialization
    # ========================================================

    def _initialize_weights(self):

        nn.init.trunc_normal_(
            self.positional_embedding,
            std=0.02
        )

        nn.init.xavier_uniform_(
            self.token_attention.weight
        )

        nn.init.zeros_(
            self.token_attention.bias
        )

    # ========================================================
    # Forward Pass
    # ========================================================

    def forward(self, x):

        # ----------------------------------------------------
        # Input
        #
        # [B, 4, 224, 224]
        # ----------------------------------------------------

        if x.ndim != 4:

            raise ValueError(
                "Expected input shape "
                "[B, C, H, W]."
            )

        # ----------------------------------------------------
        # CNN Stem
        #
        # [B, 4, 224, 224]
        #          ?
        # [B, 128, 224, 224]
        # ----------------------------------------------------

        x = self.cnn_stem(x)

        # ----------------------------------------------------
        # Patch Embedding
        #
        # [B, 128, 224, 224]
        #          ?
        # [B, 256, 14, 14]
        # ----------------------------------------------------

        x = self.patch_embedding(x)

        # ----------------------------------------------------
        # Convert feature map to tokens
        #
        # [B, 256, 14, 14]
        #          ?
        # [B, 196, 256]
        # ----------------------------------------------------

        x = x.flatten(
            2
        ).transpose(
            1,
            2
        )

        # ----------------------------------------------------
        # Positional encoding
        # ----------------------------------------------------

        x = (
            x +
            self.positional_embedding
        )

        # ----------------------------------------------------
        # Standard Transformer
        # ----------------------------------------------------

        x = self.transformer(x)

        # ----------------------------------------------------
        # Attention-based token aggregation
        # ----------------------------------------------------

        attention_scores = (
            self.token_attention(x)
        )

        attention_weights = torch.softmax(
            attention_scores,
            dim=1
        )

        # ----------------------------------------------------
        # Weighted global representation
        #
        # [B, 196, 256]
        #          ?
        # [B, 256]
        # ----------------------------------------------------

        global_feature = torch.sum(
            attention_weights * x,
            dim=1
        )

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        logits = self.classifier(
            global_feature
        )

        return {
            "logits": logits,
            "features": global_feature,
            "attention": attention_weights
        }


# ============================================================
# Simple Test
# ============================================================

if __name__ == "__main__":

    model = BaselineViT()

    x = torch.randn(
        2,
        4,
        224,
        224
    )

    output = model(x)

    print(
        "Input shape:",
        x.shape
    )

    print(
        "Logits shape:",
        output["logits"].shape
    )

    print(
        "Feature shape:",
        output["features"].shape
    )

    print(
        "Attention shape:",
        output["attention"].shape
    )

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        "Trainable parameters:",
        f"{total_parameters:,}"
    )