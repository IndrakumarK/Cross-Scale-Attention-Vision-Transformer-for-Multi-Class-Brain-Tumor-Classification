"""
CSA-ViT
=======

Cross-Scale Attention Vision Transformer for Multi-Class
Brain Tumor Classification Using Multi-Sequence MRI.

Architecture
------------

T1 -----+
T1CE ---¦
T2 -----¦ ? CNN Stem ? Multi-Scale Patch Embedding
FLAIR --+                         ¦
                                  ?
                         HCSAM Cross-Scale Attention
                                  ¦
                                  ?
                         Adaptive Sequence Attention
                                  ¦
                                  ?
                         Transformer Encoder
                                  ¦
                                  ?
                         Attention Pooling
                                  ¦
                                  ?
                         Classification Head
                                  ¦
                                  ?
              Healthy / Glioma / Meningioma / Pituitary
"""

import torch
import torch.nn as nn

from .cnn_stem import CNNStem
from .multiscale_patch_embedding import MultiScalePatchEmbedding
from .hcsam import HCSAM
from .sequence_attention import SequenceAttention
from .transformer_encoder import TransformerEncoder
from .attention_pooling import AttentionPooling


# ============================================================
# CSA-ViT
# ============================================================

class CSAViT(nn.Module):
    """
    Cross-Scale Attention Vision Transformer.

    Parameters
    ----------
    in_channels : int
        Number of MRI input channels.

        Default:
            4 = T1, T1CE, T2, FLAIR

    stem_channels : tuple
        CNN stem channel configuration.

        Default:
            (64, 128)

    embed_dim : int
        Common embedding dimension.

        Default:
            256

    patch_sizes : tuple
        Hierarchical patch sizes.

        Default:
            (8, 16, 32)

    hcsam_blocks : int
        Number of HCSAM blocks.

        Default:
            6

    heads : int
        Number of attention heads.

        Default:
            8

    transformer_layers : int
        Number of transformer encoder layers.

        Default:
            6

    dropout : float
        Dropout probability.

        Default:
            0.1

    num_classes : int
        Number of diagnostic categories.

        Default:
            4
    """

    def __init__(
        self,
        in_channels=4,
        stem_channels=(64, 128),
        embed_dim=256,
        patch_sizes=(8, 16, 32),
        hcsam_blocks=6,
        heads=8,
        transformer_layers=6,
        dropout=0.1,
        num_classes=4
    ):
        super().__init__()

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_sizes = patch_sizes
        self.num_classes = num_classes

        # ====================================================
        # 1. CNN Stem
        # ====================================================

        self.cnn_stem = CNNStem(
            in_channels=in_channels,
            channels=stem_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

        stem_output_channels = stem_channels[-1]

        # ====================================================
        # 2. Hierarchical Multi-Scale Patch Embedding
        # ====================================================

        self.patch_embedding = MultiScalePatchEmbedding(
            in_channels=stem_output_channels,
            embed_dim=embed_dim,
            patch_sizes=patch_sizes
        )

        # ====================================================
        # 3. HCSAM
        # ====================================================

        self.hcsam = HCSAM(
            embed_dim=embed_dim,
            num_heads=heads,
            num_blocks=hcsam_blocks,
            dropout=dropout
        )

        # ====================================================
        # 4. Sequence Attention Module
        # ====================================================

        self.sequence_attention = SequenceAttention(
            embedding_dim=embed_dim,
            num_modalities=4,
            dropout=dropout
        )

        # ====================================================
        # 5. Transformer Encoder
        # ====================================================

        self.transformer = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=heads,
            num_layers=transformer_layers,
            dropout=dropout
        )

        # ====================================================
        # 6. Attention-Based Feature Aggregation
        # ====================================================

        self.attention_pooling = AttentionPooling(
            embedding_dim=embed_dim
        )

        # ====================================================
        # 7. Classification Head
        # ====================================================

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

        # ====================================================
        # Initialize classifier
        # ====================================================

        self._initialize_classifier()

    # ========================================================
    # Classifier Initialization
    # ========================================================

    def _initialize_classifier(self):

        for module in self.classifier:

            if isinstance(
                module,
                nn.Linear
            ):

                nn.init.xavier_uniform_(
                    module.weight
                )

                if module.bias is not None:

                    nn.init.zeros_(
                        module.bias
                    )

    # ========================================================
    # Process One MRI Sequence
    # ========================================================

    def _process_modality(
        self,
        x
    ):
        """
        Process one MRI modality through:

            CNN Stem
                ?
            Multi-scale patch embedding
                ?
            HCSAM

        Parameters
        ----------
        x : torch.Tensor

            Shape:
                [B, 1, H, W]

        Returns
        -------
        feature : torch.Tensor
            Refined feature representation.

        """

        # ----------------------------------------------------
        # CNN local feature extraction
        # ----------------------------------------------------

        feature_map = self.cnn_stem(
            x
        )

        # ----------------------------------------------------
        # Multi-scale token generation
        # ----------------------------------------------------

        multi_scale_tokens = (
            self.patch_embedding(
                feature_map
            )
        )

        # ----------------------------------------------------
        # HCSAM cross-scale refinement
        # ----------------------------------------------------

        refined = self.hcsam(
            multi_scale_tokens
        )

        return refined

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        x
    ):
        """
        Forward pass.

        Input
        -----
        x:

            [B, 4, 224, 224]

        Channel order:

            0 -> T1
            1 -> T1CE
            2 -> T2
            3 -> FLAIR

        Output
        ------
        Dictionary containing:

            logits
            probabilities
            features
            token_attention
            sequence_attention
            cross_scale_attention
        """

        # ====================================================
        # Input validation
        # ====================================================

        if x.ndim != 4:

            raise ValueError(
                "CSA-ViT expects input with shape "
                "[B, C, H, W]."
            )

        if x.shape[1] != self.in_channels:

            raise ValueError(
                f"Expected {self.in_channels} input channels, "
                f"but received {x.shape[1]}."
            )

        # ====================================================
        # Split MRI modalities
        # ====================================================

        modality_features = []

        for modality_index in range(4):

            modality = x[
                :,
                modality_index:modality_index + 1,
                :,
                :
            ]

            feature = self._process_modality(
                modality
            )

            modality_features.append(
                feature
            )

        # ====================================================
        # Adaptive Sequence Attention
        # ====================================================

        fused_features, sequence_weights = (
            self.sequence_attention(
                modality_features
            )
        )

        # ====================================================
        # Transformer Global Context Modeling
        # ====================================================

        transformer_features = self.transformer(
            fused_features
        )

        # ====================================================
        # Attention-Based Feature Aggregation
        # ====================================================

        global_feature, token_attention = (
            self.attention_pooling(
                transformer_features
            )
        )

        # ====================================================
        # Classification
        # ====================================================

        logits = self.classifier(
            global_feature
        )

        probabilities = torch.softmax(
            logits,
            dim=-1
        )

        # ====================================================
        # Return model outputs
        # ====================================================

        return {
            "logits": logits,

            "probabilities": probabilities,

            "features": global_feature,

            "token_attention": token_attention,

            "sequence_attention": sequence_weights,

            "cross_scale_attention": None
        }


# ============================================================
# Model Parameter Count
# ============================================================

def count_parameters(
    model
):
    """
    Return the number of trainable parameters.
    """

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


# ============================================================
# Model Summary
# ============================================================

def model_summary(
    model
):
    """
    Print basic model information.
    """

    parameters = count_parameters(
        model
    )

    print(
        "\nCSA-ViT Model Summary"
    )

    print(
        "=" * 60
    )

    print(
        f"Trainable parameters: "
        f"{parameters:,}"
    )

    print(
        f"Embedding dimension: "
        f"{model.embed_dim}"
    )

    print(
        f"Patch sizes: "
        f"{model.patch_sizes}"
    )

    print(
        f"Number of classes: "
        f"{model.num_classes}"
    )

    print(
        "=" * 60
    )


# ============================================================
# Simple Test
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Example multi-sequence MRI input
    # --------------------------------------------------------

    x = torch.randn(
        2,
        4,
        224,
        224
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = CSAViT(
        in_channels=4,
        stem_channels=(64, 128),
        embed_dim=256,
        patch_sizes=(8, 16, 32),
        hcsam_blocks=6,
        heads=8,
        transformer_layers=6,
        dropout=0.1,
        num_classes=4
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    model_summary(
        model
    )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(
            x
        )

    print(
        "\nInput shape:",
        x.shape
    )

    print(
        "Logits shape:",
        output["logits"].shape
    )

    print(
        "Probabilities shape:",
        output["probabilities"].shape
    )

    print(
        "Global feature shape:",
        output["features"].shape
    )

    print(
        "Sequence attention shape:",
        output[
            "sequence_attention"
        ].shape
    )

    print(
        "Token attention shape:",
        output[
            "token_attention"
        ].shape
    )