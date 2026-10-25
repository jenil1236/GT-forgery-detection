"""
visual_encoder.py - Visual expert encoder.

EfficientNet-B0 backbone with a 2-layer projection MLP.
Outputs L2-normalised 128-d embeddings.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class VisualEncoder(nn.Module):
    """
    Visual modality encoder.

    Architecture:
        EfficientNet-B0 (ImageNet pretrained, classifier removed)
        -> AdaptiveAvgPool
        -> ProjectionMLP: Linear(1280, 512) -> BN -> ReLU -> Linear(512, embed_dim)
        -> L2 normalisation
    """

    def __init__(self, embed_dim: int = 128, freeze_backbone: bool = False):
        super().__init__()
        self.embed_dim = embed_dim

        # Backbone: EfficientNet-B0
        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        # Remove the classifier head, keep features
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        backbone_out_dim = 1280  # EfficientNet-B0 feature dim

        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

        # Projection MLP
        self.projector = nn.Sequential(
            nn.Linear(backbone_out_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) ImageNet-normalised
        Returns:
            embedding: (B, embed_dim) L2-normalised
        """
        feats = self.features(x)         # (B, 1280, H', W')
        feats = self.avgpool(feats)       # (B, 1280, 1, 1)
        feats = torch.flatten(feats, 1)  # (B, 1280)
        emb = self.projector(feats)       # (B, embed_dim)
        return F.normalize(emb, dim=-1)  # L2 normalise

    def get_embedding_dim(self) -> int:
        return self.embed_dim