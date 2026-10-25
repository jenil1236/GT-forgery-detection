from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAlignLayer(nn.Module):
    """
    Projects 3 modality feature maps to a unified d_model=256 at G x G resolution.

    Input  per stream: variable channels, variable spatial
    Output per stream: (B, G*G, d_model) = (B, 49, 256) with G=7
    """

    def __init__(self, d_model: int = 256, G: int = 7):
        super().__init__()
        self.d_model = d_model
        self.G = G
        # 1x1 conv per modality: projects channels -> d_model
        self.proj_visual = nn.Conv2d(1280, d_model, kernel_size=1, bias=False)
        self.proj_text   = nn.Conv2d(32,   d_model, kernel_size=1, bias=False)
        self.proj_signal = nn.Conv2d(256,  d_model, kernel_size=1, bias=False)
        self.pool = nn.AdaptiveAvgPool2d((G, G))

    def _align(self, proj: nn.Module, feat: torch.Tensor) -> torch.Tensor:
        # feat: (B, C, H, W)
        x = proj(feat)        # (B, d_model, H, W)
        x = self.pool(x)      # (B, d_model, G, G)
        B, C, H, W = x.shape
        x = x.flatten(2)      # (B, d_model, G*G)
        x = x.permute(0, 2, 1)  # (B, G*G, d_model)
        return x

    def forward(self, feats: dict) -> dict:
        """
        Args: feats dict from FeatureExtractor
        Returns: dict with same keys, values (B, G*G, d_model)
        """
        return {
            "visual_ref":  self._align(self.proj_visual, feats["visual_ref"]),
            "visual_test": self._align(self.proj_visual, feats["visual_test"]),
            "text_ref":    self._align(self.proj_text,   feats["text_ref"]),
            "text_test":   self._align(self.proj_text,   feats["text_test"]),
            "signal_ref":  self._align(self.proj_signal, feats["signal_ref"]),
            "signal_test": self._align(self.proj_signal, feats["signal_test"]),
        }
