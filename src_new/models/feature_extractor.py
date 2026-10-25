from __future__ import annotations
from typing import Dict, Tuple
import torch
import torch.nn as nn


class FeatureExtractor(nn.Module):
    """
    Wraps the three Stage-1 encoders and taps spatial feature maps
    BEFORE the global-average-pool / projection MLP.

    Visual : (B, 1280, 7, 7)  -- after EfficientNet-B0 features block
    Text   : (B, 32,   H, W)  -- after first conv in _image_conv
    Signal : (B, 256,  7, 7)  -- after layer3 ResBlock, before pool

    All encoder parameters are frozen.
    """

    def __init__(self, visual_enc, text_enc, signal_enc):
        super().__init__()
        self.visual_enc = visual_enc
        self.text_enc   = text_enc
        self.signal_enc = signal_enc
        # Freeze all encoder parameters
        for m in [self.visual_enc, self.text_enc, self.signal_enc]:
            for p in m.parameters():
                p.requires_grad = False

    def _extract_visual(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 224, 224)
        feats = self.visual_enc.features(x)   # (B, 1280, 7, 7)
        return feats

    def _extract_text(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 384, 384)
        # _image_conv: Conv2d(3->32, k=7, s=4) -> ReLU -> AdaptiveAvgPool(4,4) -> Flatten -> Linear
        # Tap after the first conv+relu, before pool
        conv_relu = self.text_enc._image_conv[:2]   # Conv2d + ReLU
        feats = conv_relu(x)   # (B, 32, H, W)
        return feats

    def _extract_signal(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 31, 224, 224)
        se = self.signal_enc
        h = se.stem(x)
        h = se.layer1(h)
        h = se.down1(h)
        h = se.layer2(h)
        h = se.down2(h)
        h = se.layer3(h)   # (B, 256, 7, 7)
        return h

    @torch.no_grad()
    def forward(
        self,
        ref_visual:  torch.Tensor,
        ref_text:    torch.Tensor,
        ref_signal:  torch.Tensor,
        test_visual: torch.Tensor,
        test_text:   torch.Tensor,
        test_signal: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns dict of spatial feature maps:
          visual_ref/test : (B, 1280, 7, 7)
          text_ref/test   : (B, 32, H, W)
          signal_ref/test : (B, 256, 7, 7)
        """
        return {
            "visual_ref":  self._extract_visual(ref_visual),
            "visual_test": self._extract_visual(test_visual),
            "text_ref":    self._extract_text(ref_text),
            "text_test":   self._extract_text(test_text),
            "signal_ref":  self._extract_signal(ref_signal),
            "signal_test": self._extract_signal(test_signal),
        }
