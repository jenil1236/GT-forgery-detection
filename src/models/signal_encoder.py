"""
signal_encoder.py - Signal expert encoder.

Processes 31-channel input (30 SRM filters + 1 FFT magnitude).
ResNet-18-like architecture with non-trainable SRM first layer.
Outputs L2-normalised 128-d embeddings.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.transforms import SRM_FILTERS


class SRMLayer(nn.Module):
    """
    Non-trainable first layer that applies 30 SRM filters.
    Input:  (B, 3, H, W) RGB image tensor
    Output: (B, 30, H, W) SRM response
    """

    def __init__(self):
        super().__init__()
        # Average the 3 channels to grayscale first, then apply SRM
        # SRM filters: (30, 1, 5, 5)
        srm = torch.tensor(SRM_FILTERS, dtype=torch.float32)
        self.register_buffer("srm_filters", srm, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) -- already in normalised range
        # Convert to grayscale: weighted average
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]  # (B,1,H,W)
        out = F.conv2d(gray, self.srm_filters, padding=2)  # (B,30,H,W)
        return torch.tanh(out / 10.0)


class _ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class _ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = _ConvBnRelu(channels, channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.conv2(self.conv1(x)))


class SignalEncoder(nn.Module):
    """
    Signal modality encoder.

    Architecture:
        SRM layer (non-trainable): RGB -> 30-ch SRM response
        FFT channel from transform is concatenated -> 31-ch total
        Conv stem: 31 -> 64 channels
        4 x ResBlock + strided conv
        GlobalAvgPool -> Linear(256, embed_dim) -> L2 normalise

    Input shape: (B, 31, H, W)
      Channels: [:30] = SRM residuals (from transform), [30] = FFT magnitude
    """

    def __init__(self, embed_dim: int = 128, in_channels: int = 31):
        super().__init__()
        self.embed_dim = embed_dim

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        # Residual blocks with progressive downsampling
        self.layer1 = nn.Sequential(_ResBlock(64), _ResBlock(64))
        self.down1 = _ConvBnRelu(64, 128, stride=2)
        self.layer2 = nn.Sequential(_ResBlock(128), _ResBlock(128))
        self.down2 = _ConvBnRelu(128, 256, stride=2)
        self.layer3 = nn.Sequential(_ResBlock(256))

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.projector = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 31, H, W) - SRM + FFT channels from SignalTransform
        Returns:
            embedding: (B, embed_dim) L2-normalised
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.down1(x)
        x = self.layer2(x)
        x = self.down2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        emb = self.projector(x)
        return F.normalize(emb, dim=-1)

    def get_embedding_dim(self) -> int:
        return self.embed_dim