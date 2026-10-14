"""
text_encoder.py - Text expert encoder.

Uses microsoft/trocr-small-printed to extract text features, combined
with a spatial layout histogram for positional structure.
Features are cached to disk to avoid re-running TrOCR every epoch.
"""

from __future__ import annotations
import hashlib
import os
import pickle
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


def _get_image_hash(img: Image.Image) -> str:
    arr = np.array(img.resize((64, 64)).convert("L"))
    return hashlib.md5(arr.tobytes()).hexdigest()[:16]


class TrOCRFeatureExtractor(nn.Module):
    """
    Extracts text features using TrOCR encoder.
    The decoder is NOT used — we only need encoder representations.
    """

    def __init__(self, model_name: str = "microsoft/trocr-small-printed",
                 cache_dir: Optional[str] = None):
        super().__init__()
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._model = None
        self._processor = None
        self._trocr_dim = 256  # trocr-small hidden size

    def _load(self) -> None:
        """Lazy load TrOCR to avoid long import at module level."""
        if self._model is not None:
            return
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        print(f"  Loading TrOCR: {self.model_name}")
        kwargs = {}
        if self.cache_dir:
            kwargs["cache_dir"] = self.cache_dir
        self._processor = TrOCRProcessor.from_pretrained(self.model_name, **kwargs)
        model = VisionEncoderDecoderModel.from_pretrained(self.model_name, **kwargs)
        self._model = model.encoder  # only the ViT encoder
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad = False
        print(f"  TrOCR encoder loaded. Hidden dim: {self._trocr_dim}")

    @torch.no_grad()
    def extract(self, img: Image.Image, device: str = "cpu") -> torch.Tensor:
        """
        Extract mean-pooled encoder hidden states.
        Returns: (trocr_dim,) tensor
        """
        self._load()
        self._model.to(device)
        pixel_values = self._processor(images=img, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)
        outputs = self._model(pixel_values=pixel_values)
        # Mean pool over sequence: (1, seq_len, hidden) -> (hidden,)
        hidden = outputs.last_hidden_state.squeeze(0).mean(0)
        return hidden.cpu()


class LayoutFeatureExtractor:
    """
    Extracts an 8x8 spatial layout feature from a grayscale document image.
    Computes text density (fraction of dark pixels) per grid cell.
    Output: 64-d vector.
    """

    def __init__(self, grid_size: int = 8):
        self.grid_size = grid_size

    def extract(self, img: Image.Image) -> torch.Tensor:
        gray = img.convert("L")
        W, H = gray.size
        arr = np.array(gray)

        # Text pixel = dark pixel (< 128)
        binary = (arr < 128).astype(np.float32)
        cell_h = H // self.grid_size
        cell_w = W // self.grid_size

        features = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                y1, y2 = i * cell_h, (i + 1) * cell_h
                x1, x2 = j * cell_w, (j + 1) * cell_w
                cell = binary[y1:y2, x1:x2]
                density = cell.mean() if cell.size > 0 else 0.0
                features.append(density)

        return torch.tensor(features, dtype=torch.float32)  # (64,)


class TextEncoder(nn.Module):
    """
    Text modality encoder.

    Combines TrOCR encoder features (256-d) with layout features (64-d).
    Projects to embed_dim with a linear layer.

    Features are cached to features_cache_dir to speed up subsequent epochs.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        model_name: str = "microsoft/trocr-small-printed",
        features_cache_dir: Optional[str] = None,
        models_cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.features_cache_dir = features_cache_dir
        if features_cache_dir:
            os.makedirs(features_cache_dir, exist_ok=True)

        self.trocr = TrOCRFeatureExtractor(
            model_name=model_name,
            cache_dir=models_cache_dir,
        )
        self.layout = LayoutFeatureExtractor(grid_size=8)

        trocr_dim = 256   # trocr-small hidden size
        layout_dim = 64
        combined_dim = trocr_dim + layout_dim

        # Trainable projection
        self.projector = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, embed_dim),
        )
        # Fallback CNN for when image tensors are passed directly (no TrOCR cache)
        # FIX: register as nn.Module so it participates in .to(device) and state_dict()
        self._image_conv: nn.Module = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=4, padding=3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 16, 320),
        )

    def _get_cache_path(self, img_hash: str) -> Optional[str]:
        if not self.features_cache_dir:
            return None
        return os.path.join(self.features_cache_dir, f"{img_hash}.pkl")

    def _try_load_cache(self, img_hash: str) -> Optional[torch.Tensor]:
        path = self._get_cache_path(img_hash)
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        return None

    def _save_cache(self, img_hash: str, feats: torch.Tensor) -> None:
        path = self._get_cache_path(img_hash)
        if path:
            with open(path, "wb") as f:
                pickle.dump(feats, f)

    def extract_raw_features(
        self,
        imgs: list,  # list of PIL Images
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        Extract raw TrOCR + layout features for a list of PIL images.
        Returns: (B, trocr_dim + layout_dim) tensor, uses cache.
        """
        all_feats = []
        for img in imgs:
            img_hash = _get_image_hash(img)
            cached = self._try_load_cache(img_hash)
            if cached is not None:
                all_feats.append(cached)
            else:
                trocr_feat = self.trocr.extract(img, device=device)  # (256,)
                layout_feat = self.layout.extract(img)                 # (64,)
                combined = torch.cat([trocr_feat, layout_feat], dim=0) # (320,)
                self._save_cache(img_hash, combined)
                all_feats.append(combined)

        return torch.stack(all_feats, dim=0)  # (B, 320)

    def forward_from_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass from pre-computed raw feature tensor.
        Args:
            x: (B, 320) raw features
        Returns:
            embedding: (B, embed_dim) L2-normalised
        """
        emb = self.projector(x)
        return F.normalize(emb, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Either (B, 320) pre-extracted features, or (B, 3, H, W) image tensor.
               When image tensor is provided, uses the image-based projection path.
        Returns:
            embedding: (B, embed_dim) L2-normalised
        """
        if x.dim() == 4:
            # Input is an image tensor (B, 3, H, W) from TextTransform
            # Use the image_projector path instead of TrOCR features
            return self.forward_from_image(x)
        return self.forward_from_tensor(x)

    def forward_from_image(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process a (B, 3, H, W) image tensor without TrOCR.
        Uses a lightweight CNN to extract features for training.
        Falls back to this when pre-extracted features are unavailable.
        """
        # FIX: _image_conv is now a registered nn.Module built in __init__,
        # so it moves with .to(device) automatically ? no manual relocation needed.
        feats = self._image_conv(x)  # (B, 320)
        emb = self.projector(feats)
        return F.normalize(emb, dim=-1)

    def get_embedding_dim(self) -> int:
        return self.embed_dim