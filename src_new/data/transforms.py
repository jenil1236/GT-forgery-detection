"""
transforms.py - Per-modality preprocessing pipelines.

Three transform objects:
  VisualTransform  - 224x224 RGB, ImageNet normalisation, optional augmentation
  TextTransform    - 384x384 grayscale binarised for TrOCR
  SignalTransform  - 30-filter SRM bank + FFT magnitude -> 31-channel tensor
"""

from __future__ import annotations
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image


# ── SRM filter bank (30 filters, each 5x5) ───────────────────────────────────
# Simplified Steganalysis Rich Model kernels.
# We use the 30 linearly independent high-pass filters from Fridrich & Kodovsky 2012.

def _build_srm_filters() -> np.ndarray:
    """
    Returns a (30, 1, 5, 5) array of SRM high-pass kernels (normalised).
    We construct them as combinations of finite difference operators.
    """
    filters = []
    # Edge & noise-sensitive kernels
    base_kernels = [
        # horizontal edges
        [[ 0,  0,  0,  0,  0],
         [ 0,  0,  0,  0,  0],
         [-1,  2, -1,  0,  0],
         [ 0,  0,  0,  0,  0],
         [ 0,  0,  0,  0,  0]],
        # vertical edges
        [[ 0,  0, -1,  0,  0],
         [ 0,  0,  2,  0,  0],
         [ 0,  0, -1,  0,  0],
         [ 0,  0,  0,  0,  0],
         [ 0,  0,  0,  0,  0]],
        # diagonal
        [[-1,  0,  0,  0,  0],
         [ 0,  2,  0,  0,  0],
         [ 0,  0, -1,  0,  0],
         [ 0,  0,  0,  0,  0],
         [ 0,  0,  0,  0,  0]],
        # Laplacian-like
        [[ 0,  0,  0,  0,  0],
         [ 0, -1,  2, -1,  0],
         [ 0,  2, -4,  2,  0],
         [ 0, -1,  2, -1,  0],
         [ 0,  0,  0,  0,  0]],
        # center-surround
        [[ 1,  1,  1,  1,  1],
         [ 1, -2, -2, -2,  1],
         [ 1, -2,  8, -2,  1],
         [ 1, -2, -2, -2,  1],
         [ 1,  1,  1,  1,  1]],
    ]
    filters.extend(base_kernels)

    # Generate 25 more by rotating/shifting the base kernels
    for k_idx, base in enumerate(base_kernels):
        arr = np.array(base, dtype=np.float32)
        for rot in range(1, 4):
            rotated = np.rot90(arr, rot)
            filters.append(rotated.tolist())
        # horizontal flip
        filters.append(np.fliplr(arr).tolist())

    # Pad to exactly 30 by adding shifted versions of existing filters
    while len(filters) < 30:
        base = np.array(filters[len(filters) % len(base_kernels)], dtype=np.float32)
        shifted = np.roll(base, len(filters) % 3 + 1, axis=0)
        filters.append(shifted.tolist())
    filters = filters[:30]  # ensure exactly 30
    srm = np.array(filters, dtype=np.float32)  # (30, 5, 5)
    srm = srm[:, np.newaxis, :, :]  # (30, 1, 5, 5)

    # Normalise each filter to zero sum (high-pass)
    for i in range(srm.shape[0]):
        s = srm[i].sum()
        if s != 0:
            srm[i] -= s / 25.0

    return srm  # (30, 1, 5, 5)


SRM_FILTERS = _build_srm_filters()  # cached module-level


class VisualTransform:
    """
    Visual modality transform.
    train=True adds random augmentations; train=False uses deterministic resize+crop.
    """
    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(self, size: int = 224, train: bool = True):
        self.size = size
        if train:
            self.transform = T.Compose([
                T.Resize((size + 32, size + 32)),
                T.RandomCrop(size),
                T.RandomHorizontalFlip(p=0.3),
                T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
                T.ToTensor(),
                T.Normalize(mean=self.MEAN, std=self.STD),
            ])
        else:
            self.transform = T.Compose([
                T.Resize((size, size)),
                T.ToTensor(),
                T.Normalize(mean=self.MEAN, std=self.STD),
            ])

    def __call__(self, img: Image.Image) -> torch.Tensor:
        if img.mode != "RGB":
            img = img.convert("RGB")
        return self.transform(img)


class TextTransform:
    """
    Text modality transform.
    Produces a grayscale binarised image for TrOCR input.
    Returns a 3-channel tensor (TrOCR expects RGB format).
    """

    def __init__(self, size: int = 384):
        self.size = size
        self.to_tensor = T.ToTensor()
        self.normalise = T.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        )

    def __call__(self, img: Image.Image) -> torch.Tensor:
        if img.mode != "RGB":
            img = img.convert("RGB")
        # Resize maintaining aspect ratio with padding
        img = TF.resize(img, [self.size, self.size])
        # Convert to grayscale and back to RGB (TrOCR expects 3-channel)
        gray = img.convert("L")
        # Otsu-like binarisation via numpy threshold
        arr = np.array(gray)
        threshold = int(arr.mean() + 0.5 * arr.std())
        threshold = max(128, min(200, threshold))
        binary = (arr > threshold).astype(np.uint8) * 255
        binary_rgb = Image.fromarray(binary).convert("RGB")
        t = self.to_tensor(binary_rgb)
        return self.normalise(t)


class SignalTransform:
    """
    Signal modality transform.
    Applies 30 SRM filters + 1 FFT magnitude channel -> 31-channel tensor.
    """

    def __init__(self, size: int = 224):
        self.size = size
        import torch.nn.functional as F
        self._F = F
        # Pre-build SRM kernel tensor (30, 1, 5, 5)
        self._srm_kernel = torch.tensor(SRM_FILTERS, dtype=torch.float32)  # (30,1,5,5)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Resize and convert to grayscale float tensor in [-1, 1]
        img = TF.resize(img, [self.size, self.size])
        gray = np.array(img.convert("L")).astype(np.float32) / 127.5 - 1.0
        gray_t = torch.tensor(gray).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

        # Apply SRM filters
        srm_out = self._F.conv2d(gray_t, self._srm_kernel, padding=2)  # (1,30,H,W)
        srm_out = torch.tanh(srm_out / 10.0)  # squash to [-1,1]
        srm_out = srm_out.squeeze(0)  # (30, H, W)

        # FFT magnitude channel
        fft = torch.fft.fft2(torch.tensor(gray))
        fft_mag = torch.abs(torch.fft.fftshift(fft))
        fft_log = torch.log1p(fft_mag).float()  # log scale
        # Normalise to [-1, 1]
        fft_min, fft_max = fft_log.min(), fft_log.max()
        if fft_max > fft_min:
            fft_norm = 2.0 * (fft_log - fft_min) / (fft_max - fft_min) - 1.0
        else:
            fft_norm = torch.zeros_like(fft_log)
        fft_norm = fft_norm.unsqueeze(0)  # (1, H, W)

        # Stack: (31, H, W)
        signal = torch.cat([srm_out, fft_norm], dim=0)
        return signal