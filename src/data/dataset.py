"""
dataset.py - PairedDocumentDataset

Each sample:  (reference, test, label)
  reference = a REAL image of the SAME doc_type (different instance)
  test      = real or forged image
  label     = 0 (genuine) | 1 (forged)

Returns a dict with keys:
  ref_visual,  ref_text,  ref_signal    -> reference image, 3 modalities
  test_visual, test_text, test_signal   -> test image, 3 modalities
  label        -> int tensor {0, 1}
  doc_type     -> string

ZeroShotDataset wraps the same logic for held-out document types.
"""

from __future__ import annotations
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image

from src.data.transforms import VisualTransform, TextTransform, SignalTransform


class PairedDocumentDataset(Dataset):
    """
    Builds (reference, test, label) triplets from processed/ directory.

    Directory layout expected:
        processed/{doc_type}/real/{split}/img_*.png
        processed/{doc_type}/forged/{split}/img_*.png
    """

    def __init__(
        self,
        processed_dir: str,
        doc_types: List[str],
        split: str = "train",       # "train" | "test" | "val" | "all"
        image_size: int = 224,
        text_size: int = 384,
        train_mode: bool = True,
        seed: int = 42,
    ):
        self.processed_dir = processed_dir
        self.doc_types = doc_types
        self.split = split
        self.train_mode = train_mode
        self.rng = random.Random(seed)

        # Transforms
        self.visual_t = VisualTransform(size=image_size, train=train_mode)
        self.text_t   = TextTransform(size=text_size)
        self.signal_t = SignalTransform(size=image_size)

        # Build index: list of (real_path, forged_path, doc_type)
        self.real_paths: List[Tuple[str, str]] = []  # (path, doc_type)
        self.samples: List[Tuple[str, int, str]] = []  # (path, label, doc_type)
        self._build_index()

        # Group real images by doc_type for reference selection
        self.real_by_type: Dict[str, List[str]] = {dt: [] for dt in doc_types}
        for path, doc_type in self.real_paths:
            self.real_by_type[doc_type].append(path)

    def _build_index(self) -> None:
        for doc_type in self.doc_types:
            splits_to_load = self._resolve_splits(doc_type)
            for split_dir in splits_to_load:
                real_dir = os.path.join(
                    self.processed_dir, doc_type, "real", split_dir
                )
                forged_dir = os.path.join(
                    self.processed_dir, doc_type, "forged", split_dir
                )
                if not os.path.isdir(real_dir):
                    continue

                real_files = sorted([
                    f for f in os.listdir(real_dir)
                    if f.endswith(".png") and not f.startswith(".")
                ])
                for fname in real_files:
                    real_path = os.path.join(real_dir, fname)
                    self.real_paths.append((real_path, doc_type))
                    self.samples.append((real_path, 0, doc_type))  # genuine

                    if os.path.isdir(forged_dir):
                        forged_path = os.path.join(forged_dir, fname)
                        if os.path.exists(forged_path):
                            self.samples.append((forged_path, 1, doc_type))  # forged

    def _resolve_splits(self, doc_type: str) -> List[str]:
        """Map split name to actual subdirectory names."""
        if self.split == "all":
            return ["train", "test", "val", "all"]
        # cord has train/test/val; rvl_cdip has 'all'
        if doc_type == "cord":
            mapping = {"train": ["train"], "test": ["test"], "val": ["val"]}
        else:
            mapping = {"train": ["all"], "test": ["all"], "val": ["all"]}
        return mapping.get(self.split, [self.split])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        test_path, label, doc_type = self.samples[idx]

        # Select reference: a REAL image from the same doc_type, different from test_path
        candidates = [p for p in self.real_by_type.get(doc_type, []) if p != test_path]
        if not candidates:
            # fallback: use itself as reference (degenerate but won't crash)
            candidates = [test_path]
        ref_path = self.rng.choice(candidates)

        # Load images
        try:
            ref_img = Image.open(ref_path).convert("RGB")
        except Exception:
            ref_img = Image.new("RGB", (224, 224), (200, 200, 200))

        try:
            test_img = Image.open(test_path).convert("RGB")
        except Exception:
            test_img = Image.new("RGB", (224, 224), (200, 200, 200))

        return {
            "ref_visual":  self.visual_t(ref_img),
            "ref_text":    self.text_t(ref_img),
            "ref_signal":  self.signal_t(ref_img),
            "test_visual": self.visual_t(test_img),
            "test_text":   self.text_t(test_img),
            "test_signal": self.signal_t(test_img),
            "label":       torch.tensor(label, dtype=torch.long),
            "doc_type":    doc_type,
        }


class ZeroShotDataset(PairedDocumentDataset):
    """
    Zero-shot evaluation dataset for held-out document types.
    Identical to PairedDocumentDataset but always uses train_mode=False
    and clearly marks samples as zero-shot.
    """

    def __init__(self, processed_dir: str, doc_types: List[str], **kwargs):
        kwargs["train_mode"] = False
        super().__init__(processed_dir, doc_types, **kwargs)

    def __getitem__(self, idx: int) -> Dict:
        sample = super().__getitem__(idx)
        sample["zero_shot"] = torch.tensor(1, dtype=torch.uint8)
        return sample


def build_dataloaders(cfg, processed_dir: str):
    """
    Convenience function to build all DataLoaders from config.
    Returns: train_loader, val_loader, zeroshot_loader
    """
    from torch.utils.data import DataLoader

    train_ds = PairedDocumentDataset(
        processed_dir=processed_dir,
        doc_types=cfg.train_doc_types,
        split="train",
        image_size=cfg.image_size,
        text_size=cfg.text_image_size,
        train_mode=True,
        seed=cfg.seed,
    )
    val_ds = PairedDocumentDataset(
        processed_dir=processed_dir,
        doc_types=cfg.train_doc_types,
        split="test",
        image_size=cfg.image_size,
        text_size=cfg.text_image_size,
        train_mode=False,
        seed=cfg.seed,
    )
    zs_ds = ZeroShotDataset(
        processed_dir=processed_dir,
        doc_types=cfg.zeroshot_doc_types,
        split="all",
        image_size=cfg.image_size,
        text_size=cfg.text_image_size,
        seed=cfg.seed,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.stage1_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device == "cuda"),
    )
    zs_loader = DataLoader(
        zs_ds,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device == "cuda"),
    )

    return train_loader, val_loader, zs_loader