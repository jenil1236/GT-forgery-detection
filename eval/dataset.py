from __future__ import annotations
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.transforms import VisualTransform, TextTransform, SignalTransform

FORGERY_FOLDERS = [
    "fraud1_copy_and_move",
    "fraud2_face_morphing",
    "fraud3_face_replacement",
    "fraud4_combined",
]

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp")


class EvalDataset(Dataset):
    """
    Evaluation dataset for a new IDNet country (e.g. SVK).
    Same pairing as training: each original paired with a different random
    original as reference. All originals used (no train/val split needed).
    """

    def __init__(self, data_dir: str, image_size: int = 224,
                 text_image_size: int = 384, seed: int = 42) -> None:
        self.data_dir  = Path(data_dir)
        self.rng       = random.Random(seed)
        self.visual_tf = VisualTransform(size=image_size, train=False)
        self.text_tf   = TextTransform(size=text_image_size)
        self.signal_tf = SignalTransform(size=image_size)

        pos_dir = self.data_dir / "positive"
        if not pos_dir.is_dir():
            raise FileNotFoundError(f"positive/ not found in {self.data_dir}")

        self.originals: List[str] = sorted(
            f for f in os.listdir(pos_dir) if f.lower().endswith(IMG_EXT)
        )
        if not self.originals:
            raise ValueError(f"No images found in {pos_dir}")

        self.fraud_dirs: Dict[str, Path] = {
            ft: self.data_dir / ft
            for ft in FORGERY_FOLDERS
            if (self.data_dir / ft).is_dir()
        }

        self.samples: List[Tuple[str, int, str, Optional[Path]]] = []
        for fname in self.originals:
            self.samples.append((fname, 0, "genuine", None))
            for ft, fd in self.fraud_dirs.items():
                candidate = fd / fname
                if candidate.exists():
                    self.samples.append((fname, 1, ft, candidate))

        n_g = sum(1 for s in self.samples if s[1] == 0)
        n_f = sum(1 for s in self.samples if s[1] == 1)
        print(f"[EvalDataset] {self.data_dir.name} | originals={len(self.originals)} | "
              f"samples={len(self.samples)} (genuine={n_g}, forged={n_f})")
        for ft in FORGERY_FOLDERS:
            cnt = sum(1 for s in self.samples if s[2] == ft)
            print(f"  {ft:<35s}: {cnt}")
        print()

    def _pick_ref(self, exclude: str) -> str:
        candidates = [f for f in self.originals if f != exclude]
        return self.rng.choice(candidates) if candidates else exclude

    def _load_rgb(self, path: Path) -> Optional[Image.Image]:
        try:
            img = Image.open(path)
            img.load()
            return img.convert("RGB")
        except Exception:
            return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        fname, label, forgery_type, fraud_path = self.samples[idx]
        pos_dir   = self.data_dir / "positive"
        test_path = pos_dir / fname if label == 0 else fraud_path
        test_img  = self._load_rgb(test_path)
        ref_img   = self._load_rgb(pos_dir / self._pick_ref(fname))
        if test_img is None or ref_img is None:
            for off in range(1, len(self.samples)):
                return self.__getitem__((idx + off) % len(self.samples))
        return {
            "ref_visual":  self.visual_tf(ref_img),
            "ref_text":    self.text_tf(ref_img),
            "ref_signal":  self.signal_tf(ref_img),
            "test_visual": self.visual_tf(test_img),
            "test_text":   self.text_tf(test_img),
            "test_signal": self.signal_tf(test_img),
            "label":        torch.tensor(label, dtype=torch.long),
            "forgery_type": forgery_type,
        }


def _collate(batch):
    out = {}
    for k in batch[0]:
        if k == "forgery_type":
            out[k] = [b[k] for b in batch]
        else:
            out[k] = torch.stack([b[k] for b in batch])
    return out


def build_eval_loader(data_dir: str, batch_size: int = 16,
                      num_workers: int = 4, seed: int = 42) -> DataLoader:
    ds = EvalDataset(data_dir=data_dir, seed=seed)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        collate_fn=_collate, drop_last=False,
    )
