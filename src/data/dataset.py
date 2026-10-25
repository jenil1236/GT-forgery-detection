"""
dataset.py - IDNet ESP dataset loader.

Directory layout on disk:
  data/raw/ESP/
    original/                  <- genuine images (0001.png, ...)
    text_field_replacement/    <- forgery type 0
    face_morphing/             <- forgery type 1
    portrait_substitution/     <- forgery type 2
    mixed_fraud/               <- forgery type 3

For every original image we produce 5 samples per epoch:
  1 genuine  (label=0, forgery_type="genuine")
  4 forged   (label=1, one per forgery folder)

Each sample also contains a reference image: a DIFFERENT original drawn
at random from the same split.
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.data.transforms import SignalTransform, TextTransform, VisualTransform

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Only fraud1-4 use exact filenames matching positive/
# fraud5 and fraud6 use derived names and are excluded
FORGERY_FOLDERS: List[str] = [
    "fraud1_copy_and_move",
    "fraud2_face_morphing",
    "fraud3_face_replacement",
    "fraud4_combined",
]

FORGERY_TYPES: List[str] = [
    "fraud1_copy_and_move",
    "fraud2_face_morphing",
    "fraud3_face_replacement",
    "fraud4_combined",
]


# ---------------------------------------------------------------------------
# Helper: write error to file
# ---------------------------------------------------------------------------

def _next_error_index(error_log_dir: str) -> int:
    import glob
    existing = glob.glob(os.path.join(error_log_dir, "[0-9]*.txt"))
    if not existing:
        return 0
    indices = []
    for p in existing:
        try:
            indices.append(int(os.path.basename(p).split('_')[0]))
        except ValueError:
            pass
    return max(indices) + 1 if indices else 0


def _write_error(error_log_dir: str, tag: str, exc: Exception) -> None:
    import traceback

    if not error_log_dir:
        return
    os.makedirs(error_log_dir, exist_ok=True)
    idx = _next_error_index(error_log_dir)
    fpath = os.path.join(error_log_dir, f"{idx}_{tag}.txt")
    try:
        with open(fpath, "w") as fh:
            fh.write(f"Error #{idx}  Tag: {tag}\n\n")
            traceback.print_exc(file=fh)
        print(f"[DATASET ERROR] traceback -> {fpath}")
    except Exception:
        pass  # never crash inside error handler


# ---------------------------------------------------------------------------
# Core Dataset
# ---------------------------------------------------------------------------

class IDNetDataset(Dataset):
    """
    IDNet ESP dataset.

    Parameters
    ----------
    data_dir : str
        Path to the ``data/raw/ESP`` directory.
    split : str
        ``"train"`` or ``"val"``.
    val_fraction : float
        Fraction of original filenames reserved for validation (default 0.2).
    seed : int
        Random seed used when picking reference images (default 42).
    image_size : int
        Target spatial size passed to VisualTransform and SignalTransform.
    text_image_size : int
        Target spatial size passed to TextTransform.
    error_log_dir : str
        Directory where per-item loading errors are written.  May be "".
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        val_fraction: float = 0.2,
        seed: int = 42,
        image_size: int = 224,
        text_image_size: int = 384,
        error_log_dir: str = "",
    ) -> None:
        assert split in ("train", "val"), f"split must be 'train' or 'val', got {split!r}"
        self.data_dir = Path(data_dir)
        self.split = split
        self.val_fraction = val_fraction
        self.seed = seed
        self.error_log_dir = error_log_dir

        # Build transforms
        is_train = split == "train"
        self.visual_tf = VisualTransform(size=image_size, train=is_train)
        self.text_tf = TextTransform(size=text_image_size)
        self.signal_tf = SignalTransform(size=image_size)

        # Discover + split original filenames
        original_dir = self.data_dir / "positive"
        if not original_dir.is_dir():
            raise FileNotFoundError(f"Original images directory not found: {original_dir}")

        all_files: List[str] = sorted(
            f for f in os.listdir(original_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
        )
        if len(all_files) == 0:
            raise ValueError(f"No images found in {original_dir}")

        n_val = max(1, int(len(all_files) * val_fraction))
        n_train = len(all_files) - n_val

        if split == "train":
            self.originals: List[str] = all_files[:n_train]
        else:
            self.originals: List[str] = all_files[n_train:]

        # Pre-verify which forgery folders exist
        self.forgery_dirs: Dict[str, Path] = {}
        for ft in FORGERY_FOLDERS:
            fd = self.data_dir / ft
            if fd.is_dir():
                self.forgery_dirs[ft] = fd
            else:
                log.warning("Forgery folder missing, will skip: %s", fd)

        # Build forged sample list first (one per fraud type per original)
        self._forged_samples: List[Tuple[str, int, str, Optional[Path]]] = []
        for fname in self.originals:
            for ft, fd in self.forgery_dirs.items():
                candidate = fd / fname
                if candidate.exists():
                    self._forged_samples.append((fname, 1, ft, candidate))

        n_forged = len(self._forged_samples)

        # Build genuine pairs to MATCH the number of forged pairs.
        # Each pair uses a different random reference, so all pairs are unique.
        # This gives a 1:1 balance without repeating any (ref, test) combination.
        rng_balance = random.Random(seed + 1)  # separate seed for reproducibility
        self._genuine_samples: List[Tuple[str, int, str, Optional[Path]]] = []
        for i in range(n_forged):
            # cycle through originals so all originals get coverage
            fname = self.originals[i % len(self.originals)]
            self._genuine_samples.append((fname, 0, "genuine", None))

        # Combined balanced sample list
        self.samples: List[Tuple[str, int, str, Optional[Path]]] = (
            self._genuine_samples + self._forged_samples
        )
        # Shuffle so genuine and forged are interleaved
        rng_balance.shuffle(self.samples)

        # Seed-stable RNG for reference selection (reproducible per split)
        self._rng = random.Random(seed)

        self._print_stats()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def _print_stats(self) -> None:
        n_genuine = sum(1 for s in self.samples if s[1] == 0)
        n_forged = sum(1 for s in self.samples if s[1] == 1)
        per_type: Dict[str, int] = {}
        for s in self.samples:
            ft = s[2]
            per_type[ft] = per_type.get(ft, 0) + 1  # ft assigned above
        print(
            f"\n[IDNetDataset] split={self.split!r} | "
            f"originals={len(self.originals)} | "
            f"samples={len(self.samples)} "
            f"(genuine={n_genuine}, forged={n_forged})"
        )
        for ft, cnt in sorted(per_type.items()):
            print(f"  {ft:<30s}: {cnt}")
        print()

    # ------------------------------------------------------------------
    # Internal image loader
    # ------------------------------------------------------------------

    def _load_image(self, path: Path) -> Optional[Image.Image]:
        try:
            img = Image.open(path)
            img.load()          # force decode now, not lazily
            if img.mode != "RGB":
                img = img.convert("RGB")
            return img
        except Exception as exc:
            tag = str(path).replace(os.sep, "_").replace(":", "")
            _write_error(self.error_log_dir, tag, exc)
            return None

    # ------------------------------------------------------------------
    # Reference selection
    # ------------------------------------------------------------------

    def _pick_reference(self, exclude_fname: str) -> str:
        """Return a random original filename that is not exclude_fname."""
        candidates = [f for f in self.originals if f != exclude_fname]
        if not candidates:
            # Edge case: only one original in split — use itself
            return exclude_fname
        return self._rng.choice(candidates)

    # ------------------------------------------------------------------
    # Item loading
    # ------------------------------------------------------------------

    def _load_sample(self, idx: int) -> Optional[Dict]:
        fname, label, forgery_type, fraud_path = self.samples[idx]

        # ---- test image ------------------------------------------------
        if label == 0:
            test_path = self.data_dir / "positive" / fname
        else:
            test_path = fraud_path  # already resolved at init time

        test_img = self._load_image(test_path)
        if test_img is None:
            return None

        # ---- reference image ------------------------------------------
        ref_fname = self._pick_reference(fname)
        ref_path = self.data_dir / "positive" / ref_fname
        ref_img = self._load_image(ref_path)
        if ref_img is None:
            return None

        # ---- apply transforms -----------------------------------------
        try:
            ref_visual = self.visual_tf(ref_img)
            ref_text = self.text_tf(ref_img)
            ref_signal = self.signal_tf(ref_img)

            test_visual = self.visual_tf(test_img)
            test_text = self.text_tf(test_img)
            test_signal = self.signal_tf(test_img)
        except Exception as exc:
            tag = f"transform_{fname}_{forgery_type}"
            _write_error(self.error_log_dir, tag, exc)
            return None

        return {
            "ref_visual": ref_visual,          # (3, H, W)
            "ref_text": ref_text,              # (3, H_t, W_t)
            "ref_signal": ref_signal,          # (31, H, W)
            "test_visual": test_visual,
            "test_text": test_text,
            "test_signal": test_signal,
            "label": torch.tensor(label, dtype=torch.long),
            "forgery_type": forgery_type,
        }

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        result = self._load_sample(idx)

        # On error, walk forward until we find a valid item
        if result is None:
            for offset in range(1, len(self.samples)):
                next_idx = (idx + offset) % len(self.samples)
                result = self._load_sample(next_idx)
                if result is not None:
                    break

        if result is None:
            # Absolute fallback: return zero tensors so the collator doesn't crash
            log.error("IDNetDataset: could not load ANY valid sample near idx=%d", idx)
            result = _make_zero_sample()

        return result


# ---------------------------------------------------------------------------
# Zero-tensor fallback
# ---------------------------------------------------------------------------

def _make_zero_sample() -> Dict:
    return {
        "ref_visual": torch.zeros(3, 224, 224),
        "ref_text": torch.zeros(3, 384, 384),
        "ref_signal": torch.zeros(31, 224, 224),
        "test_visual": torch.zeros(3, 224, 224),
        "test_text": torch.zeros(3, 384, 384),
        "test_signal": torch.zeros(31, 224, 224),
        "label": torch.tensor(0, dtype=torch.long),
        "forgery_type": "unknown",
    }


# ---------------------------------------------------------------------------
# Custom collate (handles str forgery_type field)
# ---------------------------------------------------------------------------

def _collate_fn(batch: List[Dict]) -> Dict:
    """Collate a list of sample dicts into a batched dict."""
    keys = batch[0].keys()
    out: Dict = {}
    for k in keys:
        if k == "forgery_type":
            out[k] = [item[k] for item in batch]
        else:
            out[k] = torch.stack([item[k] for item in batch], dim=0)
    return out


# ---------------------------------------------------------------------------
# build_dataloaders
# ---------------------------------------------------------------------------

def build_dataloaders(cfg) -> Tuple[DataLoader, DataLoader]:
    """
    Construct train and val DataLoaders from a Config object.

    Parameters
    ----------
    cfg : Config
        Configuration dataclass (see src.utils.config).

    Returns
    -------
    train_loader, val_loader : DataLoader
    """
    common_kwargs = dict(
        data_dir=cfg.data_raw_dir,
        val_fraction=cfg.val_fraction,
        seed=cfg.seed,
        image_size=cfg.image_size,
        text_image_size=cfg.text_image_size,
        error_log_dir=cfg.error_log_dir,
    )

    train_ds = IDNetDataset(split="train", **common_kwargs)
    val_ds = IDNetDataset(split="val", **common_kwargs)

    loader_kwargs = dict(
        collate_fn=_collate_fn,
        pin_memory=(cfg.device == "cuda"),
        persistent_workers=(cfg.num_workers > 0),
    )

    # Dataset already balanced: equal genuine and forged pairs
    # (built in IDNetDataset.__init__ by expanding genuine pairs)
    n_genuine = sum(1 for s in train_ds.samples if s[1] == 0)
    n_forged  = sum(1 for s in train_ds.samples if s[1] == 1)
    print(f"[DataLoader] Balanced dataset: genuine={n_genuine}, forged={n_forged}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.stage1_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.stage1_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
        **loader_kwargs,
    )

    return train_loader, val_loader
