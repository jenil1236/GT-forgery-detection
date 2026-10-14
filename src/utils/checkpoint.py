"""
checkpoint.py — Save, load, and resume training checkpoints.

Checkpoint format (dict saved as .pt):
  {
    "epoch":        int,
    "stage":        str,   e.g. "stage1_visual", "stage2"
    "model_state":  OrderedDict,
    "optimizer_state": dict | None,
    "scheduler_state": dict | None,
    "best_metric":  float,
    "metric_name":  str,
    "config":       dict,
    "run_id":       str,
  }
"""

from __future__ import annotations
import os
import glob
import re
import shutil
from typing import Any, Dict, Optional, Tuple

import torch


def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_dir: str,
    filename: str,
    is_best: bool = False,
    keep_last_n: int = 3,
) -> str:
    """
    Save a checkpoint to checkpoint_dir/filename.
    If is_best, also copy to checkpoint_dir/best_<stage>.pt.
    Prune old checkpoints keeping only the last keep_last_n per stage.
    Returns the saved path.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    torch.save(state, path)

    if is_best:
        stage = state.get("stage", "model")
        best_path = os.path.join(checkpoint_dir, f"best_{stage}.pt")
        shutil.copy2(path, best_path)

    # prune old checkpoints for this stage
    stage = state.get("stage", "model")
    _prune_checkpoints(checkpoint_dir, stage, keep_last_n)

    return path


def load_checkpoint(path: str, device: str = "cpu") -> Dict[str, Any]:
    """Load a checkpoint from path."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(path, map_location=device, weights_only=True)
    return state


def find_latest_checkpoint(checkpoint_dir: str, stage: Optional[str] = None) -> Optional[str]:
    """
    Find the latest checkpoint in checkpoint_dir, optionally filtered by stage.
    Returns None if no checkpoint found.
    Prefers best_<stage>.pt if it exists, otherwise the highest-epoch .pt file.
    """
    if not os.path.isdir(checkpoint_dir):
        return None

    # prefer best checkpoint for the stage
    if stage:
        best = os.path.join(checkpoint_dir, f"best_{stage}.pt")
        if os.path.isfile(best):
            return best

    pattern = os.path.join(checkpoint_dir, "*.pt")
    files = glob.glob(pattern)
    if not files:
        return None

    # filter by stage if provided
    if stage:
        files = [f for f in files if stage in os.path.basename(f)]
    if not files:
        return None

    # sort by modification time — newest first
    files.sort(key=lambda f: os.path.getmtime(f), reverse=True)
    return files[0]


def find_best_checkpoint(checkpoint_dir: str, stage: str) -> Optional[str]:
    """Return best_<stage>.pt if it exists."""
    path = os.path.join(checkpoint_dir, f"best_{stage}.pt")
    return path if os.path.isfile(path) else None


def checkpoint_filename(stage: str, epoch: int) -> str:
    return f"{stage}_epoch{epoch:04d}.pt"


def _prune_checkpoints(checkpoint_dir: str, stage: str, keep_n: int) -> None:
    """Delete old checkpoints for a stage, keeping the most recent keep_n."""
    pattern = os.path.join(checkpoint_dir, f"{stage}_epoch*.pt")
    files = sorted(glob.glob(pattern), key=lambda f: os.path.getmtime(f))
    for old in files[:-keep_n]:
        try:
            os.remove(old)
        except OSError:
            pass


def load_model_weights(model: torch.nn.Module, checkpoint_path: str,
                       device: str = "cpu", strict: bool = True) -> Tuple[torch.nn.Module, int]:
    """
    Load model weights from a checkpoint.
    Returns (model, epoch) tuple.
    """
    state = load_checkpoint(checkpoint_path, device=device)
    model.load_state_dict(state["model_state"], strict=strict)
    epoch = state.get("epoch", 0)
    return model, epoch
