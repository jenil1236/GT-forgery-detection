"""
config.py - Training configuration dataclass.

All paths are relative to project_root by default.
Call cfg.resolve_paths(root) to make them absolute, then
cfg.ensure_dirs() to create them on disk.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict


@dataclass
class Config:
    # ── Directory structure ────────────────────────────────────────────────
    project_root: str = "."
    data_dir: str = "data"
    output_dir: str = "output"
    checkpoint_dir: str = "output/checkpoints"
    metrics_dir: str = "output/metrics"
    logs_dir: str = "output/logs"
    error_log_dir: str = "output/errors"
    models_cache_dir: str = "data/models_cache"
    features_cache_dir: str = "data/features_cache"
    data_raw_dir: str = "dataset/IDNet/archives/EST/EST"
    doc_type: str = "ESP"

    # ── Input / transforms ─────────────────────────────────────────────────
    image_size: int = 224
    text_image_size: int = 384
    signal_channels: int = 31

    # ── Data split ─────────────────────────────────────────────────────────
    val_fraction: float = 0.2

    # ── Model ──────────────────────────────────────────────────────────────
    embedding_dim: int = 128
    judge_hidden_dim: int = 64
    expert_belief_hidden: int = 128
    n_experts: int = 3

    # ── Stage 1 (metric learning pretraining) ─────────────────────────────
    stage1_epochs: int = 30
    stage1_batch_size: int = 32
    stage1_lr: float = 1e-4
    stage1_weight_decay: float = 1e-4
    infonce_temperature: float = 0.07
    triplet_margin: float = 0.5
    contrastive_alpha: float = 0.3
    mixed_precision: bool = False  # disabled: NaN overflow with L2-norm + LayerNorm projectors

    # ── Stage 2 (MAPPO) ────────────────────────────────────────────────────
    stage2_total_epochs: int = 50
    ppo_clip: float = 0.2
    ppo_epochs: int = 4
    gae_lambda: float = 0.95
    gamma: float = 0.99
    entropy_bonus: float = 0.01
    value_loss_coef: float = 0.5
    query_cost: float = 0.1
    decorr_penalty: float = 0.05
    max_episode_steps: int = 4
    stage2_lr_actor: float = 3e-4
    stage2_lr_critic: float = 1e-3
    cost_false_negative: float = 2.0
    cost_false_positive: float = 1.0

    # ── Runtime ────────────────────────────────────────────────────────────
    seed: int = 42
    num_workers: int = 4
    device: str = "cuda"
    log_every_n_steps: int = 10
    checkpoint_every_n_epochs: int = 5

    # -- Q-Former (new Stage 2)
    qf_epochs: int          = 30
    qf_lr: float            = 3e-4
    qf_weight_decay: float  = 1e-4
    qf_d_model: int         = 256
    qf_n_queries: int       = 16
    qf_n_layers: int        = 2
    qf_n_heads: int         = 8
    qf_ffn_dim: int         = 1024
    qf_dropout: float       = 0.1
    qf_belief_hidden: int   = 128
    qf_G: int               = 7
    output_dir_new: str     = "output_new"
    checkpoint_dir_new: str = "output_new/checkpoints"
    metrics_dir_new: str    = "output_new/metrics"
    logs_dir_new: str       = "output_new/logs"
    error_log_dir_new: str  = "output_new/errors"

    # -- Execution flags
    smoke_test: bool = False
    skip_stage1: bool = False
    skip_stage2: bool = False

    # ── Run metadata ───────────────────────────────────────────────────────
    run_id: str = ""
    run_note: str = ""

    # ------------------------------------------------------------------
    # Post-init adjustments
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        # Smoke-test overrides: tiny run for pipeline validation
        if self.smoke_test:
            self.stage1_epochs = 2
            self.stage2_total_epochs = 4
            self.qf_epochs = 2
            self.stage1_batch_size = 2
            self.num_workers = 0
            self.mixed_precision = False  # avoid NaN with tiny batches

        # Auto-fall-back to CPU when CUDA is unavailable
        if self.device == "cuda":
            import torch
            if not torch.cuda.is_available():
                self.device = "cpu"

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------

    def resolve_paths(self, root: str) -> "Config":
        """
        Make all relative directory paths absolute by prepending *root*.
        Absolute paths are left unchanged.
        Called once at startup after setting project_root.
        """
        self.project_root = os.path.abspath(root)

        path_fields = [
            "data_dir",
            "output_dir",
            "checkpoint_dir",
            "metrics_dir",
            "logs_dir",
            "error_log_dir",
            "models_cache_dir",
            "features_cache_dir",
            "data_raw_dir",
            "output_dir_new",
            "checkpoint_dir_new",
            "metrics_dir_new",
            "logs_dir_new",
            "error_log_dir_new",
        ]
        for attr in path_fields:
            val = getattr(self, attr)
            if not os.path.isabs(val):
                setattr(self, attr, os.path.join(self.project_root, val))

        return self

    def ensure_dirs(self) -> None:
        """Create all output / cache directories on disk."""
        dirs = [
            self.output_dir,
            self.checkpoint_dir,
            self.metrics_dir,
            self.logs_dir,
            self.error_log_dir,
            self.models_cache_dir,
            self.features_cache_dir,
            self.output_dir_new,
            self.checkpoint_dir_new,
            self.metrics_dir_new,
            self.logs_dir_new,
            self.error_log_dir_new,
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        # Only forward keys that are recognised dataclass fields
        import dataclasses
        known = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)

    def save(self, path: str) -> None:
        """Serialise config to a JSON file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "Config":
        """Deserialise config from a JSON file."""
        with open(path) as fh:
            d = json.load(fh)
        return cls.from_dict(d)
