"""
config.py — Centralised configuration dataclass.
All hyperparameters and paths live here. Serialisable to/from JSON.
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class Config:
    # ── Paths ────────────────────────────────────────────────────────────────
    project_root: str = "."
    data_dir: str = "data"
    output_dir: str = "output"
    checkpoint_dir: str = "output/checkpoints"
    metrics_dir: str = "output/metrics"
    logs_dir: str = "output/logs"
    models_cache_dir: str = "data/models_cache"   # HF_HOME points here
    features_cache_dir: str = "data/features_cache"
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"

    # ── Dataset ──────────────────────────────────────────────────────────────
    # RVL-CDIP classes: 0=letter,1=form,2=email,3=handwritten,4=advertisement,
    #                   5=scientific_report,6=scientific_publication,7=specification,
    #                   8=file_folder,9=news_article,10=budget,11=invoice,
    #                   12=presentation,13=questionnaire,14=resume,15=memo
    train_doc_types: List[str] = field(default_factory=lambda: [
        "letter", "form", "invoice", "budget", "memo", "resume", "cord"
    ])
    zeroshot_doc_types: List[str] = field(default_factory=lambda: [
        "advertisement", "scientific_report"
    ])
    rvl_samples_per_class_train: int = 500
    rvl_samples_per_class_test: int = 300
    cord_max_samples: int = 800   # keep manageable
    image_size: int = 224          # visual + signal input
    text_image_size: int = 384     # TrOCR input
    signal_channels: int = 31      # 30 SRM + 1 FFT

    # ── Model dims ───────────────────────────────────────────────────────────
    embedding_dim: int = 128       # all encoders project to this
    judge_hidden_dim: int = 64
    expert_belief_hidden: int = 128
    n_experts: int = 3

    # ── Stage 1 (pretraining) ────────────────────────────────────────────────
    stage1_epochs: int = 30
    stage1_batch_size: int = 64
    stage1_lr: float = 1e-4
    stage1_weight_decay: float = 1e-4
    infonce_temperature: float = 0.07
    triplet_margin: float = 0.5
    contrastive_alpha: float = 0.3  # weight for alignment loss

    # ── Stage 2 (MARL/MAPPO) ────────────────────────────────────────────────
    stage2_total_epochs: int = 50   # 5 cycles × (5 judge + 5 expert) epochs
    stage2_batch_episodes: int = 32
    ppo_clip: float = 0.2
    ppo_epochs: int = 4             # inner PPO update epochs
    gae_lambda: float = 0.95
    entropy_bonus: float = 0.01
    value_loss_coef: float = 0.5
    query_cost: float = 0.1         # judge query cost c
    decorr_penalty: float = 0.05    # decorrelation penalty weight
    max_episode_steps: int = 4      # 3 expert queries + declare
    gamma: float = 0.99
    stage2_lr_actor: float = 3e-4
    stage2_lr_critic: float = 1e-3
    expert_dropout_prob: float = 0.3  # for Shapley training

    # ── Judge reward asymmetry ────────────────────────────────────────────────
    cost_false_negative: float = 2.0  # missing a forgery costs more
    cost_false_positive: float = 1.0

    # ── Evaluation ──────────────────────────────────────────────────────────
    ece_n_bins: int = 15
    eval_batch_size: int = 32

    # ── System ──────────────────────────────────────────────────────────────
    seed: int = 42
    num_workers: int = 4
    device: str = "cuda"           # overridden at runtime if CUDA unavailable
    mixed_precision: bool = True
    log_every_n_steps: int = 10
    checkpoint_every_n_epochs: int = 5

    # ── TrOCR ───────────────────────────────────────────────────────────────
    trocr_model_name: str = "microsoft/trocr-small-printed"

    # ── Run metadata ────────────────────────────────────────────────────────
    run_id: str = ""               # filled at runtime
    run_note: str = ""

    # ── Pipeline flags ──────────────────────────────────────────────────────
    skip_download: bool = False
    skip_forge: bool = False
    skip_stage1: bool = False
    skip_stage2: bool = False
    skip_eval: bool = False

    def resolve_paths(self, root: Optional[str] = None) -> "Config":
        """Make all relative paths absolute from the given root."""
        if root is None:
            root = self.project_root
        for attr in [
            "data_dir", "output_dir", "checkpoint_dir", "metrics_dir",
            "logs_dir", "models_cache_dir", "features_cache_dir",
            "raw_data_dir", "processed_data_dir"
        ]:
            val = getattr(self, attr)
            if not os.path.isabs(val):
                setattr(self, attr, os.path.join(root, val))
        return self

    def ensure_dirs(self) -> None:
        """Create all output directories if they don't exist."""
        for attr in [
            "output_dir", "checkpoint_dir", "metrics_dir", "logs_dir",
            "models_cache_dir", "features_cache_dir",
            "raw_data_dir", "processed_data_dir"
        ]:
            os.makedirs(getattr(self, attr), exist_ok=True)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def __post_init__(self):
        import torch
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
