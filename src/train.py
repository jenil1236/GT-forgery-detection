"""
train.py - Main training entry point.

Pipeline:
  1. Parse args + build Config
  2. Setup logger, seed, print GPU info
  3. Build IDNetDataset dataloaders
  4. Build model system (encoders + agents + judge)
  5. Stage 1: metric learning pretraining
  6. Stage 2: MAPPO cooperative training
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import torch

# Ensure project root is on sys.path so 'src.*' imports work
# regardless of where python is invoked from
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Error logging helper
# ---------------------------------------------------------------------------

def _next_error_index(error_dir: str) -> int:
    """Auto-increment: returns 0, 1, 2 ... based on files already in error_dir."""
    import glob
    existing = glob.glob(os.path.join(error_dir, "[0-9]*.txt"))
    if not existing:
        return 0
    indices = []
    for p in existing:
        base = os.path.basename(p)  # e.g. '3_stage1.txt'
        try:
            indices.append(int(base.split('_')[0]))
        except ValueError:
            pass
    return max(indices) + 1 if indices else 0


def _log_error(error_dir: str, step: str, run_id: str, exc: Exception) -> None:
    import traceback

    os.makedirs(error_dir, exist_ok=True)
    idx = _next_error_index(error_dir)
    fpath = os.path.join(error_dir, f"{idx}_{step}.txt")
    with open(fpath, "w") as fh:
        fh.write(f"Error #{idx}\nStep: {step}\nRun ID: {run_id}\n\n")
        traceback.print_exc(file=fh)
    print(f"[ERROR] Traceback saved -> {fpath}")


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# GPU info
# ---------------------------------------------------------------------------

def _print_gpu_info(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_gb = round(props.total_memory / 1e9, 1)
        print(f"[GPU] {props.name}  |  VRAM: {vram_gb} GB  |  "
              f"CUDA {torch.version.cuda}")
    else:
        print("[WARNING] Running on CPU — training will be very slow.")


# ---------------------------------------------------------------------------
# Model system builder
# ---------------------------------------------------------------------------

def build_system(cfg, logger):
    """Instantiate all model components and return them."""
    from src.models.visual_encoder import VisualEncoder
    from src.models.text_encoder import TextEncoder
    from src.models.signal_encoder import SignalEncoder
    from src.models.agents import ExpertAgent, MultiExpertSystem
    from src.models.judge import LogOddsJudge

    logger.info("Building model system …")

    visual_enc = VisualEncoder(embed_dim=cfg.embedding_dim)
    text_enc = TextEncoder(
        embed_dim=cfg.embedding_dim,
        features_cache_dir=cfg.features_cache_dir,
        models_cache_dir=cfg.models_cache_dir,
    )
    signal_enc = SignalEncoder(
        embed_dim=cfg.embedding_dim,
        in_channels=cfg.signal_channels,
    )

    agents = {
        "visual": ExpertAgent(
            encoder=visual_enc,
            modality="visual",
            embed_dim=cfg.embedding_dim,
            belief_hidden=cfg.expert_belief_hidden,
        ),
        "text": ExpertAgent(
            encoder=text_enc,
            modality="text",
            embed_dim=cfg.embedding_dim,
            belief_hidden=cfg.expert_belief_hidden,
        ),
        "signal": ExpertAgent(
            encoder=signal_enc,
            modality="signal",
            embed_dim=cfg.embedding_dim,
            belief_hidden=cfg.expert_belief_hidden,
        ),
    }
    expert_system = MultiExpertSystem(agents)

    judge = LogOddsJudge(
        n_experts=cfg.n_experts,
        hidden_dim=cfg.judge_hidden_dim,
        query_cost=cfg.query_cost,
    )

    # Parameter count
    def _n_params(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    logger.info(
        f"  VisualEncoder   params: {_n_params(visual_enc):,}"
    )
    logger.info(
        f"  TextEncoder     params: {_n_params(text_enc):,}"
    )
    logger.info(
        f"  SignalEncoder   params: {_n_params(signal_enc):,}"
    )
    logger.info(
        f"  ExpertSystem    params: {_n_params(expert_system):,}"
    )
    logger.info(
        f"  LogOddsJudge    params: {_n_params(judge):,}"
    )

    return visual_enc, text_enc, signal_enc, expert_system, judge


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="GT Forgery Detection — Training")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the latest checkpoint.")
    parser.add_argument("--project_root", type=str, default=".",
                        help="Root directory of the project (default: current dir).")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device: 'cuda' or 'cpu'.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Global random seed (default: 42).")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Run a minimal smoke test (2 stage-1 epochs, 4 stage-2 epochs).")
    parser.add_argument("--note", type=str, default="",
                        help="Free-text note attached to this run.")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Config
    # ------------------------------------------------------------------
    from src.utils.config import Config
    from src.utils.logger import Logger, make_run_id

    cfg = Config(
        smoke_test=args.smoke_test,
        run_note=args.note,
        seed=args.seed,
    )
    cfg.resolve_paths(args.project_root)

    if args.device is not None:
        # Override after __post_init__ so the user's choice wins (then re-check CUDA)
        if args.device == "cuda" and not torch.cuda.is_available():
            print("[WARNING] --device cuda requested but CUDA unavailable; falling back to cpu.")
            cfg.device = "cpu"
        else:
            cfg.device = args.device

    cfg.run_id = make_run_id()
    cfg.ensure_dirs()

    # Persist config for reproducibility
    config_path = os.path.join(cfg.output_dir, f"config_{cfg.run_id}.json")
    cfg.save(config_path)

    # ------------------------------------------------------------------
    # 2. Logger
    # ------------------------------------------------------------------
    log_file = os.path.join(cfg.logs_dir, f"{cfg.run_id}.log")
    logger = Logger(
        metrics_dir=cfg.metrics_dir,
        run_id=cfg.run_id,
        log_file=log_file,
    )
    logger.info(f"Run ID     : {cfg.run_id}")
    logger.info(f"Note       : {cfg.run_note or '(none)'}")
    logger.info(f"Project    : {cfg.project_root}")
    logger.info(f"Device     : {cfg.device}")
    logger.info(f"Smoke test : {cfg.smoke_test}")
    logger.info(f"Config     : {config_path}")
    logger.log_config(cfg.to_dict())

    # ------------------------------------------------------------------
    # 3. Seed + GPU info
    # ------------------------------------------------------------------
    _set_seed(cfg.seed)
    _print_gpu_info(cfg.device)

    # ------------------------------------------------------------------
    # 4. Dataloaders
    # ------------------------------------------------------------------
    logger.info("Building dataloaders …")
    try:
        from src.data.dataset import build_dataloaders
        train_loader, val_loader = build_dataloaders(cfg)
        logger.info(
            f"  Train batches: {len(train_loader)}  "
            f"Val batches: {len(val_loader)}"
        )
    except Exception as exc:
        _log_error(cfg.error_log_dir, "build_dataloaders", cfg.run_id, exc)
        raise

    # ------------------------------------------------------------------
    # 5. Model system
    # ------------------------------------------------------------------
    try:
        visual_enc, text_enc, signal_enc, expert_system, judge = build_system(cfg, logger)
    except Exception as exc:
        _log_error(cfg.error_log_dir, "build_system", cfg.run_id, exc)
        raise

    # ------------------------------------------------------------------
    # 6. Stage 1: metric learning pretraining
    # ------------------------------------------------------------------
    if not cfg.skip_stage1:
        logger.info("Starting Stage 1 …")
        try:
            from src.training.pretrain import Stage1Trainer
            stage1 = Stage1Trainer(
                visual_encoder=visual_enc,
                text_encoder=text_enc,
                signal_encoder=signal_enc,
                cfg=cfg,
                logger=logger,
                device=cfg.device,
                error_log_dir=cfg.error_log_dir,
            )
            stage1.run(train_loader, val_loader, resume=args.resume)
        except Exception as exc:
            _log_error(cfg.error_log_dir, "stage1_training", cfg.run_id, exc)
            raise
    else:
        logger.info("Stage 1 skipped (--skip_stage1 or cfg.skip_stage1=True).")

    # ------------------------------------------------------------------
    # 7. Stage 2: MAPPO cooperative training
    # ------------------------------------------------------------------
    if not cfg.skip_stage2:
        # Sanitise weights before Stage 2: GradScaler can leave NaN in fp16 params
        _nan_count = 0
        for pname, param in expert_system.named_parameters():
            if torch.isnan(param.data).any() or torch.isinf(param.data).any():
                torch.nn.init.normal_(param.data, mean=0.0, std=0.01)
                _nan_count += 1
        if _nan_count:
            logger.info(f"  WARNING: reset {_nan_count} NaN/Inf param tensors before Stage 2")
        expert_system = expert_system.float()
        judge = judge.float()
        logger.info("Starting Stage 2 …")
        try:
            from src.training.marl import MAPPOTrainer
            stage2 = MAPPOTrainer(
                expert_system=expert_system,
                judge=judge,
                cfg=cfg,
                logger=logger,
                device=cfg.device,
                error_log_dir=cfg.error_log_dir,
            )
            stage2.run(train_loader, val_loader, resume=args.resume)
        except Exception as exc:
            _log_error(cfg.error_log_dir, "stage2_training", cfg.run_id, exc)
            raise
    else:
        logger.info("Stage 2 skipped (--skip_stage2 or cfg.skip_stage2=True).")

    # ------------------------------------------------------------------
    # 8. Done
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Training complete.")
    logger.info(f"  Checkpoints : {cfg.checkpoint_dir}")
    logger.info(f"  Metrics     : {cfg.metrics_dir}")
    logger.info(f"  Errors      : {cfg.error_log_dir}")
    logger.info("=" * 60)
    logger.close()


if __name__ == "__main__":
    main()
