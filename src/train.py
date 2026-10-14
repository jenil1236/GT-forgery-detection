"""
train.py - Main training entry point.

Usage:
    python src/train.py                  # fresh run
    python src/train.py --resume         # auto-resume from latest checkpoint
    python src/train.py --skip_stage1    # skip encoder pretraining
    python src/train.py --eval_only      # evaluation only (requires stage2 checkpoint)

Pipeline:
    1. Download datasets (CORD-v2, RVL-CDIP subset)
    2. Generate synthetic forgeries
    3. Stage 1: metric learning for visual, text, signal encoders
    4. Stage 2: MAPPO cooperative multi-agent training
    5. Evaluation: seen types, zero-shot, Shapley values, robustness
"""

from __future__ import annotations
import argparse
import os
import random
import sys

import numpy as np
import torch


def _set_hf_home(project_root: str) -> None:
    cache = os.path.join(project_root, "data", "models_cache")
    os.makedirs(cache, exist_ok=True)
    os.environ["HF_HOME"] = cache
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache, "transformers")
    os.environ["HF_DATASETS_CACHE"] = os.path.join(cache, "datasets")
    print(f"HF_HOME set to: {cache}")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GT Forgery Detection Training")
    p.add_argument("--resume", action="store_true",
                   help="Auto-resume from latest checkpoint")
    p.add_argument("--project_root", default=".",
                   help="Project root directory")
    p.add_argument("--skip_download", action="store_true")
    p.add_argument("--skip_forge", action="store_true")
    p.add_argument("--skip_stage1", action="store_true")
    p.add_argument("--skip_stage2", action="store_true")
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--device", default=None,
                   help="Override device (cuda/cpu)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--note", default="", help="Run note for logging")
    return p.parse_args()


def build_system(cfg, device: str):
    """Instantiate all model components from config."""
    from src.models.visual_encoder import VisualEncoder
    from src.models.text_encoder import TextEncoder
    from src.models.signal_encoder import SignalEncoder
    from src.models.agents import ExpertAgent, MultiExpertSystem
    from src.models.judge import LogOddsJudge

    visual_enc = VisualEncoder(embed_dim=cfg.embedding_dim)
    text_enc   = TextEncoder(
        embed_dim=cfg.embedding_dim,
        model_name=cfg.trocr_model_name,
        features_cache_dir=cfg.features_cache_dir,
        models_cache_dir=cfg.models_cache_dir,
    )
    signal_enc = SignalEncoder(embed_dim=cfg.embedding_dim)

    agents = {
        "visual": ExpertAgent(visual_enc, "visual", cfg.embedding_dim, cfg.expert_belief_hidden),
        "text":   ExpertAgent(text_enc,   "text",   cfg.embedding_dim, cfg.expert_belief_hidden),
        "signal": ExpertAgent(signal_enc, "signal", cfg.embedding_dim, cfg.expert_belief_hidden),
    }
    expert_system = MultiExpertSystem(agents)
    judge = LogOddsJudge(
        n_experts=cfg.n_experts,
        hidden_dim=cfg.judge_hidden_dim,
        query_cost=cfg.query_cost,
    )

    return visual_enc, text_enc, signal_enc, expert_system, judge


def main():
    args = parse_args()
    project_root = os.path.abspath(args.project_root)

    # Must set HF_HOME before any HF imports
    _set_hf_home(project_root)

    # Now safe to import HF-dependent modules
    from src.utils.config import Config
    from src.utils.logger import Logger, make_run_id
    from src.utils.checkpoint import find_latest_checkpoint

    # ── Config ───────────────────────────────────────────────────────────────
    cfg = Config(project_root=project_root)
    cfg.resolve_paths(project_root)
    cfg.run_id = make_run_id()
    cfg.run_note = args.note
    if args.device:
        cfg.device = args.device
    if args.skip_download:  cfg.skip_download = True
    if args.skip_forge:     cfg.skip_forge = True
    if args.skip_stage1:    cfg.skip_stage1 = True
    if args.skip_stage2:    cfg.skip_stage2 = True
    cfg.ensure_dirs()
    cfg.save(os.path.join(cfg.output_dir, "config.json"))

    # ── Logger ───────────────────────────────────────────────────────────────
    log_file = os.path.join(cfg.logs_dir, f"{cfg.run_id}.log")
    logger = Logger(
        metrics_dir=cfg.metrics_dir,
        run_id=cfg.run_id,
        log_file=log_file,
    )

    _set_seed(args.seed)

    # ── Print header ─────────────────────────────────────────────────────────
    logger.info("="*60)
    logger.info("  Game-Theoretic Reference-Conditioned Forgery Detection")
    logger.info("="*60)
    logger.info(f"  Run ID:      {cfg.run_id}")
    logger.info(f"  Project root:{project_root}")
    logger.info(f"  Device:      {cfg.device}")
    logger.info(f"  Resume:      {args.resume}")

    # GPU info
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        logger.info(f"  GPU:         {gpu.name}")
        logger.info(f"  VRAM:        {gpu.total_memory / 1e9:.1f} GB")
    else:
        logger.info("  GPU:         Not available — running on CPU")

    logger.log_config(cfg.to_dict())

    # ── Step 1: Download datasets ─────────────────────────────────────────────
    if not cfg.skip_download:
        logger.info("\n[STEP 1/5] Downloading datasets ...")
        from src.data.download import download_cord, download_rvl_cdip
        raw_dir = cfg.raw_data_dir
        download_cord(raw_dir, max_samples=cfg.cord_max_samples, seed=args.seed)
        download_rvl_cdip(
            raw_dir,
            train_classes=["letter", "form", "invoice", "budget", "memo", "resume"],
            zeroshot_classes=["advertisement", "scientific_report"],
            samples_per_class_train=cfg.rvl_samples_per_class_train,
            samples_per_class_zeroshot=cfg.rvl_samples_per_class_test,
            seed=args.seed,
        )
    else:
        logger.info("[STEP 1/5] Download skipped.")

    # ── Step 2: Generate forgeries ───────────────────────────────────────────
    if not cfg.skip_forge:
        logger.info("\n[STEP 2/5] Generating synthetic forgeries ...")
        from src.data.forge import run_forge_pipeline
        all_types = cfg.train_doc_types + cfg.zeroshot_doc_types
        stats = run_forge_pipeline(
            cfg.raw_data_dir, cfg.processed_data_dir,
            doc_types=all_types, seed=args.seed,
        )
        logger.info(f"  Forgery stats: {stats}")
    else:
        logger.info("[STEP 2/5] Forgery generation skipped.")

    # ── Build datasets ────────────────────────────────────────────────────────
    logger.info("\nBuilding DataLoaders ...")
    from src.data.dataset import build_dataloaders
    train_loader, val_loader, zs_loader = build_dataloaders(cfg, cfg.processed_data_dir)
    logger.info(f"  Train samples:     {len(train_loader.dataset)}")
    logger.info(f"  Val samples:       {len(val_loader.dataset)}")
    logger.info(f"  Zero-shot samples: {len(zs_loader.dataset)}")

    # ── Build system ──────────────────────────────────────────────────────────
    logger.info("\nBuilding model system ...")
    visual_enc, text_enc, signal_enc, expert_system, judge = build_system(cfg, cfg.device)

    n_params = sum(p.numel() for p in expert_system.parameters() if p.requires_grad)
    n_params += sum(p.numel() for p in judge.parameters() if p.requires_grad)
    logger.info(f"  Trainable parameters: {n_params:,}")

    # ── Step 3: Stage 1 pretraining ───────────────────────────────────────────
    if not cfg.skip_stage1 and not args.eval_only:
        logger.info("\n[STEP 3/5] Stage 1: Encoder pretraining ...")
        from src.training.pretrain import Stage1Trainer
        trainer1 = Stage1Trainer(
            visual_enc, text_enc, signal_enc,
            cfg, logger, device=cfg.device,
        )
        trainer1.run(train_loader, val_loader, resume=args.resume)
    else:
        logger.info("[STEP 3/5] Stage 1 skipped — loading best checkpoints ...")
        from src.utils.checkpoint import find_best_checkpoint, load_model_weights
        for name, model in [("stage1_visual", visual_enc), ("stage1_text", text_enc), ("stage1_signal", signal_enc)]:
            best = find_best_checkpoint(cfg.checkpoint_dir, name)
            if best:
                load_model_weights(model, best, device=cfg.device)
                logger.info(f"  Loaded {name} from {best}")

    # ── Step 4: Stage 2 MARL ─────────────────────────────────────────────────
    if not cfg.skip_stage2 and not args.eval_only:
        logger.info("\n[STEP 4/5] Stage 2: MAPPO Multi-Agent Training ...")
        from src.training.marl import MAPPOTrainer
        trainer2 = MAPPOTrainer(
            expert_system, judge,
            cfg, logger, device=cfg.device,
        )
        trainer2.run(train_loader, val_loader, resume=args.resume)
    else:
        logger.info("[STEP 4/5] Stage 2 skipped — loading best checkpoint ...")
        from src.utils.checkpoint import find_best_checkpoint, load_checkpoint
        best = find_best_checkpoint(cfg.checkpoint_dir, "stage2")
        if best:
            state = load_checkpoint(best, device=cfg.device)
            ms = state.get("model_state", {})
            if "experts" in ms:
                expert_system.load_state_dict(ms["experts"])
            if "judge" in ms:
                judge.load_state_dict(ms["judge"])
            logger.info(f"  Loaded stage2 from {best}")

    # ── Step 5: Evaluation ────────────────────────────────────────────────────
    logger.info("\n[STEP 5/5] Final Evaluation ...")
    from src.evaluation.evaluate import Evaluator
    evaluator = Evaluator(expert_system, judge, cfg, logger, device=cfg.device)
    results = evaluator.run_full_evaluation(val_loader, zs_loader)

    # Final summary
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"  Seen AUC:       {results['seen'].get('auc', 0):.4f}")
    logger.info(f"  Zero-shot AUC:  {results['zero_shot'].get('auc', 0):.4f}")
    logger.info(f"  Seen EER:       {results['seen'].get('eer', 0):.4f}")
    logger.info(f"  Private code:   {results['robustness'].get('private_code_detected', False)}")
    logger.info("="*60)

    logger.close()


if __name__ == "__main__":
    main()