from __future__ import annotations
import argparse, os, random, sys, traceback
from pathlib import Path

import numpy as np
import torch

# Ensure project root on path
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _log_error(error_dir, step, run_id, exc):
    os.makedirs(error_dir, exist_ok=True)
    import glob
    existing = glob.glob(os.path.join(error_dir, "[0-9]*.txt"))
    idxs = []
    for p in existing:
        try: idxs.append(int(os.path.basename(p).split("_")[0]))
        except: pass
    idx = max(idxs) + 1 if idxs else 0
    fpath = os.path.join(error_dir, f"{idx}_{step}.txt")
    with open(fpath, "w") as f:
        f.write(f"Step: {step}\nRun ID: {run_id}\n\n")
        traceback.print_exc(file=f)
    print(f"[ERROR] Traceback -> {fpath}")


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def _gpu_info():
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"[GPU] {p.name}  |  VRAM: {p.total_memory/1e9:.1f} GB  |  CUDA {torch.version.cuda}")
    else:
        print("[WARNING] No GPU -- training will be slow")


def parse_args():
    p = argparse.ArgumentParser(description="GT Forgery -- Q-Former Training")
    p.add_argument("--resume",       action="store_true")
    p.add_argument("--project_root", default=".", type=str)
    p.add_argument("--device",       default=None, type=str)
    p.add_argument("--seed",         default=42, type=int)
    p.add_argument("--smoke_test",   action="store_true")
    p.add_argument("--skip_stage1",  action="store_true")
    p.add_argument("--note",         default="", type=str)
    return p.parse_args()


def build_encoders(cfg, device):
    from src_new.models.visual_encoder import VisualEncoder
    from src_new.models.text_encoder   import TextEncoder
    from src_new.models.signal_encoder import SignalEncoder
    visual_enc = VisualEncoder(embed_dim=cfg.embedding_dim)
    text_enc   = TextEncoder(
        embed_dim=cfg.embedding_dim,
        features_cache_dir=cfg.features_cache_dir,
        models_cache_dir=cfg.models_cache_dir,
    )
    signal_enc = SignalEncoder(embed_dim=cfg.embedding_dim)
    return visual_enc.to(device), text_enc.to(device), signal_enc.to(device)


def load_stage1_best(visual_enc, text_enc, signal_enc, cfg, logger):
    from src_new.utils.checkpoint import find_best_checkpoint, load_model_weights
    for name, enc in [("stage1_visual", visual_enc),
                      ("stage1_text",   text_enc),
                      ("stage1_signal", signal_enc)]:
        ckpt = find_best_checkpoint(cfg.checkpoint_dir, name)
        if ckpt:
            load_model_weights(enc, ckpt, device=cfg.device)
            logger.info(f"  Loaded best {name} from {ckpt}")
        else:
            logger.info(f"  No Stage-1 checkpoint for {name} -- using random init")


def main():
    args = parse_args()
    root = os.path.abspath(args.project_root)

    from src_new.utils.config import Config
    from src_new.utils.logger import Logger, make_run_id

    cfg = Config(smoke_test=args.smoke_test, run_note=args.note, seed=args.seed)
    cfg.resolve_paths(root)
    if args.device:
        if args.device == "cuda" and not torch.cuda.is_available():
            cfg.device = "cpu"
        else:
            cfg.device = args.device
    cfg.run_id = make_run_id()
    cfg.ensure_dirs()
    cfg.save(os.path.join(cfg.output_dir_new, f"config_{cfg.run_id}.json"))

    log_file = os.path.join(cfg.logs_dir_new, f"{cfg.run_id}.log")
    logger   = Logger(metrics_dir=cfg.metrics_dir_new, run_id=cfg.run_id,
                      log_file=log_file)

    logger.info("=" * 62)
    logger.info("  GT Forgery -- Q-Former Forensic Verifier")
    logger.info("=" * 62)
    logger.info(f"  Run ID     : {cfg.run_id}")
    logger.info(f"  Project    : {root}")
    logger.info(f"  Device     : {cfg.device}")
    logger.info(f"  Smoke test : {cfg.smoke_test}")
    logger.log_config(cfg.to_dict())

    _set_seed(cfg.seed)
    _gpu_info()

    # -- DataLoaders
    logger.info("Building dataloaders...")
    try:
        from src_new.data.dataset import build_dataloaders
        train_loader, val_loader = build_dataloaders(cfg)
        logger.info(f"  Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")
    except Exception as exc:
        _log_error(cfg.error_log_dir_new, "build_dataloaders", cfg.run_id, exc)
        raise

    # -- Build encoders
    logger.info("Building encoders...")
    try:
        visual_enc, text_enc, signal_enc = build_encoders(cfg, cfg.device)
    except Exception as exc:
        _log_error(cfg.error_log_dir_new, "build_encoders", cfg.run_id, exc)
        raise

    # -- Stage 1: metric learning (reused from src/)
    if not args.skip_stage1 and not cfg.skip_stage1:
        logger.info("Starting Stage 1 (metric learning)...")
        try:
            from src.training.pretrain import Stage1Trainer
            stage1 = Stage1Trainer(
                visual_encoder=visual_enc, text_encoder=text_enc,
                signal_encoder=signal_enc, cfg=cfg, logger=logger,
                device=cfg.device, error_log_dir=cfg.error_log_dir_new,
            )
            stage1.run(train_loader, val_loader, resume=args.resume)
        except Exception as exc:
            _log_error(cfg.error_log_dir_new, "stage1", cfg.run_id, exc)
            raise
    else:
        logger.info("Stage 1 skipped -- loading best checkpoints")

    # -- Load best Stage-1 weights before Q-Former training
    try:
        from src_new.utils.checkpoint import find_latest_checkpoint
        _s2_exists = find_latest_checkpoint(cfg.checkpoint_dir_new, "qformer") is not None
        if not _s2_exists:
            load_stage1_best(visual_enc, text_enc, signal_enc, cfg, logger)
        else:
            logger.info("  Resuming Q-Former -- encoder weights from Q-Former checkpoint")
    except Exception as exc:
        _log_error(cfg.error_log_dir_new, "load_stage1_best", cfg.run_id, exc)
        raise

    # -- Build ForensicVerifier
    logger.info("Building ForensicVerifier (Q-Former)...")
    try:
        from src_new.models.forensic_verifier import ForensicVerifier
        model = ForensicVerifier(
            visual_enc=visual_enc, text_enc=text_enc, signal_enc=signal_enc,
            d_model=cfg.qf_d_model, G=cfg.qf_G,
            n_queries=cfg.qf_n_queries, n_layers=cfg.qf_n_layers,
            n_heads=cfg.qf_n_heads, ffn_dim=cfg.qf_ffn_dim,
            dropout=cfg.qf_dropout, belief_hidden=cfg.qf_belief_hidden,
        ).to(cfg.device)
        pc = model.param_counts()
        logger.info(f"  Trainable: {pc['trainable']:,}  Frozen: {pc['frozen']:,}")
    except Exception as exc:
        _log_error(cfg.error_log_dir_new, "build_model", cfg.run_id, exc)
        raise

    # -- Q-Former training
    logger.info("Starting Q-Former training...")
    try:
        from src_new.training.train_qformer import QFormerTrainer
        trainer = QFormerTrainer(
            model=model, cfg=cfg, logger=logger,
            device=cfg.device, error_log_dir=cfg.error_log_dir_new,
        )
        # Override checkpoint dir to use output_new
        cfg.checkpoint_dir = cfg.checkpoint_dir_new
        trainer.run(train_loader, val_loader, resume=args.resume)
    except Exception as exc:
        _log_error(cfg.error_log_dir_new, "qformer_training", cfg.run_id, exc)
        raise

    logger.info("=" * 62)
    logger.info("Training complete.")
    logger.info(f"  Checkpoints : {cfg.checkpoint_dir_new}")
    logger.info(f"  Metrics     : {cfg.metrics_dir_new}")
    logger.info(f"  Errors      : {cfg.error_log_dir_new}")
    logger.info("=" * 62)
    logger.close()


if __name__ == "__main__":
    main()
