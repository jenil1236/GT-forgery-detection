from __future__ import annotations
import argparse, json, os, sys, traceback
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src_new.models.visual_encoder    import VisualEncoder
from src_new.models.text_encoder      import TextEncoder
from src_new.models.signal_encoder    import SignalEncoder
from src_new.models.forensic_verifier import ForensicVerifier
from src_new.utils.checkpoint         import find_best_checkpoint, load_model_weights
from src_new.utils.logger             import Logger, make_run_id
from eval_new.dataset                 import build_eval_loader
from eval_new.evaluator               import run_evaluation


def parse_args():
    p = argparse.ArgumentParser(description="GT Q-Former Evaluation")
    p.add_argument("--data_dir",          required=True,
                   help="Eval dataset dir (e.g. .../SVK/svk)")
    p.add_argument("--stage1_ckpt_dir",   default="output/checkpoints",
                   help="Dir with best_stage1_*.pt (from old training)")
    p.add_argument("--stage2_ckpt_dir",   default="output_new/checkpoints",
                   help="Dir with best_qformer.pt (from new training)")
    p.add_argument("--output_dir",        default="output_new/eval")
    p.add_argument("--project_root",      default=".")
    p.add_argument("--batch_size",        type=int, default=32)
    p.add_argument("--num_workers",       type=int, default=8)
    p.add_argument("--device",            default=None)
    p.add_argument("--seed",              type=int, default=42)
    return p.parse_args()


def load_model(stage1_ckpt_dir, stage2_ckpt_dir, device, root):
    # Build encoders
    visual_enc = VisualEncoder(embed_dim=128).to(device)
    text_enc   = TextEncoder(
        embed_dim=128,
        features_cache_dir=os.path.join(root, "data", "features_cache"),
        models_cache_dir=os.path.join(root, "data", "models_cache"),
    ).to(device)
    signal_enc = SignalEncoder(embed_dim=128).to(device)

    # Load best Stage-1 encoder weights
    for name, enc in [("stage1_visual", visual_enc),
                      ("stage1_text",   text_enc),
                      ("stage1_signal", signal_enc)]:
        ckpt = find_best_checkpoint(stage1_ckpt_dir, name)
        if ckpt:
            load_model_weights(enc, ckpt, device=device)
            print(f"  Loaded {name} from {ckpt}")
        else:
            print(f"  WARNING: no checkpoint for {name}")

    # Build ForensicVerifier
    model = ForensicVerifier(
        visual_enc=visual_enc, text_enc=text_enc, signal_enc=signal_enc,
        d_model=256, G=7, n_queries=16, n_layers=2,
        n_heads=8, ffn_dim=1024, dropout=0.1, belief_hidden=128,
    ).to(device)

    # Load best Q-Former weights
    qf_ckpt = find_best_checkpoint(stage2_ckpt_dir, "qformer")
    if qf_ckpt is None:
        import glob
        files = sorted(glob.glob(os.path.join(stage2_ckpt_dir, "qformer_epoch*.pt")))
        qf_ckpt = files[-1] if files else None
    if qf_ckpt:
        try:
            import numpy as np
            torch.serialization.add_safe_globals([np.core.multiarray.scalar, np.dtype])
        except Exception: pass
        try:
            state = torch.load(qf_ckpt, map_location=device, weights_only=True)
        except Exception:
            state = torch.load(qf_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"])
        epoch  = state.get("epoch", 0)
        metric = state.get("best_metric", 0.0)
        print(f"  Loaded Q-Former from {qf_ckpt} (epoch {epoch}, val_auc={metric:.4f})")
    else:
        print("  WARNING: no Q-Former checkpoint found")

    return model


def write_report(results, path):
    lines = []
    lines.append("=" * 65)
    lines.append("GT Q-FORMER -- EVALUATION REPORT")
    lines.append("Dataset   : " + str(results.get("dataset", "unknown")))
    lines.append("Timestamp : " + str(results.get("timestamp", "")))
    lines.append("=" * 65)
    lines.append("")
    lines.append("[OVERALL METRICS]")
    ov = results.get("overall", {})
    for k in ["accuracy", "auc", "eer", "ece", "n_samples", "n_positive", "n_negative"]:
        if k in ov:
            v = ov[k]
            lines.append(f"  {k:<20s}: {v:.4f}" if isinstance(v, float) else f"  {k:<20s}: {v}")
    lines.append("")
    lines.append("[CONFUSION MATRIX]")
    cm = ov.get("confusion_matrix", {})
    tn, fp = cm.get("tn",0), cm.get("fp",0)
    fn, tp = cm.get("fn",0), cm.get("tp",0)
    lines.append(f"              Predicted")
    lines.append(f"              Genuine   Forged")
    lines.append(f"  Actual Genuine  {tn:>7}  {fp:>7}")
    lines.append(f"  Actual Forged   {fn:>7}  {tp:>7}")
    lines.append(f"  Precision : {cm.get('precision',0):.4f}")
    lines.append(f"  Recall    : {cm.get('recall',0):.4f}")
    lines.append(f"  F1        : {cm.get('f1',0):.4f}")
    lines.append("")
    lines.append("[PER FORGERY TYPE]")
    for ft, m in results.get("per_forgery_type", {}).items():
        lines.append(f"  {ft}")
        for k in ["accuracy", "auc", "eer", "n_samples"]:
            if k in m:
                v = m[k]
                lines.append(f"    {k:<20s}: {v:.4f}" if isinstance(v, float) else f"    {k:<20s}: {v}")
    lines.append("")
    lines.append("[Q-FORMER QUERY ANALYSIS]")
    qs = results.get("query_stats", {})
    if qs:
        lines.append(f"  Most active query : Q{qs.get('most_active_query',0)}")
        wts = qs.get("mean_weights", [])
        lgs = qs.get("mean_logits", [])
        for i, (w, l) in enumerate(zip(wts, lgs)):
            lines.append(f"  Q{i:02d}  weight={w:.4f}  avg_logit={l:+.4f}")
    lines.append("")
    lines.append("=" * 65)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()
    root = os.path.abspath(args.project_root)

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    os.makedirs(args.output_dir, exist_ok=True)
    run_id   = make_run_id()
    log_file = os.path.join(args.output_dir, f"{run_id}_eval.log")
    logger   = Logger(metrics_dir=args.output_dir, run_id=run_id, log_file=log_file)

    logger.info("=" * 65)
    logger.info("  GT Q-Former Forensic Verifier -- Evaluation")
    logger.info("=" * 65)
    logger.info(f"  Data dir         : {args.data_dir}")
    logger.info(f"  Stage1 ckpts     : {args.stage1_ckpt_dir}")
    logger.info(f"  Stage2 ckpts     : {args.stage2_ckpt_dir}")
    logger.info(f"  Output           : {args.output_dir}")
    logger.info(f"  Device           : {device}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        logger.info(f"  GPU              : {p.name} | {p.total_memory/1e9:.1f} GB")

    # -- Dataloader (natural distribution -- NOT balanced)
    logger.info("Building eval dataloader (natural distribution)...")
    loader = build_eval_loader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # -- Load model
    logger.info("Loading model...")
    s1_dir = args.stage1_ckpt_dir
    s2_dir = args.stage2_ckpt_dir
    if not os.path.isabs(s1_dir): s1_dir = os.path.join(root, s1_dir)
    if not os.path.isabs(s2_dir): s2_dir = os.path.join(root, s2_dir)
    model = load_model(s1_dir, s2_dir, device, root)

    # -- Run evaluation
    logger.info("Running evaluation...")
    import time
    t0      = time.time()
    results = run_evaluation(model=model, loader=loader, device=device)
    elapsed = time.time() - t0

    results["dataset"]   = str(Path(args.data_dir).name)
    results["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results["elapsed_s"] = round(elapsed, 1)

    # -- Log summary
    ov = results["overall"]
    cm = ov.get("confusion_matrix", {})
    logger.info("")
    logger.info("=" * 65)
    logger.info("RESULTS")
    logger.info(f"  Accuracy  : {ov['accuracy']:.4f}")
    logger.info(f"  AUC       : {ov['auc']:.4f}")
    logger.info(f"  EER       : {ov['eer']:.4f}")
    logger.info(f"  ECE       : {ov['ece']:.4f}")
    logger.info(f"  Precision : {cm.get('precision',0):.4f}")
    logger.info(f"  Recall    : {cm.get('recall',0):.4f}")
    logger.info(f"  F1        : {cm.get('f1',0):.4f}")
    logger.info("")
    logger.info("  Per forgery type:")
    for ft, m in results["per_forgery_type"].items():
        logger.info(f"    {ft:<35s} acc={m['accuracy']:.4f}  auc={m['auc']:.4f}")
    logger.info("=" * 65)
    logger.info(f"  Elapsed: {elapsed:.1f}s")

    # -- Save outputs
    json_path = os.path.join(args.output_dir, f"{run_id}_eval_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"  JSON   -> {json_path}")

    report_path = os.path.join(args.output_dir, f"{run_id}_eval_report.txt")
    write_report(results, report_path)
    logger.info(f"  Report -> {report_path}")

    logger.close()


if __name__ == "__main__":
    main()
