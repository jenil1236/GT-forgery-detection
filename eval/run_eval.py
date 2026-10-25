from __future__ import annotations
import argparse, json, os, sys, traceback
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.visual_encoder import VisualEncoder
from src.models.text_encoder   import TextEncoder
from src.models.signal_encoder import SignalEncoder
from src.models.agents         import ExpertAgent, MultiExpertSystem
from src.models.judge          import LogOddsJudge
from src.utils.checkpoint      import load_checkpoint, find_best_checkpoint
from src.utils.logger          import Logger, make_run_id
from eval.dataset              import build_eval_loader
from eval.evaluator            import run_evaluation


def parse_args():
    p = argparse.ArgumentParser(description="GT Forgery Detection -- Evaluation")
    p.add_argument("--data_dir",   required=True,
                   help="Path to eval dataset dir (e.g. .../SVK/svk)")
    p.add_argument("--checkpoint_dir", default="output/checkpoints",
                   help="Dir containing best_stage2.pt")
    p.add_argument("--output_dir",  default="output/eval",
                   help="Where to save eval results")
    p.add_argument("--project_root", default=".",
                   help="Project root (same as train.py)")
    p.add_argument("--batch_size",  type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device",      default=None)
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


def build_system(device: str, embed_dim: int = 128,
                 judge_hidden: int = 64, expert_hidden: int = 128,
                 n_experts: int = 3, query_cost: float = 0.1,
                 features_cache: str = "", models_cache: str = ""):
    visual_enc = VisualEncoder(embed_dim=embed_dim)
    text_enc   = TextEncoder(embed_dim=embed_dim,
                             features_cache_dir=features_cache or None,
                             models_cache_dir=models_cache or None)
    signal_enc = SignalEncoder(embed_dim=embed_dim)
    agents = {
        "visual": ExpertAgent(visual_enc, "visual", embed_dim, expert_hidden),
        "text":   ExpertAgent(text_enc,   "text",   embed_dim, expert_hidden),
        "signal": ExpertAgent(signal_enc, "signal", embed_dim, expert_hidden),
    }
    expert_system = MultiExpertSystem(agents)
    judge = LogOddsJudge(n_experts=n_experts, hidden_dim=judge_hidden,
                         query_cost=query_cost)
    return expert_system, judge


def load_stage2(expert_system, judge, checkpoint_dir: str, device: str):
    ckpt = find_best_checkpoint(checkpoint_dir, "stage2")
    if ckpt is None:
        ckpt_files = [f for f in os.listdir(checkpoint_dir)
                      if "stage2" in f and f.endswith(".pt")]
        if not ckpt_files:
            raise FileNotFoundError(
                f"No stage2 checkpoint found in {checkpoint_dir}")
        ckpt = os.path.join(checkpoint_dir,
                            sorted(ckpt_files)[-1])
    print(f"Loading checkpoint: {ckpt}")
    state = load_checkpoint(ckpt, device=device)
    ms = state.get("model_state", {})
    if "experts" in ms:
        expert_system.load_state_dict(ms["experts"])
    if "judge" in ms:
        judge.load_state_dict(ms["judge"])
    epoch = state.get("epoch", -1)
    metric = state.get("best_metric", 0.0)
    print(f"  Loaded from epoch {epoch}, train best_val_auc={metric:.4f}")


def write_report(results: dict, path: str) -> None:
    lines = []
    lines.append("=" * 62)
    lines.append("GT FORGERY DETECTION -- EVALUATION REPORT")
    lines.append("Dataset   : " + str(results.get("dataset", "unknown")))
    lines.append("Timestamp : " + str(results.get("timestamp", "")))
    lines.append("=" * 62)
    lines.append("")
    lines.append("[OVERALL]")
    ov = results.get("overall", {})
    for k in ["accuracy", "auc", "eer", "ece", "mean_n_queries",
              "n_samples", "n_positive", "n_negative"]:
        if k in ov:
            v = ov[k]
            lines.append(f"  {k:<25s}: {v:.4f}" if isinstance(v, float)
                         else f"  {k:<25s}: {v}")
    lines.append("")
    lines.append("[PER FORGERY TYPE]")
    for ft, m in results.get("per_forgery_type", {}).items():
        lines.append(f"  {ft}")
        for k in ["accuracy", "auc", "eer", "n_samples"]:
            if k in m:
                v = m[k]
                lines.append(f"    {k:<20s}: {v:.4f}" if isinstance(v, float)
                             else f"    {k:<20s}: {v}")
    lines.append("")
    lines.append("[SHAPLEY VALUES -- ALL SAMPLES]")
    for exp, stats in results.get("shapley_all", {}).items():
        mv, sv, nv, xv = stats["mean"], stats["std"], stats["min"], stats["max"]
        lines.append(f"  {exp:<10s}  mean={mv:.4f}  std={sv:.4f}  "
                     f"min={nv:.4f}  max={xv:.4f}")
    lines.append("")
    lines.append("[SHAPLEY VALUES -- FORGED ONLY]")
    for exp, stats in results.get("shapley_forged", {}).items():
        mv, sv = stats["mean"], stats["std"]
        lines.append(f"  {exp:<10s}  mean={mv:.4f}  std={sv:.4f}")
    lines.append("")
    lines.append("=" * 62)
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
    logger   = Logger(metrics_dir=args.output_dir, run_id=run_id,
                      log_file=log_file)

    logger.info("=" * 62)
    logger.info("  GT Forgery Detection -- Evaluation")
    logger.info("=" * 62)
    logger.info(f"  Data dir    : {args.data_dir}")
    logger.info(f"  Checkpoints : {args.checkpoint_dir}")
    logger.info(f"  Output      : {args.output_dir}")
    logger.info(f"  Device      : {device}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        logger.info(f"  GPU         : {p.name} | {p.total_memory/1e9:.1f} GB")

    # ── build dataloader ──────────────────────────────────────────────────────
    logger.info("Building eval dataloader...")
    loader = build_eval_loader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # ── build + load model ────────────────────────────────────────────────────
    logger.info("Building model system...")
    ckpt_dir = args.checkpoint_dir
    if not os.path.isabs(ckpt_dir):
        ckpt_dir = os.path.join(root, ckpt_dir)
    features_cache = os.path.join(root, "data", "features_cache")
    models_cache   = os.path.join(root, "data", "models_cache")

    expert_system, judge = build_system(
        device=device,
        features_cache=features_cache,
        models_cache=models_cache,
    )
    load_stage2(expert_system, judge, ckpt_dir, device)
    expert_system = expert_system.to(device)
    judge         = judge.to(device)

    # ── run evaluation ────────────────────────────────────────────────────────
    logger.info("Running evaluation...")
    t0 = __import__("time").time()
    results = run_evaluation(
        expert_system=expert_system,
        judge=judge,
        loader=loader,
        device=device,
    )
    elapsed = __import__("time").time() - t0

    # attach metadata
    results["dataset"]   = str(Path(args.data_dir).name)
    results["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results["elapsed_s"] = round(elapsed, 1)

    # ── log summary ───────────────────────────────────────────────────────────
    ov = results["overall"]
    logger.info("")
    logger.info("=" * 62)
    logger.info("RESULTS")
    logger.info(f"  Accuracy     : {ov['accuracy']:.4f}")
    logger.info(f"  AUC          : {ov['auc']:.4f}")
    logger.info(f"  EER          : {ov['eer']:.4f}")
    logger.info(f"  ECE          : {ov['ece']:.4f}")
    logger.info(f"  Mean queries : {ov['mean_n_queries']:.2f}")
    logger.info("")
    logger.info("  Per forgery type:")
    for ft, m in results["per_forgery_type"].items():
        logger.info(f"    {ft:<35s} acc={m['accuracy']:.4f}  auc={m['auc']:.4f}")
    logger.info("")
    logger.info("  Shapley values (mean contribution per expert):")
    for exp, s in results["shapley_all"].items():
        logger.info(f"    {exp:<10s}  {s['mean']:+.4f}  (std={s['std']:.4f})")
    logger.info("=" * 62)
    logger.info(f"  Elapsed: {elapsed:.1f}s")

    # ── save outputs ──────────────────────────────────────────────────────────
    # full JSON (includes per-sample shapley)
    json_path = os.path.join(args.output_dir, f"{run_id}_eval_metrics.json")
    # per_sample_shapley can be large -- save separately
    per_sample = results.pop("per_sample_shapley", [])
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"  Metrics JSON  -> {json_path}")

    ps_path = os.path.join(args.output_dir, f"{run_id}_per_sample_shapley.json")
    with open(ps_path, "w", encoding="utf-8") as f:
        json.dump(per_sample, f)
    logger.info(f"  Per-sample    -> {ps_path}")

    report_path = os.path.join(args.output_dir, f"{run_id}_eval_report.txt")
    results["per_sample_shapley"] = per_sample
    write_report(results, report_path)
    logger.info(f"  Report        -> {report_path}")

    logger.close()


if __name__ == "__main__":
    main()
