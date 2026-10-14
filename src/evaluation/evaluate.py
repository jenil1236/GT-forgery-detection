"""
evaluate.py - Full evaluation pipeline.

Runs on:
  (a) Seen document types (in-distribution)
  (b) Zero-shot held-out document types (out-of-distribution)

Also runs the private-protocol robustness test:
  Reinitialise the judge weights and measure accuracy retained.
  If accuracy collapses -> experts learned a private code.
"""

from __future__ import annotations
import json
import os
import copy
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from src.models.agents import MultiExpertSystem
from src.models.judge import LogOddsJudge, DECLARE_ACTION
from src.training.marl import EpisodeBuffer
from src.evaluation.metrics import compute_all_metrics, speed_accuracy_curve
from src.training.shapley import ExactShapleyComputer
from src.utils.logger import Logger


class Evaluator:
    """Full evaluation for the GT Forgery Detection system."""

    def __init__(
        self,
        expert_system: MultiExpertSystem,
        judge: LogOddsJudge,
        cfg,
        logger: Logger,
        device: str = "cuda",
    ):
        self.expert_system = expert_system.to(device)
        self.judge = judge.to(device)
        self.cfg = cfg
        self.logger = logger
        self.device = device
        self.modalities = ["visual", "text", "signal"]

    @torch.no_grad()
    def _run_episodes(self, loader: DataLoader) -> Dict:
        self.expert_system.eval()
        self.judge.eval()

        all_labels, all_scores, all_preds = [], [], []
        n_queries_list = []
        preds_by_n_experts = {1: [], 2: [], 3: []}  # stores (label, pred) pairs

        for batch in self.logger.progress(loader, desc="  Evaluating", leave=False):
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            B = batch["label"].size(0)
            labels = batch["label"].to(self.device)
            lambda_t = torch.zeros(B, device=self.device)
            eb_slot = torch.full((B, self.cfg.n_experts), 0.5, device=self.device)
            avail = torch.ones(B, self.cfg.n_experts + 1, dtype=torch.bool, device=self.device)
            done = torch.zeros(B, dtype=torch.bool, device=self.device)
            n_q = torch.zeros(B, device=self.device)

            eo = self.expert_system(batch, deterministic=True)
            jout = None

            for step in range(self.cfg.max_episode_steps):
                jout = self.judge(
                    lambda_t=lambda_t, step=step,
                    max_steps=self.cfg.max_episode_steps,
                    expert_beliefs=eb_slot, available_mask=avail,
                )
                action = jout["action"]
                nd = done.clone()
                for b in range(B):
                    if done[b]: continue
                    a = action[b].item()
                    is_last = (step == self.cfg.max_episode_steps - 1)
                    if a == DECLARE_ACTION or is_last:
                        nd[b] = True
                    elif a < self.cfg.n_experts:
                        nm = self.modalities[a]
                        q = eo[nm]["belief"][b].squeeze().clamp(1e-7, 1-1e-7) if nm in eo else torch.tensor(0.5, device=self.device)
                        lambda_t[b] = LogOddsJudge.update_lambda(lambda_t[b:b+1], q.unsqueeze(0)).squeeze()
                        eb_slot[b, a] = q
                        avail[b, a] = False
                        n_q[b] += 1
                        # Record per-n_experts predictions
                        nq_int = int(n_q[b].item())
                        if nq_int in preds_by_n_experts:
                            bel_now = torch.sigmoid(lambda_t[b]).item()
                            # FIX: store (label, belief) pair so label/pred counts match
                            preds_by_n_experts[nq_int].append(
                                (labels[b].item(), bel_now)
                            )
                done = nd
                if done.all(): break

            if jout is not None:
                beliefs = jout["belief"].cpu()
                preds = (beliefs >= 0.5).long()
                all_scores.extend(beliefs.tolist())
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.cpu().tolist())
                n_queries_list.extend(n_q.cpu().tolist())

        mean_n_queries = sum(n_queries_list) / max(len(n_queries_list), 1)
        metrics = compute_all_metrics(all_labels, all_scores, n_bins=self.cfg.ece_n_bins)
        metrics["mean_n_queries"] = mean_n_queries

        # Speed-accuracy curve
        # FIX: build matched (labels, preds) per bucket ? do NOT use all_labels
        curve_labels_per_n = {}
        curve_preds = {}
        curve_mean_q = {}
        for n_exp, items in preds_by_n_experts.items():
            if items:
                lbs = [x[0] for x in items]
                pds = [1 if x[1] >= 0.5 else 0 for x in items]
                curve_labels_per_n[n_exp] = lbs
                curve_preds[n_exp] = pds
                curve_mean_q[n_exp] = n_exp
        from src.evaluation.metrics import compute_accuracy
        speed_curve = []
        for n_exp in sorted(curve_preds.keys()):
            acc = compute_accuracy(curve_labels_per_n[n_exp], curve_preds[n_exp])
            speed_curve.append({"n_experts": n_exp, "accuracy": acc, "mean_queries": n_exp})
        metrics["speed_accuracy_curve"] = speed_curve

        return metrics

    @torch.no_grad()
    def private_protocol_robustness(self, loader: DataLoader) -> Dict:
        """
        Reinitialise judge weights and measure accuracy.
        If accuracy collapses vs original: experts learned a private code.
        """
        self.logger.info("  Running private protocol robustness test ...")
        original_acc = self._run_episodes(loader)["accuracy"]

        # Reinitialise judge
        fresh_judge = copy.deepcopy(self.judge)
        for m in fresh_judge.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()

        orig_judge = self.judge
        self.judge = fresh_judge.to(self.device)
        fresh_acc = self._run_episodes(loader)["accuracy"]
        self.judge = orig_judge

        retention = fresh_acc / max(original_acc, 1e-8)
        private_code_detected = retention < 0.6

        return {
            "original_accuracy": original_acc,
            "fresh_judge_accuracy": fresh_acc,
            "retention_ratio": retention,
            "private_code_detected": private_code_detected,
        }

    def run_full_evaluation(
        self,
        val_loader: DataLoader,
        zeroshot_loader: DataLoader,
    ) -> Dict:
        """Run all evaluations and save results."""
        self.logger.info("=" * 60)
        self.logger.info("FINAL EVALUATION")
        self.logger.info("=" * 60)

        # Seen types
        self.logger.info("  [1/4] Seen document types ...")
        seen_metrics = self._run_episodes(val_loader)
        self.logger.log_eval(seen_metrics, split="seen")

        # Zero-shot
        self.logger.info("  [2/4] Zero-shot (unseen) document types ...")
        zs_metrics = self._run_episodes(zeroshot_loader)
        self.logger.log_eval(zs_metrics, split="zero_shot")

        # Private protocol robustness
        self.logger.info("  [3/4] Private protocol robustness test ...")
        robustness = self.private_protocol_robustness(val_loader)
        self.logger.log_eval(robustness, split="robustness")

        # Shapley values
        self.logger.info("  [4/4] Shapley value attribution ...")
        shapley_computer = ExactShapleyComputer(
            self.judge, self.expert_system, val_loader,
            device=self.device, max_batches=30,
        )
        phi, v_S = shapley_computer.compute()
        core = shapley_computer.core_check(phi, v_S)
        shapley_results = {
            "shapley_values": phi,
            "coalition_values": v_S,
            "core_membership": core,
        }
        self.logger.log_eval(shapley_results, split="shapley")

        # Aggregate
        results = {
            "seen": seen_metrics,
            "zero_shot": zs_metrics,
            "robustness": robustness,
            "shapley": shapley_results,
        }

        # Save to file
        out_path = os.path.join(self.cfg.metrics_dir, "eval_metrics.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        self.logger.info(f"Evaluation results saved -> {out_path}")

        # Save human-readable report
        report_path = os.path.join(self.cfg.metrics_dir, "eval_report.txt")
        self._write_report(results, report_path)
        self.logger.info(f"Report saved -> {report_path}")

        return results

    def _write_report(self, results: Dict, path: str) -> None:
        lines = []
        lines.append("="*60)
        lines.append("GT FORGERY DETECTION — EVALUATION REPORT")
        lines.append("="*60)

        for split in ["seen", "zero_shot"]:
            if split not in results: continue
            m = results[split]
            lines.append(f"\n[{split.upper()}]")
            for k in ["accuracy", "auc", "eer", "ece", "mean_n_queries", "n_samples"]:
                if k in m:
                    v = m[k]
                    lines.append(f"  {k:<25s}: {v:.4f}" if isinstance(v, float) else f"  {k:<25s}: {v}")

        if "robustness" in results:
            r = results["robustness"]
            lines.append(f"\n[PRIVATE PROTOCOL ROBUSTNESS]")
            lines.append(f"  Original accuracy     : {r.get('original_accuracy', 0):.4f}")
            lines.append(f"  Fresh judge accuracy  : {r.get('fresh_judge_accuracy', 0):.4f}")
            lines.append(f"  Retention ratio       : {r.get('retention_ratio', 0):.4f}")
            lines.append(f"  Private code detected : {r.get('private_code_detected', False)}")

        if "shapley" in results:
            s = results["shapley"]
            lines.append(f"\n[SHAPLEY VALUES]")
            phi = s.get("shapley_values", {})
            for name, val in phi.items():
                in_core = s.get("core_membership", {}).get(name, False)
                lines.append(f"  phi_{name:<10s}: {val:.4f}  [in_core={in_core}]")

        lines.append("\n" + "="*60)

        with open(path, "w") as f:
            f.write("\n".join(lines))