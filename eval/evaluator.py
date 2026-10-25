from __future__ import annotations
import json, os, sys, time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.judge import LogOddsJudge, DECLARE_ACTION
from eval.metrics  import compute_all
from eval.shapley  import compute_shapley_for_sample, aggregate_shapley

MODALITIES = ["visual", "text", "signal"]


@torch.no_grad()
def run_evaluation(
    expert_system,
    judge: LogOddsJudge,
    loader,
    device: str,
    max_episode_steps: int = 4,
    query_cost: float = 0.1,
) -> Dict:
    expert_system.eval()
    judge.eval()

    all_labels:       List[int]   = []
    all_scores:       List[float] = []
    all_n_queries:    List[float] = []
    all_forgery_type: List[str]   = []
    per_sample_phi:   List[Dict]  = []

    for batch in tqdm(loader, desc="  Evaluating", dynamic_ncols=True):
        batch_device = {k: v.to(device) if torch.is_tensor(v) else v
                        for k, v in batch.items()}
        B      = batch_device["label"].size(0)
        labels = batch_device["label"]
        ftypes = batch["forgery_type"]

        lambda_t = torch.zeros(B, device=device)
        eb_slot  = torch.full((B, 3), 0.5, device=device)
        avail    = torch.ones(B, 4, dtype=torch.bool, device=device)
        done     = torch.zeros(B, dtype=torch.bool, device=device)
        n_q      = torch.zeros(B, device=device)

        eo = expert_system(batch_device, deterministic=True)

        # sanitise NaN/Inf
        for nm in eo:
            for key in ("eR", "eT", "belief"):
                if key in eo[nm] and torch.is_tensor(eo[nm][key]):
                    if torch.isnan(eo[nm][key]).any() or torch.isinf(eo[nm][key]).any():
                        eo[nm][key] = torch.nan_to_num(
                            eo[nm][key], nan=0.0, posinf=1.0, neginf=-1.0)

        for step in range(max_episode_steps):
            jout   = judge(lambda_t=lambda_t, step=step,
                           max_steps=max_episode_steps,
                           expert_beliefs=eb_slot, available_mask=avail)
            action = jout["action"]
            nd     = done.clone()
            for b in range(B):
                if done[b]: continue
                a       = action[b].item()
                is_last = (step == max_episode_steps - 1)
                if a == DECLARE_ACTION or is_last:
                    nd[b] = True
                elif a < 3:
                    nm = MODALITIES[a]
                    q  = eo[nm]["belief"][b].squeeze().clamp(1e-7, 1-1e-7)
                    lambda_t[b] = LogOddsJudge.update_lambda(
                        lambda_t[b:b+1], q.unsqueeze(0)).squeeze()
                    eb_slot[b, a] = q
                    avail[b, a]   = False
                    n_q[b]       += 1
            done = nd
            if done.all(): break

        beliefs = jout["belief"].cpu()
        all_scores.extend(beliefs.tolist())
        all_labels.extend(labels.cpu().tolist())
        all_n_queries.extend(n_q.cpu().tolist())
        all_forgery_type.extend(ftypes)

        # per-sample Shapley values
        for b in range(B):
            eb = {MODALITIES[i]: float(eb_slot[b, i].cpu()) for i in range(3)}
            phi = compute_shapley_for_sample(eb)
            per_sample_phi.append({
                **phi,
                "label":        int(labels[b].cpu()),
                "forgery_type": ftypes[b],
                "score":        float(beliefs[b]),
                "n_queries":    float(n_q[b].cpu()),
            })

    # ── overall metrics ──────────────────────────────────────────────────────
    overall = compute_all(all_labels, all_scores)
    overall["mean_n_queries"] = float(sum(all_n_queries) / max(len(all_n_queries), 1))

    # ── per-forgery-type metrics ──────────────────────────────────────────────
    per_type: Dict[str, Dict] = {}
    type_buckets: Dict[str, Dict] = defaultdict(lambda: {"labels": [], "scores": []})
    for lbl, sc, ft in zip(all_labels, all_scores, all_forgery_type):
        type_buckets[ft]["labels"].append(lbl)
        type_buckets[ft]["scores"].append(sc)
    for ft, bucket in type_buckets.items():
        per_type[ft] = compute_all(bucket["labels"], bucket["scores"])

    # ── Shapley aggregates ────────────────────────────────────────────────────
    phi_all    = aggregate_shapley(per_sample_phi)
    phi_forged = aggregate_shapley([p for p in per_sample_phi if p["label"] == 1])
    phi_genuine= aggregate_shapley([p for p in per_sample_phi if p["label"] == 0])

    return {
        "overall":         overall,
        "per_forgery_type": per_type,
        "shapley_all":      phi_all,
        "shapley_forged":   phi_forged,
        "shapley_genuine":  phi_genuine,
        "per_sample_shapley": per_sample_phi,
    }
