from __future__ import annotations
import json, os, sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_new.metrics import compute_all, compute_accuracy, compute_auc, compute_eer, compute_ece

FORGERY_TYPES = [
    "fraud1_copy_and_move",
    "fraud2_face_morphing",
    "fraud3_face_replacement",
    "fraud4_combined",
    "genuine",
]


@torch.no_grad()
def run_evaluation(
    model,
    loader,
    device: str,
) -> Dict:
    model.eval()

    all_labels:       List[int]   = []
    all_scores:       List[float] = []
    all_forgery_type: List[str]   = []
    all_per_query_logits = []
    all_query_weights    = []

    for batch in tqdm(loader, desc="  Evaluating", dynamic_ncols=True):
        labels = batch["label"].to(device)
        ftypes = batch["forgery_type"]

        out    = model(batch)
        belief = out["belief"].cpu()

        all_scores.extend(belief.tolist())
        all_labels.extend(labels.cpu().tolist())
        all_forgery_type.extend(ftypes)
        all_per_query_logits.extend(out["per_query_logits"].cpu().tolist())
        all_query_weights.extend(out["query_weights"].cpu().tolist())

    # -- Overall metrics
    overall = compute_all(all_labels, all_scores)
    preds   = [1 if s >= 0.5 else 0 for s in all_scores]

    # -- Confusion matrix
    cm = confusion_matrix(all_labels, preds, labels=[0, 1]).tolist()
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    overall["confusion_matrix"] = {
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "precision": tp / max(tp + fp, 1),
        "recall":    tp / max(tp + fn, 1),
        "f1":        2*tp / max(2*tp + fp + fn, 1),
    }

    # -- Per forgery type
    per_type: Dict[str, Dict] = {}
    buckets = defaultdict(lambda: {"labels": [], "scores": []})
    for lbl, sc, ft in zip(all_labels, all_scores, all_forgery_type):
        buckets[ft]["labels"].append(lbl)
        buckets[ft]["scores"].append(sc)
    for ft, bucket in buckets.items():
        m = compute_all(bucket["labels"], bucket["scores"])
        # per-type confusion matrix
        pt_preds = [1 if s >= 0.5 else 0 for s in bucket["scores"]]
        pt_cm    = confusion_matrix(bucket["labels"], pt_preds, labels=[0,1]).tolist()
        m["confusion_matrix"] = pt_cm
        per_type[ft] = m

    # -- Query analysis (interpretability)
    q_logits = np.array(all_per_query_logits)  # (N, 16)
    q_weights = np.array(all_query_weights)     # (N, 16)
    query_stats = {
        "mean_logits":  q_logits.mean(axis=0).tolist(),
        "mean_weights": q_weights.mean(axis=0).tolist(),
        "most_active_query": int(q_weights.mean(axis=0).argmax()),
    }

    return {
        "overall":          overall,
        "per_forgery_type": per_type,
        "query_stats":      query_stats,
    }
