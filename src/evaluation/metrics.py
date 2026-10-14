import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from typing import Dict, List, Tuple


def compute_auc(labels: List[int], scores: List[float]) -> float:
    if len(set(labels)) < 2:
        return 0.5
    try:
        return float(roc_auc_score(labels, scores))
    except Exception:
        return 0.5


def compute_eer(labels: List[int], scores: List[float]) -> float:
    if len(set(labels)) < 2:
        return 0.5
    try:
        fpr, tpr, thresholds = roc_curve(labels, scores)
        fnr = 1.0 - tpr
        eer_idx = np.argmin(np.abs(fpr - fnr))
        return float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    except Exception:
        return 0.5


def compute_ece(labels: List[int], probs: List[float], n_bins: int = 15) -> float:
    labels_arr = np.array(labels)
    probs_arr = np.array(probs)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs_arr >= bins[i]) & (probs_arr < bins[i + 1])
        if mask.sum() == 0:
            continue
        avg_conf = probs_arr[mask].mean()
        avg_acc = labels_arr[mask].mean()
        ece += (mask.sum() / len(labels_arr)) * abs(avg_conf - avg_acc)
    return float(ece)


def compute_accuracy(labels: List[int], preds: List[int]) -> float:
    if not labels:
        return 0.0
    return float(sum(l == p for l, p in zip(labels, preds)) / len(labels))


def speed_accuracy_curve(
    labels: List[int],
    preds_per_n_experts: Dict[int, List[int]],
    mean_queries: Dict[int, float],
) -> List[Dict]:
    curve = []
    for n, preds in sorted(preds_per_n_experts.items()):
        acc = compute_accuracy(labels, preds)
        curve.append({"n_experts": n, "accuracy": acc, "mean_queries": mean_queries.get(n, n)})
    return curve


def compute_all_metrics(
    labels: List[int],
    scores: List[float],
    n_bins: int = 15,
) -> Dict:
    preds = [1 if s >= 0.5 else 0 for s in scores]
    return {
        "accuracy": compute_accuracy(labels, preds),
        "auc": compute_auc(labels, scores),
        "eer": compute_eer(labels, scores),
        "ece": compute_ece(labels, scores, n_bins=n_bins),
        "n_samples": len(labels),
        "n_positive": int(sum(labels)),
        "n_negative": int(len(labels) - sum(labels)),
    }