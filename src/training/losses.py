"""
losses.py - All loss functions and reward computations.

  InfoNCELoss              - contrastive pretraining (Stage 1)
  ContrastiveAlignmentLoss - push eR-eT apart for forged, together for genuine
  ProperScoringRuleLoss    - judge training signal (log score)
  MarketScoringRuleReward  - expert credit assignment
  DecorrelationPenalty     - prevent expert collapse
"""

from __future__ import annotations
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1e-7


# ── Stage 1 losses ────────────────────────────────────────────────────────────

class InfoNCELoss(nn.Module):
    """
    InfoNCE / NT-Xent contrastive loss.

    For each anchor (reference embedding), the positive is the genuine test
    embedding of the same document and negatives are all other test embeddings
    in the batch (both forged and different documents).

    Reference: Oord et al. 2018 / Chen et al. SimCLR 2020.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        anchors: torch.Tensor,    # (B, D) reference embeddings
        positives: torch.Tensor,  # (B, D) genuine test embeddings (same doc type)
        negatives: Optional[torch.Tensor] = None,  # (B*k, D) extra negatives (forged)
    ) -> torch.Tensor:
        """
        If negatives is None, uses in-batch negatives only (other positives).
        """
        B = anchors.size(0)
        # Concat positives and optional extra negatives
        if negatives is not None:
            all_keys = torch.cat([positives, negatives], dim=0)  # (B + B*k, D)
        else:
            all_keys = positives  # (B, D)

        # Cosine similarity matrix
        sim = torch.mm(anchors, all_keys.T) / self.temperature  # (B, N)

        # Labels: position i is the positive for anchor i
        labels = torch.arange(B, device=anchors.device)

        loss = F.cross_entropy(sim, labels)
        return loss


class TripletLoss(nn.Module):
    """
    Triplet margin loss.
    anchor = reference, positive = genuine test, negative = forged test.
    """

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.loss_fn = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(
        self,
        anchor: torch.Tensor,    # (B, D)
        positive: torch.Tensor,  # (B, D)
        negative: torch.Tensor,  # (B, D)
    ) -> torch.Tensor:
        return self.loss_fn(anchor, positive, negative)


class ContrastiveAlignmentLoss(nn.Module):
    """
    For genuine pairs:  minimise ||eR - eT||_2
    For forged pairs:   maximise ||eR - eT||_2 (up to a margin)

    This is the 'contrastive alignment' shaping term from the proposal.
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        eR: torch.Tensor,     # (B, D) reference embeddings
        eT: torch.Tensor,     # (B, D) test embeddings
        labels: torch.Tensor, # (B,) 0=genuine, 1=forged
    ) -> torch.Tensor:
        dist = F.pairwise_distance(eR, eT, p=2)  # (B,)

        genuine = (labels == 0).float()
        forged  = (labels == 1).float()

        # Genuine: minimise distance
        loss_genuine = genuine * dist.pow(2)

        # Forged: penalise if distance < margin
        loss_forged = forged * F.relu(self.margin - dist).pow(2)

        return (loss_genuine + loss_forged).mean()


# ── Stage 2 losses / rewards ──────────────────────────────────────────────────

class ProperScoringRuleLoss(nn.Module):
    """
    Strictly proper scoring rule for the judge.
    Loss = -log q_J(y) + c * n_queries + asymmetric FP/FN costs

    Minimising this is equivalent to maximising:
        r_J = log q_J(y) - c * n_queries - FP/FN penalties

    Args:
        query_cost:        cost per query c
        cost_fn:           cost of false negative (missing forgery)
        cost_fp:           cost of false positive (false alarm)
    """

    def __init__(
        self,
        query_cost: float = 0.1,
        cost_fn: float = 2.0,
        cost_fp: float = 1.0,
    ):
        super().__init__()
        self.query_cost = query_cost
        self.cost_fn = cost_fn
        self.cost_fp = cost_fp

    def forward(
        self,
        belief: torch.Tensor,          # (B,) judge's belief q_J
        labels: torch.Tensor,          # (B,) ground truth {0, 1}
        n_queries: torch.Tensor,       # (B,) number of queries used
        predictions: Optional[torch.Tensor] = None,  # (B,) binary predictions
    ) -> torch.Tensor:
        q = belief.clamp(EPS, 1.0 - EPS)
        y = labels.float()

        # Log-score: E[log q(y)] — proper scoring rule
        log_score = y * torch.log(q) + (1 - y) * torch.log(1.0 - q)

        # Query cost
        query_penalty = self.query_cost * n_queries.float()

        # FP/FN asymmetric cost (applied after decision)
        asym_cost = torch.zeros_like(q)
        if predictions is not None:
            pred = predictions.float()
            # False negative: y=1, pred=0
            fn_mask = ((y == 1) & (pred == 0)).float()
            # False positive: y=0, pred=1
            fp_mask = ((y == 0) & (pred == 1)).float()
            asym_cost = self.cost_fn * fn_mask + self.cost_fp * fp_mask

        # Loss = negative reward
        reward = log_score - query_penalty - asym_cost
        return -reward.mean()


def market_scoring_rule_rewards(
    beliefs: torch.Tensor,    # (B, T) running beliefs after each expert, T = n queried
    prior: float = 0.5,
    labels: torch.Tensor = None,  # (B,)
) -> torch.Tensor:
    """
    Compute per-expert MSR rewards.

    r_t = log b_t(y) - log b_{t-1}(y)

    The payments telescope:
        sum(r_t) = log b_K(y) - log b_0(y)

    Args:
        beliefs: (B, T) beliefs after each of T expert contributions
        prior:   initial belief b_0 (scalar)
        labels:  (B,) ground truth labels

    Returns:
        rewards: (B, T) per-step per-sample rewards
    """
    if labels is None:
        raise ValueError("labels required for MSR rewards")

    B, T = beliefs.shape
    y = labels.float()  # (B,)

    # b_{t-1}: prepend prior column
    prior_col = torch.full((B, 1), prior, device=beliefs.device)
    prev_beliefs = torch.cat([prior_col, beliefs[:, :-1]], dim=1)  # (B, T)

    def log_score(b, y_):
        b_ = b.clamp(EPS, 1.0 - EPS)
        return y_.unsqueeze(1) * torch.log(b_) + (1 - y_).unsqueeze(1) * torch.log(1 - b_)

    r = log_score(beliefs, y) - log_score(prev_beliefs, y)  # (B, T)
    return r  # (B, T)


class DecorrelationPenalty(nn.Module):
    """
    Penalises correlation between expert embeddings.
    Without this, experts collapse to the same features and
    the Shapley analysis becomes meaningless.

    Penalty = mean(|corr(e_i, e_j)|) for i != j
    """

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (B, n_experts, embed_dim)
        Returns:
            scalar penalty
        """
        B, N, D = embeddings.shape
        # Mean-centre each expert's embeddings across batch
        emb_c = embeddings - embeddings.mean(dim=0, keepdim=True)  # (B, N, D)

        total_corr = torch.tensor(0.0, device=embeddings.device)
        count = 0
        for i in range(N):
            for j in range(i + 1, N):
                # Correlation between expert i and j over batch
                ei = emb_c[:, i, :]  # (B, D)
                ej = emb_c[:, j, :]  # (B, D)
                # Pearson: mean of cosine similarity over feature dimension
                corr = F.cosine_similarity(ei, ej, dim=-1).abs().mean()
                total_corr = total_corr + corr
                count += 1

        return total_corr / max(count, 1)