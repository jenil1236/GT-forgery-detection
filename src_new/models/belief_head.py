from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class InterpretableBeliefHead(nn.Module):
    """
    Takes Q-Former output (B, n_q, d_model) and produces:
      - per_query_logits: (B, n_q)  each query verdict
      - query_weights:    (B, n_q)  softmax importance weights
      - final_logit:      (B,)      weighted sum
      - belief:           (B,)      sigmoid(final_logit)
    """

    def __init__(self, d_model: int = 256, hidden: int = 128):
        super().__init__()
        # Shared belief MLP applied to each query independently
        self.belief_mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        # Weight MLP: how important is each query
        self.weight_mlp = nn.Linear(d_model, 1)

    def forward(self, queries: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            queries: (B, n_q, d_model) from Q-Former
        Returns dict:
            per_query_logits: (B, n_q)
            query_weights:    (B, n_q)  sums to 1
            final_logit:      (B,)
            belief:           (B,)  in [0,1]
        """
        # Per-query logits: (B, n_q, 1) -> (B, n_q)
        per_q_logits = self.belief_mlp(queries).squeeze(-1)
        # Weights: (B, n_q, 1) -> (B, n_q) -> softmax
        raw_weights  = self.weight_mlp(queries).squeeze(-1)
        weights      = F.softmax(raw_weights, dim=-1)
        # Weighted sum: (B,)
        final_logit  = (per_q_logits * weights).sum(dim=-1)
        belief       = torch.sigmoid(final_logit)
        return {
            "per_query_logits": per_q_logits,
            "query_weights":    weights,
            "final_logit":      final_logit,
            "belief":           belief,
        }
