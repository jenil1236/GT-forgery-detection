"""
judge.py - Log-odds accumulator judge (Table 1, option c from proposal).

Architecture:
  State:  lambda_t (log-odds scalar), step_t (int), b_t (belief = sigma(lambda_t))
  Update: lambda_{t+1} = lambda_t + Delta_i  where Delta_i = log(q_i / (1 - q_i))
  Action: {query_expert_0, query_expert_1, query_expert_2, declare}  (4 actions)
  Heads:
    - probability head: outputs b_t (for proper scoring rule loss)
    - query-policy head (actor): 2-layer MLP -> 4 logits
    - value head (centralised critic): 2-layer MLP -> scalar
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


DECLARE_ACTION = 3   # index of 'declare' in action space
N_JUDGE_ACTIONS = 4  # query_0, query_1, query_2, declare
EPS = 1e-6           # numerical stability


class LogOddsJudge(nn.Module):
    """
    Log-odds accumulator judge.

    The judge state is the running log-odds lambda_t.
    Expert beliefs q_i are converted to log-likelihood ratios
    and accumulated: lambda_t += log(q_i / (1 - q_i)).

    The policy (which expert to query next or whether to declare)
    is learned by a small MLP over (lambda_t, b_t, step_t / H).
    """

    def __init__(
        self,
        n_experts: int = 3,
        hidden_dim: int = 64,
        query_cost: float = 0.1,
        prior_log_odds: float = 0.0,   # log-odds corresponding to 50/50 prior
    ):
        super().__init__()
        self.n_experts = n_experts
        self.query_cost = query_cost
        self.prior_log_odds = prior_log_odds
        n_actions = n_experts + 1  # query each expert + declare

        # State feature: [lambda_t, b_t, step/H, ...per-expert beliefs received so far]
        # We use 2 + n_experts features (lambda, b, normalised_step, + n_expert slots)
        state_dim = 2 + 1 + n_experts  # lambda, b, step/H, expert_beliefs

        # Policy (actor) head
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, n_actions),
        )

        # Value (critic) head - takes richer state (also sees all expert embeddings)
        # For CTDE the critic gets extra info at training time
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        # Probability head: reports judge's current belief (for proper scoring rule)
        # This is just sigma(lambda_t) -- no learned params needed, but we add
        # a small calibration bias for flexibility
        self.calibration_bias = nn.Parameter(torch.zeros(1))

    def _make_state(
        self,
        lambda_t: torch.Tensor,         # (B,) current log-odds
        step: int,
        max_steps: int,
        expert_beliefs: torch.Tensor,   # (B, n_experts) beliefs received; -1 if not yet queried
    ) -> torch.Tensor:
        """Build state feature vector for the policy/value networks."""
        B = lambda_t.size(0)
        b_t = torch.sigmoid(lambda_t)   # (B,)
        step_norm = torch.full((B,), step / max(max_steps, 1),
                               device=lambda_t.device)
        state = torch.stack([lambda_t, b_t, step_norm], dim=-1)  # (B, 3)
        state = torch.cat([state, expert_beliefs], dim=-1)         # (B, 3+n_experts)
        return state

    def forward(
        self,
        lambda_t: torch.Tensor,
        step: int,
        max_steps: int,
        expert_beliefs: torch.Tensor,
        available_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute judge action and value for one episode step.

        Args:
            lambda_t:       (B,) running log-odds
            step:           current step index (int)
            max_steps:      max episode steps (int)
            expert_beliefs: (B, n_experts) beliefs received; 0.5 if expert not yet queried
            available_mask: (B, n_actions) bool, False = unavailable (already queried)

        Returns dict:
            action:         (B,) sampled action index
            log_prob:       (B,) log probability of action
            entropy:        (B,) entropy of action distribution
            value:          (B,) state value estimate
            belief:         (B,) current probability of forgery (sigma(lambda_t + bias))
            action_logits:  (B, n_actions) raw logits
        """
        state = self._make_state(lambda_t, step, max_steps, expert_beliefs)
        logits = self.policy_net(state)  # (B, n_actions)

        # Mask unavailable actions (already-queried experts)
        if available_mask is not None:
            logits = logits.masked_fill(~available_mask, float("-inf"))

        # Force declare at final step
        if step >= max_steps - 1:
            logits = torch.full_like(logits, float("-inf"))
            logits[:, DECLARE_ACTION] = 0.0

        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()          # (B,)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        value = self.value_net(state).squeeze(-1)  # (B,)

        belief = torch.sigmoid(lambda_t + self.calibration_bias)  # (B,)

        return {
            "action": action,
            "log_prob": log_prob,
            "entropy": entropy,
            "value": value,
            "belief": belief,
            "action_logits": logits,
        }

    def compute_log_prob_entropy(
        self,
        lambda_t: torch.Tensor,
        step: int,
        max_steps: int,
        expert_beliefs: torch.Tensor,
        stored_action: torch.Tensor,
        available_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Re-compute log_prob and entropy for PPO update.
        Returns: (log_prob, entropy, value)
        """
        state = self._make_state(lambda_t, step, max_steps, expert_beliefs)
        logits = self.policy_net(state)
        if available_mask is not None:
            logits = logits.masked_fill(~available_mask, float("-inf"))
        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(stored_action)
        entropy = dist.entropy()
        value = self.value_net(state).squeeze(-1)
        return log_prob, entropy, value

    @staticmethod
    def update_lambda(
        lambda_t: torch.Tensor,
        belief: torch.Tensor,
        eps: float = EPS,
    ) -> torch.Tensor:
        """
        Update log-odds with a new expert belief.
        lambda_{t+1} = lambda_t + log(q / (1 - q))

        Args:
            lambda_t: (B,) current log-odds
            belief:   (B, 1) or (B,) expert belief in [0,1]
        """
        q = belief.squeeze(-1).clamp(eps, 1.0 - eps)
        llr = torch.log(q / (1.0 - q))
        return lambda_t + llr

    @staticmethod
    def get_decision(belief: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Convert belief to binary decision. Returns (B,) LongTensor {0=genuine, 1=forged}."""
        return (belief >= threshold).long()