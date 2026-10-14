"""
agents.py - Expert agent wrappers with belief reporting.

Each ExpertAgent:
  - Wraps a modality encoder
  - Produces (eR, eT, belief, action_logits, action)
  - belief q in [0,1]: probability test image is forged
  - action: discrete {attend_global=0, attend_local=1, stop=2}

Actions keep the RL space small (proposal Section 2 requirement).
"""

from __future__ import annotations
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertAgent(nn.Module):
    """
    Wraps an encoder and adds:
      - Belief head:   concat(eR, eT) -> q in (0,1)
      - Control head:  concat(eR, eT) -> 3-way discrete action logits

    Args:
        encoder:      any encoder with forward(x) -> (B, embed_dim) L2-normalised
        modality:     name string for logging ('visual'/'text'/'signal')
        embed_dim:    encoder output dimension
        belief_hidden: hidden size of belief/control MLPs
    """

    def __init__(
        self,
        encoder: nn.Module,
        modality: str,
        embed_dim: int = 128,
        belief_hidden: int = 128,
    ):
        super().__init__()
        self.encoder = encoder
        self.modality = modality
        self.embed_dim = embed_dim
        combined = 2 * embed_dim  # concat(eR, eT)

        # Shared trunk for belief + action
        self.trunk = nn.Sequential(
            nn.Linear(combined, belief_hidden),
            nn.LayerNorm(belief_hidden),
            nn.ReLU(inplace=True),
        )

        # Belief head: probability test is forged
        self.belief_head = nn.Sequential(
            nn.Linear(belief_hidden, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Control action head: {attend_global, attend_local, stop}
        self.action_head = nn.Linear(belief_hidden, 3)

    def forward(
        self,
        ref: torch.Tensor,
        test: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            ref:  (B, C, H, W) reference image tensor (modality-specific)
            test: (B, C, H, W) test image tensor

        Returns dict:
            eR:           (B, embed_dim) reference embedding
            eT:           (B, embed_dim) test embedding
            belief:       (B, 1) probability of forgery
            action_logits:(B, 3) raw control action logits
            action:       (B,) sampled discrete action
            log_prob:     (B,) log probability of sampled action
        """
        eR = self.encoder(ref)   # (B, embed_dim)
        eT = self.encoder(test)  # (B, embed_dim)

        combined = torch.cat([eR, eT], dim=-1)  # (B, 2*embed_dim)
        trunk_out = self.trunk(combined)

        belief = self.belief_head(trunk_out)  # (B, 1)

        action_logits = self.action_head(trunk_out)  # (B, 3)
        action_dist = torch.distributions.Categorical(logits=action_logits)
        action = action_dist.sample()  # (B,)
        log_prob = action_dist.log_prob(action)  # (B,)

        return {
            "eR": eR,
            "eT": eT,
            "belief": belief,
            "action_logits": action_logits,
            "action": action,
            "log_prob": log_prob,
        }

    def forward_deterministic(
        self,
        ref: torch.Tensor,
        test: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Greedy (no sampling) forward pass for evaluation."""
        eR = self.encoder(ref)
        eT = self.encoder(test)
        combined = torch.cat([eR, eT], dim=-1)
        trunk_out = self.trunk(combined)
        belief = self.belief_head(trunk_out)
        action_logits = self.action_head(trunk_out)
        action = action_logits.argmax(dim=-1)
        return {
            "eR": eR,
            "eT": eT,
            "belief": belief,
            "action_logits": action_logits,
            "action": action,
        }

    def compute_log_prob(
        self, trunk_out: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Re-compute log_prob for PPO update (given stored actions)."""
        action_logits = self.action_head(trunk_out)
        dist = torch.distributions.Categorical(logits=action_logits)
        return dist.log_prob(action), dist.entropy()


class MultiExpertSystem(nn.Module):
    """
    Container for all 3 expert agents.
    Runs all experts in parallel and collects outputs.
    """

    def __init__(self, agents: Dict[str, ExpertAgent]):
        super().__init__()
        self.agents = nn.ModuleDict(agents)
        self.modality_order = list(agents.keys())  # deterministic order

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        deterministic: bool = False,
    ) -> Dict[str, Dict]:
        """
        Run all expert agents on a batch.

        Args:
            batch: dict with keys ref_visual, ref_text, ref_signal,
                                   test_visual, test_text, test_signal
            deterministic: if True, use greedy actions

        Returns:
            dict mapping modality_name -> agent output dict
        """
        modality_map = {
            "visual": ("ref_visual", "test_visual"),
            "text":   ("ref_text",   "test_text"),
            "signal": ("ref_signal", "test_signal"),
        }

        # FIX: move inputs to same device as model parameters
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        results = {}
        for name, agent in self.agents.items():
            if name not in modality_map:
                continue
            ref_key, test_key = modality_map[name]
            ref = batch[ref_key].to(device) if torch.is_tensor(batch[ref_key]) else batch[ref_key]
            test = batch[test_key].to(device) if torch.is_tensor(batch[test_key]) else batch[test_key]
            if deterministic:
                out = agent.forward_deterministic(ref, test)
            else:
                out = agent(ref, test)
            results[name] = out

        return results

    def get_beliefs(self, expert_outputs: Dict[str, Dict]) -> torch.Tensor:
        """
        Extract beliefs from all experts into a (B, n_experts) tensor.
        Order follows self.modality_order.
        """
        beliefs = []
        for name in self.modality_order:
            if name in expert_outputs:
                b = expert_outputs[name]["belief"]  # (B, 1)
                beliefs.append(b)
        return torch.cat(beliefs, dim=-1)  # (B, n_experts)

    def get_embeddings(
        self, expert_outputs: Dict[str, Dict]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Stack embeddings: returns eR_stack (B, n_experts, embed_dim),
                                         eT_stack (B, n_experts, embed_dim)
        """
        eRs, eTs = [], []
        for name in self.modality_order:
            if name in expert_outputs:
                eRs.append(expert_outputs[name]["eR"].unsqueeze(1))
                eTs.append(expert_outputs[name]["eT"].unsqueeze(1))
        eR_stack = torch.cat(eRs, dim=1)  # (B, n_experts, embed_dim)
        eT_stack = torch.cat(eTs, dim=1)
        return eR_stack, eT_stack