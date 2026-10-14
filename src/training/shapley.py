"""
shapley.py - Exact Shapley value computation for n=3 experts.

With n=3 there are exactly 8 subsets (2^3) and 6 permutations.
No Monte Carlo approximation needed.

v(S) = expected AUC using only experts in S (others replaced with mean embedding).
"""

from __future__ import annotations
from itertools import permutations
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from src.models.agents import MultiExpertSystem
from src.models.judge import LogOddsJudge


ALL_EXPERTS = [0, 1, 2]
ALL_SUBSETS = [
    [],
    [0], [1], [2],
    [0, 1], [0, 2], [1, 2],
    [0, 1, 2],
]


def _subset_key(subset: List[int]) -> str:
    return "{" + ",".join(str(i) for i in sorted(subset)) + "}"


class ExactShapleyComputer:
    """
    Computes exact Shapley values for 3 experts.

    Usage:
        computer = ExactShapleyComputer(judge, expert_system, val_loader, device)
        phi, v_S = computer.compute()
    """

    def __init__(
        self,
        judge: LogOddsJudge,
        expert_system: MultiExpertSystem,
        val_loader: DataLoader,
        device: str = "cuda",
        max_batches: int = 50,  # limit for speed
    ):
        self.judge = judge
        self.expert_system = expert_system
        self.val_loader = val_loader
        self.device = device
        self.max_batches = max_batches

        # Store mean embeddings for replacing absent experts
        self._mean_embeddings: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None

    @torch.no_grad()
    def _compute_mean_embeddings(self) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute mean (eR, eT) per expert over val set for use as default."""
        if self._mean_embeddings is not None:
            return self._mean_embeddings

        modalities = ["visual", "text", "signal"]
        sums_R = [torch.zeros(128) for _ in range(3)]  # kept on CPU intentionally (accumulated then divided)
        sums_T = [torch.zeros(128) for _ in range(3)]
        count = 0

        # FIX: MultiExpertSystem.forward now handles device transfer internally,
        # but we still need batch on the correct device for the loop
        try:
            _device = next(self.expert_system.parameters()).device
        except StopIteration:
            _device = torch.device("cpu")
        for i, batch in enumerate(self.val_loader):
            if i >= self.max_batches:
                break
            batch = {k: v.to(_device) if torch.is_tensor(v) else v for k, v in batch.items()}
            outputs = self.expert_system(batch, deterministic=True)
            for idx, name in enumerate(modalities):
                if name in outputs:
                    sums_R[idx] += outputs[name]["eR"].mean(dim=0).cpu()
                    sums_T[idx] += outputs[name]["eT"].mean(dim=0).cpu()
            count += 1

        means = {}
        for idx in range(3):
            mR = (sums_R[idx] / max(count, 1)).unsqueeze(0)  # (1, D)
            mT = (sums_T[idx] / max(count, 1)).unsqueeze(0)
            means[idx] = (mR, mT)

        self._mean_embeddings = means
        return means

    @torch.no_grad()
    def _evaluate_subset(self, subset: List[int]) -> float:
        """
        Evaluate judge AUC using only the experts in subset.
        Missing experts are replaced with their mean embedding.
        Returns AUC score.
        """
        if not subset:
            return 0.5  # random — no information

        modalities = ["visual", "text", "signal"]
        mean_embs = self._compute_mean_embeddings()

        all_beliefs = []
        all_labels = []

        # FIX: determine expert device and move batch accordingly
        try:
            _device = next(self.expert_system.parameters()).device
        except StopIteration:
            _device = torch.device("cpu")

        for i, batch in enumerate(self.val_loader):
            if i >= self.max_batches:
                break

            batch = {k: v.to(_device) if torch.is_tensor(v) else v for k, v in batch.items()}
            labels = batch["label"].cpu()  # keep labels on CPU for sklearn
            B = labels.size(0)

            outputs = self.expert_system(batch, deterministic=True)

            # Run judge episode using only experts in subset
            lambda_t = torch.zeros(B, device="cpu")
            expert_beliefs_slot = torch.full((B, 3), 0.5)  # default = 0.5 (uninformative)

            for expert_idx in subset:
                name = modalities[expert_idx]
                if name in outputs:
                    q = outputs[name]["belief"].squeeze(-1).cpu()  # (B,) on CPU
                else:
                    q = torch.full((B,), 0.5)

                lambda_t = LogOddsJudge.update_lambda(lambda_t, q.unsqueeze(-1))
                expert_beliefs_slot[:, expert_idx] = q

            # Final judge belief ? move to CPU before .numpy()
            final_belief = torch.sigmoid(lambda_t).cpu().numpy()
            all_beliefs.extend(final_belief.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

        if len(set(all_labels)) < 2:
            return 0.5  # degenerate

        try:
            return float(roc_auc_score(all_labels, all_beliefs))
        except Exception:
            return 0.5

    def compute(self) -> Tuple[Dict[int, float], Dict[str, float]]:
        """
        Compute exact Shapley values for all 3 experts.

        Shapley formula:
            phi_i = sum over S not containing i of:
                (|S|! * (n - |S| - 1)! / n!) * (v(S union {i}) - v(S))

        Returns:
            phi:  dict {expert_idx: shapley_value}
            v_S:  dict {subset_key: coalition_value}
        """
        n = 3
        print("  Computing exact Shapley values ...")

        # Step 1: compute v(S) for all 8 subsets
        v_S: Dict[str, float] = {}
        for subset in ALL_SUBSETS:
            key = _subset_key(subset)
            v = self._evaluate_subset(subset)
            v_S[key] = v
            print(f"    v({key}) = {v:.4f}")

        # Step 2: exact Shapley formula
        import math
        phi = {i: 0.0 for i in range(n)}

        for i in range(n):
            others = [j for j in ALL_EXPERTS if j != i]
            # Iterate over all subsets S of {0,1,2} \ {i}
            for size in range(n):  # |S| = 0, 1, 2
                if size == 0:
                    subsets_S = [[]]
                elif size == 1:
                    subsets_S = [[j] for j in others]
                else:
                    subsets_S = [others]

                coeff = (math.factorial(size) * math.factorial(n - size - 1)
                         / math.factorial(n))
                for S in subsets_S:
                    S_union_i = sorted(S + [i])
                    v_with    = v_S[_subset_key(S_union_i)]
                    v_without = v_S[_subset_key(S)]
                    phi[i] += coeff * (v_with - v_without)

        # Step 3: verify efficiency (sum of phi = v(grand coalition))
        v_grand = v_S[_subset_key([0, 1, 2])]
        phi_sum = sum(phi.values())

        print(f"\n  Shapley Values:")
        names = ["visual", "text", "signal"]
        for i in range(n):
            core_ok = phi[i] >= v_S[_subset_key([i])]
            print(f"    phi_{names[i]} = {phi[i]:.4f}  "
                  f"(v({{i}})={v_S[_subset_key([i])]:.4f}, in_core={core_ok})")
        print(f"  Sum(phi) = {phi_sum:.4f}, v(N) = {v_grand:.4f}, "
              f"efficiency_error = {abs(phi_sum - v_grand):.6f}")

        # Annotate phi with names
        named_phi = {names[i]: phi[i] for i in range(n)}

        return named_phi, v_S

    def core_check(self, phi: Dict[int, float], v_S: Dict[str, float]) -> Dict[str, bool]:
        """
        Check if the Shapley allocation lies in the core.
        Core conditions: phi_i >= v({i}) for all i.
        """
        names = ["visual", "text", "signal"]
        result = {}
        for i in range(3):
            name = names[i]
            phi_i = phi.get(name, phi.get(i, 0.0))
            vi = v_S[_subset_key([i])]
            result[name] = phi_i >= vi
        return result