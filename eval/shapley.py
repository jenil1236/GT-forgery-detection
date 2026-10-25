from __future__ import annotations
import math
from typing import Dict, List
import numpy as np

ALL_SUBSETS = [[], [0], [1], [2], [0,1], [0,2], [1,2], [0,1,2]]
MODALITIES  = ["visual", "text", "signal"]


def compute_shapley_for_sample(
    expert_beliefs: Dict[str, float],
    prior: float = 0.5,
) -> Dict[str, float]:
    n = 3

    def lo(p):
        p = max(1e-7, min(1.0 - 1e-7, p))
        return math.log(p / (1.0 - p))

    prior_lo = lo(prior)

    def v(subset):
        if not subset:
            return prior
        lam = prior_lo + sum(lo(expert_beliefs[MODALITIES[i]]) for i in subset)
        return 1.0 / (1.0 + math.exp(-lam))

    v_S = {tuple(sorted(s)): v(s) for s in ALL_SUBSETS}

    phi = {}
    for i in range(n):
        others = [j for j in range(n) if j != i]
        phi_i = 0.0
        for size in range(n):
            if size == 0:
                subs = [[]]
            elif size == 1:
                subs = [[j] for j in others]
            else:
                subs = [others]
            coeff = (math.factorial(size) * math.factorial(n - size - 1)
                     / math.factorial(n))
            for S in subs:
                phi_i += coeff * (
                    v_S[tuple(sorted(S + [i]))] - v_S[tuple(sorted(S))]
                )
        phi[MODALITIES[i]] = phi_i
    return phi


def aggregate_shapley(per_sample: List[Dict[str, float]]) -> Dict:
    if not per_sample:
        return {}
    result = {}
    for k in per_sample[0]:
        vals = [d[k] for d in per_sample]
        result[k] = {
            "mean": float(np.mean(vals)),
            "std":  float(np.std(vals)),
            "min":  float(np.min(vals)),
            "max":  float(np.max(vals)),
        }
    return result
