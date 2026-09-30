"""Server-side aggregation of LoRA updates (FR-01, FR-05).

Design choice (PRD open question, see BRAIN.md decision log): the default
"factors" mode averages A and B separately, i.e. an approximation of averaging
the product BA, because mean(B)·mean(A) != mean(B·A). "product" mode averages
the true products and re-factorises to rank r with a truncated SVD, which is
more faithful but needs the (d_out x d_in) product at the server.
"""
from __future__ import annotations

import numpy as np

from fedlora.model import Adapter


def weights_from_sizes(buckets_or_sizes: list[float]) -> np.ndarray:
    w = np.asarray(buckets_or_sizes, dtype=float)
    return w / w.sum()


def bucketed_weight(n: int, noise: float = 0.0, rng=None) -> float:
    """FedAvg weight from a coarsened (log2-bucketed, optionally noised) size."""
    b = 2 ** np.round(np.log2(max(n, 1)))
    if noise > 0 and rng is not None:
        b = max(1.0, b * (1 + rng.normal(0, noise)))
    return float(b)


def fed_avg_factors(global_ad: Adapter, mean_delta: np.ndarray) -> Adapter:
    """FedAvg/FedProx server step: apply the (weighted) mean delta to A and B."""
    return Adapter.from_flat(global_ad.flat() + mean_delta, global_ad)


def refactor(product: np.ndarray, like: Adapter) -> Adapter:
    r = like.A.shape[0]
    u, s, vt = np.linalg.svd(product, full_matrices=False)
    root = np.sqrt(s[:r])
    return Adapter(A=(root[:, None] * vt[:r]), B=(u[:, :r] * root[None, :]))


def fed_avg_product(global_ad: Adapter, deltas: list[np.ndarray], weights: np.ndarray) -> Adapter:
    prods = []
    for d in deltas:
        loc = Adapter.from_flat(global_ad.flat() + d, global_ad)
        prods.append(loc.B @ loc.A)
    mean = sum(w * p for w, p in zip(weights, prods))
    return refactor(mean, global_ad)
