"""Canary exposure (Carlini et al. 2019, simplified) for memorisation tests (FR-06).

Canaries inserted in a site's training data are compared against reference
canaries drawn from the same distribution but never trained on. Exposure =
log2(|R|+1) - log2(rank), where rank is the position of the inserted canary's
log-likelihood among the references (rank 1 = most memorised).
"""
from __future__ import annotations

import numpy as np

from fedlora.model import Adapter, LoRAClassifier, softmax
from fedlora.sim.data import N_CLASSES, N_FEATURES, Split

EXPOSURE_THRESHOLD_BITS = 10.0
N_REFERENCE = 1024


def reference_canaries(seed: int = 12345, n: int = N_REFERENCE) -> Split:
    rng = np.random.default_rng(seed)
    return Split(rng.dirichlet(np.full(N_FEATURES, 0.3), size=n), rng.integers(0, N_CLASSES, n))


def loglik(model: LoRAClassifier, split: Split, *adapters: Adapter) -> np.ndarray:
    p = softmax(model.logits(split.x, *adapters))
    return np.log(p[np.arange(len(split)), split.y] + 1e-12)


def exposure(model: LoRAClassifier, canaries: Split, *adapters: Adapter,
             reference: Split | None = None) -> np.ndarray:
    """Per-canary exposure in bits."""
    ref = loglik(model, reference or reference_canaries(), *adapters)
    mine = loglik(model, canaries, *adapters)
    ranks = 1 + (ref[None, :] > mine[:, None]).sum(1)
    return np.log2(len(ref) + 1) - np.log2(ranks)


def max_exposure(model, canaries, *adapters) -> float:
    return float(exposure(model, canaries, *adapters).max())


def release_gate(model, sites, *adapters, threshold: float = EXPOSURE_THRESHOLD_BITS) -> dict:
    per_site = {s.site_id: max_exposure(model, s.canaries, *adapters) for s in sites}
    worst = max(per_site.values())
    return {"per_site": per_site, "max_bits": worst, "threshold": threshold,
            "passed": worst < threshold}
