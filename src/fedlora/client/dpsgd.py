"""Per-example DP-SGD for the LoRA adapter (FR-04, sample-level DP).

Each optimiser step draws a Poisson batch (rate q = batch/n), clips every example's
gradient to norm C, sums, adds N(0, sigma^2 C^2) and divides by the expected batch size.
Protects each *record* (a patient note), independent of what the coordinator learns about
participation. The released update is post-processing of the DP-SGD iterates, so the
per-site epsilon is the composition of all local steps across rounds (sampled-Gaussian RDP
with rate q; client-sampling amplification is not credited).
"""
from __future__ import annotations

import math

import numpy as np

from fedlora.model import Adapter, LoRAClassifier, softmax
from fedlora.sim.data import Split


def per_example_grads(model: LoRAClassifier, x: np.ndarray, y: np.ndarray,
                      ad: Adapter) -> np.ndarray:
    """(n, d) per-example gradients in Adapter.flat() order (A then B)."""
    p = softmax(model.logits(x, ad))
    p[np.arange(len(y)), y] -= 1  # (n, C): dL_i/dlogits
    ax = x @ ad.A.T  # (n, r)
    dB = model.scale * np.einsum("nc,nr->ncr", p, ax)
    dA = model.scale * np.einsum("nr,nd->nrd", p @ ad.B, x)
    return np.concatenate([dA.reshape(len(y), -1), dB.reshape(len(y), -1)], axis=1)


def steps_per_round(n: int, batch: int, epochs: int) -> int:
    return epochs * math.ceil(n / batch)


def dpsgd_delta(model: LoRAClassifier, data: Split, start: Adapter, *, epochs: int, lr: float,
                batch: int, clip: float, sigma: float, rng, mu: float = 0.0,
                anchor: Adapter | None = None) -> tuple[np.ndarray, float]:
    """Run DP-SGD from `start`; returns (flattened delta, sampling rate q)."""
    n = len(data)
    q = min(1.0, batch / n)
    ad = start.copy()
    a0 = anchor.flat() if anchor is not None else None
    for _ in range(steps_per_round(n, batch, epochs)):
        mask = rng.random(n) < q
        if mask.any():
            g = per_example_grads(model, data.x[mask], data.y[mask], ad)
            norms = np.linalg.norm(g, axis=1, keepdims=True)
            g = g * np.minimum(1.0, clip / (norms + 1e-12))
            total = g.sum(0)
        else:
            total = np.zeros(ad.n_params())
        total = total + rng.normal(0.0, sigma * clip, size=total.shape)
        grad = total / (q * n)  # expected batch size
        w = ad.flat()
        if mu > 0 and a0 is not None:
            grad = grad + mu * (w - a0)  # proximal term depends only on parameters
        ad = Adapter.from_flat(w - lr * grad, ad)
    return ad.flat() - start.flat(), q
