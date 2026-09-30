"""Client-level DP for LoRA updates: clip to norm C, add Gaussian noise (FR-04)."""
from __future__ import annotations

import numpy as np


def clip_update(delta: np.ndarray, clip_norm: float) -> tuple[np.ndarray, float]:
    """Scale delta so ||delta||_2 <= clip_norm. Returns (clipped, original_norm)."""
    norm = float(np.linalg.norm(delta))
    factor = min(1.0, clip_norm / (norm + 1e-12))
    return delta * factor, norm


def add_noise(delta: np.ndarray, clip_norm: float, noise_multiplier: float,
              cohort: int, rng) -> np.ndarray:
    """Distributed Gaussian noise.

    Each of `cohort` clients adds N(0, sigma^2 C^2 / cohort) so the *sum*
    (all the server ever sees under SecAgg) carries N(0, sigma^2 C^2).
    """
    if noise_multiplier <= 0:
        return delta
    std = noise_multiplier * clip_norm / np.sqrt(max(cohort, 1))
    return delta + rng.normal(0.0, std, size=delta.shape)
