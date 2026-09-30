"""Federated clinical embedding model for retrieval (FR-08).

A linear projection E (k x d) is trained with a supervised InfoNCE loss on each
site's notes; E is aggregated with clipped, noised FedAvg (same DP path as the
adapters). Retrieval quality is measured as precision@1 of cosine nearest
neighbour on a shared test set.
"""
from __future__ import annotations

import numpy as np

from fedlora.client.dp import add_noise, clip_update
from fedlora.sim.data import N_FEATURES, Site, Split, World

DIM = 16
TAU = 0.5


def _pos_index(y: np.ndarray, rng) -> np.ndarray:
    """Random same-class positive for each anchor (self if none exists)."""
    pos = np.arange(len(y))
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        if len(idx) > 1:
            pos[idx] = rng.choice(idx, size=len(idx))
    return pos


def infonce(E: np.ndarray, x: np.ndarray, pos: np.ndarray, tau: float = TAU):
    """Loss and gradient wrt E. Diagonal is masked out of the softmax."""
    n = len(x)
    z = x @ E.T
    s = z @ z.T / tau
    np.fill_diagonal(s, -np.inf)
    s -= s.max(1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(1, keepdims=True)
    valid = pos != np.arange(n)
    loss = -np.log(p[np.arange(n), pos] + 1e-12)[valid].mean()
    g = p.copy()
    g[np.arange(n), pos] -= 1
    g[~valid] = 0
    g /= max(valid.sum(), 1)
    dz = (g + g.T) @ z / tau
    return float(loss), dz.T @ x


def local_train(E: np.ndarray, data: Split, *, epochs: int, lr: float, batch: int = 64,
                rng) -> np.ndarray:
    E = E.copy()
    for _ in range(epochs):
        order = rng.permutation(len(data))
        for s in range(0, len(order), batch):
            idx = order[s : s + batch]
            _, grad = infonce(E, data.x[idx], _pos_index(data.y[idx], rng))
            E -= lr * grad
    return E


def precision_at_1(E: np.ndarray, split: Split) -> float:
    z = split.x @ E.T
    z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-12
    s = z @ z.T
    np.fill_diagonal(s, -np.inf)
    return float((split.y[s.argmax(1)] == split.y).mean())


def federated_embeddings(world: World, *, rounds: int = 10, clip_norm: float = 2.0,
                         noise_multiplier: float = 0.0, epochs: int = 1, lr: float = 0.2,
                         seed: int = 0, dim: int = DIM) -> tuple[np.ndarray, list[float]]:
    rng = np.random.default_rng(seed)
    E = rng.normal(0, 1 / np.sqrt(N_FEATURES), (dim, N_FEATURES))
    curve = [precision_at_1(E, world.test)]
    k = len(world.sites)
    for _ in range(rounds):
        deltas = []
        for s in world.sites:
            d = (local_train(E, s.train, epochs=epochs, lr=lr, rng=rng) - E).ravel()
            d, _ = clip_update(d, clip_norm)
            deltas.append(add_noise(d, clip_norm, noise_multiplier, k, rng))
        E = E + np.mean(deltas, axis=0).reshape(E.shape)
        curve.append(precision_at_1(E, world.test))
    return E, curve
