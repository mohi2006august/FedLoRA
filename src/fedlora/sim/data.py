"""Synthetic clinical notes partitioned into non-IID hospitals (M1 simulation).

A note is a bag-of-words feature vector; the label is an ICD-style code. Each
hospital has a skewed label mix (Dirichlet) and its own vocabulary dialect.
Some notes are canaries: unique random feature patterns with random labels
used to test memorisation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

N_FEATURES = 64
N_CLASSES = 8


@dataclass
class Split:
    x: np.ndarray
    y: np.ndarray

    def __len__(self) -> int:
        return len(self.y)


@dataclass
class Site:
    site_id: str
    train: Split
    heldout: Split
    canaries: Split
    dialect: np.ndarray = field(repr=False, default=None)


@dataclass
class World:
    prototypes: np.ndarray
    sites: list[Site]
    public: Split  # small generic public set used to pretrain the base
    test: Split  # balanced global test set with no dialect


def site_name(i: int) -> str:
    """hospital-a..z, then hospital-27, hospital-28, ... (filesystem-safe ids)."""
    return f"hospital-{chr(97 + i)}" if i < 26 else f"hospital-{i + 1}"


def _notes(rng, proto, dialect, labels, length=40, dialect_strength=1.0) -> np.ndarray:
    logits = proto[labels] + dialect_strength * dialect[None, :]
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    counts = np.stack([rng.multinomial(length, pi) for pi in p]).astype(float)
    return counts / length


def make_world(
    n_sites: int = 8,
    seed: int = 0,
    dirichlet_alpha: float = 0.5,
    mean_size: int = 300,
    n_canaries: int = 10,
) -> World:
    rng = np.random.default_rng(seed)
    proto = rng.normal(0, 1.6, size=(N_CLASSES, N_FEATURES))
    zero = np.zeros(N_FEATURES)
    sites = []
    for i in range(n_sites):
        dialect = rng.normal(0, 0.8, size=N_FEATURES)
        mix = rng.dirichlet(np.full(N_CLASSES, dirichlet_alpha))
        size = int(mean_size * rng.uniform(0.3, 1.7))  # heterogeneous sizes
        ys = rng.choice(N_CLASSES, size=size, p=mix)
        x = _notes(rng, proto, dialect, ys)
        n_val = max(20, size // 5)
        yh = rng.choice(N_CLASSES, size=n_val, p=mix)
        xh = _notes(rng, proto, dialect, yh)
        cx = rng.dirichlet(np.full(N_FEATURES, 0.3), size=n_canaries)
        cy = rng.integers(0, N_CLASSES, size=n_canaries)
        sites.append(
            Site(site_name(i), Split(x, ys), Split(xh, yh), Split(cx, cy), dialect)
        )
    yp = rng.integers(0, N_CLASSES, size=200)
    public = Split(_notes(rng, proto + rng.normal(0, 1.2, proto.shape), zero, yp), yp)
    yt = rng.integers(0, N_CLASSES, size=1000)
    test = Split(_notes(rng, proto, zero, yt), yt)
    return World(proto, sites, public, test)
