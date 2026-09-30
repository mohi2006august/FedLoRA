"""Site-local QLoRA training with optional FedProx proximal term (FR-02, FR-05)."""
from __future__ import annotations

import numpy as np

from fedlora.model import Adapter, LoRAClassifier
from fedlora.sim.data import Site, Split


def local_data(site: Site) -> Split:
    """Train notes plus the site's canaries (inserted to test memorisation)."""
    return Split(np.concatenate([site.train.x, site.canaries.x]),
                 np.concatenate([site.train.y, site.canaries.y]))


def train_adapter(model: LoRAClassifier, data: Split, start: Adapter, *, epochs: int, lr: float,
                  mu: float = 0.0, anchor: Adapter | None = None, frozen: tuple[Adapter, ...] = (),
                  batch: int = 64, rng=None) -> Adapter:
    """Minibatch SGD on (A, B). `anchor` (global adapter) enables FedProx."""
    rng = rng or np.random.default_rng(0)
    ad = start.copy()
    n = len(data)
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s : s + batch]
            dA, dB = model.grads(data.x[idx], data.y[idx], ad, *frozen)
            if mu > 0 and anchor is not None:
                dA = dA + mu * (ad.A - anchor.A)
                dB = dB + mu * (ad.B - anchor.B)
            ad.A -= lr * dA
            ad.B -= lr * dB
    return ad


def local_update(model: LoRAClassifier, site: Site, global_ad: Adapter, *, epochs: int, lr: float,
                 strategy: str, mu: float, rng) -> np.ndarray:
    """Returns the raw update delta = w_local - w_global (flattened). Never raw text."""
    prox = mu if strategy == "fedprox" else 0.0
    new = train_adapter(model, local_data(site), global_ad, epochs=epochs, lr=lr, mu=prox,
                        anchor=global_ad, rng=rng)
    return new.flat() - global_ad.flat()
