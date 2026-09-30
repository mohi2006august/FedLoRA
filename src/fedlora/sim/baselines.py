"""Reference baselines: centralised (upper bound) and local-only training."""
from __future__ import annotations

import numpy as np

from fedlora.client.qlora_client import local_data, train_adapter
from fedlora.config import FedConfig
from fedlora.coordinator.federation import build_model
from fedlora.model import Adapter, macro_f1
from fedlora.sim.data import N_CLASSES, N_FEATURES, Split, World


def _init(cfg, rng) -> Adapter:
    return Adapter.init(N_FEATURES, N_CLASSES, cfg.rank, rng)


def centralised(world: World, cfg: FedConfig, epochs: int | None = None) -> dict:
    """Hypothetical pooled-data model (never possible in practice)."""
    rng = np.random.default_rng(cfg.seed + 1)
    model = build_model(world, cfg)
    pooled = [local_data(s) for s in world.sites]
    data = Split(np.concatenate([p.x for p in pooled]), np.concatenate([p.y for p in pooled]))
    ad = train_adapter(model, data, _init(cfg, rng), epochs=epochs or cfg.rounds * cfg.local_epochs,
                       lr=cfg.lr, rng=rng)
    return {"adapter": ad, "test_f1": macro_f1(world.test.y, model.predict(world.test.x, ad), N_CLASSES)}


def local_only(world: World, cfg: FedConfig, epochs: int | None = None) -> dict:
    """Each site trains alone on its own notes. Returns per-site held-out F1."""
    rng = np.random.default_rng(cfg.seed + 2)
    model = build_model(world, cfg)
    out = {}
    for s in world.sites:
        ad = train_adapter(model, local_data(s), _init(cfg, rng),
                           epochs=epochs or cfg.rounds * cfg.local_epochs, lr=cfg.lr, rng=rng)
        out[s.site_id] = {
            "heldout_f1": macro_f1(s.heldout.y, model.predict(s.heldout.x, ad), N_CLASSES),
            "test_f1": macro_f1(world.test.y, model.predict(world.test.x, ad), N_CLASSES),
        }
    return out
