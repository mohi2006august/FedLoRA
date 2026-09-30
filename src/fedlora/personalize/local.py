"""Per-site personal adapters with local model selection (FR-05).

Final site model = base + g_scale * global adapter + personal adapter. The personal adapter
and its g_scale never leave the site. Because a noisy (DP) global adapter can hurt a site that
is already well served by its own data, each site picks (g_scale, epochs) on a *local*
validation split carved from its own training notes (canaries stay in training). The held-out
set used for reporting is never touched, so selection does not inflate the reported score.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fedlora.client.qlora_client import train_adapter
from fedlora.config import FedConfig
from fedlora.model import Adapter
from fedlora.sim.data import N_CLASSES, N_FEATURES, Site, Split


def scaled(g: Adapter, s: float) -> Adapter:
    """Adapter whose contribution is s times that of g (logit contributions are linear in B)."""
    return Adapter(g.A, g.B * s)


@dataclass
class PersonalModel:
    adapter: Adapter
    g_scale: float = 1.0
    epochs: int = 0

    def adapters(self, g: Adapter) -> tuple[Adapter, ...]:
        """Adapters to pass to LoRAClassifier.logits for this site's final model."""
        if self.g_scale == 0.0:
            return (self.adapter,)
        return (scaled(g, self.g_scale), self.adapter)


def _split_local(site: Site, rng, val_frac: float = 0.2) -> tuple[Split, Split, Split]:
    n = len(site.train)
    idx = rng.permutation(n)
    nv = max(10, int(val_frac * n))
    val, tr = idx[:nv], idx[nv:]
    train = Split(np.concatenate([site.train.x[tr], site.canaries.x]),
                  np.concatenate([site.train.y[tr], site.canaries.y]))
    full = Split(np.concatenate([site.train.x, site.canaries.x]),
                 np.concatenate([site.train.y, site.canaries.y]))
    return train, Split(site.train.x[val], site.train.y[val]), full


def personalize_site(model, site: Site, g: Adapter, cfg: FedConfig, rng) -> PersonalModel:
    rank = cfg.personal_rank or cfg.rank
    if not cfg.personal_select:
        start = Adapter.init(N_FEATURES, N_CLASSES, rank, rng)
        full = Split(np.concatenate([site.train.x, site.canaries.x]),
                     np.concatenate([site.train.y, site.canaries.y]))
        ad = train_adapter(model, full, start, epochs=cfg.personal_epochs, lr=cfg.lr,
                           frozen=(g,), rng=rng)
        return PersonalModel(ad, 1.0, cfg.personal_epochs)
    train, val, full = _split_local(site, rng)
    best = (np.inf, 1.0, 1)
    for s in cfg.personal_scales:
        frozen = (scaled(g, s),) if s > 0 else ()
        ad = Adapter.init(N_FEATURES, N_CLASSES, rank, rng)
        for ep in range(1, cfg.personal_epochs + 1):
            ad = train_adapter(model, train, ad, epochs=1, lr=cfg.lr, frozen=frozen, rng=rng)
            loss = model.loss(val.x, val.y, *frozen, ad)
            if loss < best[0] - 1e-9:
                best = (loss, s, ep)
    _, s, ep = best
    frozen = (scaled(g, s),) if s > 0 else ()
    ad = train_adapter(model, full, Adapter.init(N_FEATURES, N_CLASSES, rank, rng), epochs=ep,
                       lr=cfg.lr, frozen=frozen, rng=rng)
    return PersonalModel(ad, s, ep)
