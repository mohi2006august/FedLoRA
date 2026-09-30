"""Federation orchestration: rounds, selection, stragglers, SecAgg, DP (FR-01..FR-05).

Data flow per round (SYSTEM_ARCHITECTURE.md section 3):
select K eligible sites -> local QLoRA training -> clip + DP noise -> mask ->
secure aggregation (dropouts recovered) -> global adapter update -> account eps.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fedlora.client.dp import add_noise, clip_update
from fedlora.client.qlora_client import local_data, local_update, train_adapter
from fedlora.compliance.audit import AuditLog, update_sha256
from fedlora.config import FedConfig, RoundConfig
from fedlora.coordinator.accountant import PrivacyAccountant, noise_for_budget
from fedlora.coordinator.strategies import (
    bucketed_weight, fed_avg_factors, fed_avg_product)
from fedlora.model import Adapter, LoRAClassifier, macro_f1, pretrain_base
from fedlora.secagg.masking import aggregate, mask_update
from fedlora.sim.data import N_CLASSES, N_FEATURES, Split, World, make_world


@dataclass
class FedResult:
    cfg: FedConfig
    world: World
    model: LoRAClassifier
    global_adapter: Adapter
    personal: dict[str, Adapter]
    history: list[dict]
    accountant: PrivacyAccountant
    audit: AuditLog
    excluded: dict[str, int] = field(default_factory=dict)


def build_model(world: World, cfg: FedConfig) -> LoRAClassifier:
    w0, b0 = pretrain_base(world.public.x, world.public.y, N_CLASSES, seed=cfg.seed)
    return LoRAClassifier(w0, b0, cfg.alpha, cfg.rank, quantize=True)


def resolve_sigma(cfg: FedConfig) -> float:
    """Explicit sigma, else the smallest sigma fitting the eps budget for the expected
    number of participations (with 25% margin; budget exclusion enforces the hard cap)."""
    if cfg.noise_multiplier is not None:
        return cfg.noise_multiplier
    expected = int(np.ceil(1.25 * cfg.rounds * min(cfg.clients_per_round, cfg.sites) / cfg.sites))
    return noise_for_budget(expected, cfg.target_epsilon, cfg.delta)


def _f1(model: LoRAClassifier, split: Split, *ads: Adapter) -> float:
    return macro_f1(split.y, model.predict(split.x, *ads), N_CLASSES)


def run_federation(cfg: FedConfig, world: World | None = None, progress=None) -> FedResult:
    rng = np.random.default_rng(cfg.seed)
    world = world or make_world(cfg.sites, cfg.seed, cfg.dirichlet_alpha)
    model = build_model(world, cfg)
    g = Adapter.init(N_FEATURES, N_CLASSES, cfg.rank, rng)
    acct = PrivacyAccountant(delta=cfg.delta, budget=cfg.target_epsilon)
    audit = AuditLog()
    sigma = resolve_sigma(cfg) if cfg.dp else 0.0
    history, excluded = [], {}

    for r in range(1, cfg.rounds + 1):
        eligible = [s for s in world.sites
                    if not (cfg.dp and acct.would_exceed(s.site_id, sigma))]
        for s in world.sites:
            if s not in eligible:
                excluded.setdefault(s.site_id, r)  # budget exhausted: excluded from now on
        k = min(cfg.clients_per_round, len(eligible))
        if k < cfg.min_cohort:
            audit.record("coordinator", r, "0" * 64, event="halt:min-cohort")
            break
        chosen = [eligible[i] for i in rng.choice(len(eligible), k, replace=False)]
        rc = RoundConfig(round=r, clients_per_round=k, local_epochs=cfg.local_epochs, lr=cfg.lr,
                         clip_norm=cfg.clip_norm, noise_multiplier=sigma,
                         strategy=cfg.strategy, mu=cfg.mu)
        ids = list(range(k))
        weights = np.full(k, 1.0 / k)
        if not cfg.dp:  # no DP: bucketed (coarsened) sizes weight the average
            raw = np.array([bucketed_weight(len(s.train)) for s in chosen])
            weights = raw / raw.sum()
        raw_deltas, masked = [], {}
        drops = set()
        for i, site in zip(ids, chosen):
            delta = local_update(model, site, g, epochs=rc.local_epochs, lr=rc.lr,
                                 strategy=rc.strategy, mu=rc.mu, rng=rng)
            if cfg.dp:
                delta, _ = clip_update(delta, rc.clip_norm)
                delta = add_noise(delta, rc.clip_norm, sigma, k, rng)
                acct.step(site.site_id, sigma, r)  # every selected site spends budget
            audit.record(site.site_id, r, update_sha256(delta))
            raw_deltas.append(delta)
            if rng.random() < cfg.dropout_prob:
                drops.add(i)
        survivors = [i for i in ids if i not in drops]
        if len(survivors) < cfg.min_cohort:
            audit.record("coordinator", r, "0" * 64, event="round-aborted:dropouts")
            history.append(_row(r, model, world, g, acct, k, len(drops), aborted=True))
            continue
        if cfg.aggregation == "product":
            # research option: server sees individual updates (no SecAgg)
            w = weights[survivors] / weights[survivors].sum()
            g = fed_avg_product(g, [raw_deltas[i] for i in survivors], w)
        else:
            for i in survivors:
                masked[i] = mask_update(raw_deltas[i] * weights[i], i, ids, session=r)
            total = aggregate(masked, ids, session=r, min_cohort=cfg.min_cohort)
            g = fed_avg_factors(g, total / weights[survivors].sum())
        history.append(_row(r, model, world, g, acct, k, len(drops)))
        if progress:
            progress(history[-1])

    personal = {}
    if cfg.personalize:
        for s in world.sites:
            personal[s.site_id] = train_personal(model, s, g, cfg, rng)
    for sid, r in excluded.items():
        audit.record(sid, r, "0" * 64, event="excluded:budget")
    return FedResult(cfg, world, model, g, personal, history, acct, audit, excluded)


def train_personal(model, site, g: Adapter, cfg: FedConfig, rng) -> Adapter:
    """Personal adapter on top of the frozen global one. Never leaves the site."""
    start = Adapter.init(N_FEATURES, N_CLASSES, cfg.rank, rng)
    return train_adapter(model, local_data(site), start, epochs=cfg.personal_epochs,
                         lr=cfg.lr, frozen=(g,), rng=rng)


def _row(r, model, world, g, acct, k, ndrop, aborted=False) -> dict:
    hf = [_f1(model, s.heldout, g) for s in world.sites]
    eps = [acct.epsilon(s.site_id) for s in world.sites]
    return {"round": r, "participants": k, "dropped": ndrop, "aborted": aborted,
            "global_f1": _f1(model, world.test, g), "site_f1_mean": float(np.mean(hf)),
            "eps_max": max(eps), "eps": dict(zip([s.site_id for s in world.sites], eps))}
