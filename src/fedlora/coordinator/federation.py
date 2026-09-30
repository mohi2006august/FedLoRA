"""Federation orchestration: rounds, selection, stragglers, SecAgg, DP (FR-01..FR-05).

Secure aggregation is the full double-masking protocol in secagg/protocol.py.
Data flow per round (SYSTEM_ARCHITECTURE.md section 3):
select K eligible sites -> local QLoRA training -> clip + DP noise -> mask ->
secure aggregation (dropouts recovered) -> global adapter update -> account eps.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fedlora.client.dp import add_noise, clip_update
from fedlora.client.dpsgd import dpsgd_delta, steps_per_round
from fedlora.client.qlora_client import local_data, local_update, train_adapter
from fedlora.compliance.audit import AuditLog, update_sha256
from fedlora.config import FedConfig, RoundConfig
from fedlora.coordinator.accountant import (
    INT_ORDERS, ORDERS, PrivacyAccountant, noise_for_budget)
from fedlora.coordinator.strategies import (
    bucketed_weight, fed_avg_factors, fed_avg_product)
from fedlora.model import Adapter, LoRAClassifier, macro_f1, pretrain_base
from fedlora.secagg.protocol import SecAggError, secure_sum
from fedlora.sim.data import N_CLASSES, N_FEATURES, Split, World, make_world


@dataclass
class FedResult:
    cfg: FedConfig
    world: World
    model: LoRAClassifier
    global_adapter: Adapter
    personal: dict[str, Adapter]
    history: list[dict]
    accountant: PrivacyAccountant  # the ledger that governs the budget
    accountant_alt: PrivacyAccountant  # the other accounting, for comparison
    audit: AuditLog
    excluded: dict[str, int] = field(default_factory=dict)


def build_model(world: World, cfg: FedConfig) -> LoRAClassifier:
    w0, b0 = pretrain_base(world.public.x, world.public.y, N_CLASSES, seed=cfg.seed)
    return LoRAClassifier(w0, b0, cfg.alpha, cfg.rank, quantize=True)


def resolve_sigma(cfg: FedConfig, world: World | None = None) -> float:
    """Explicit sigma, else the smallest noise multiplier fitting the eps budget.

    Sigma is the multiplier *before* dropouts: the noise actually present in a released sum
    is sigma * sqrt(n_survivors / cohort_target), so calibration divides by sqrt(1 - dropout).
    """
    if cfg.noise_multiplier is not None:
        return cfg.noise_multiplier
    k = min(cfg.clients_per_round, cfg.sites)
    if cfg.dp_mode == "sample":
        if world is None:
            raise ValueError("dp_mode='sample' needs the world to size the batches")
        n = int(np.percentile([len(local_data(s)) for s in world.sites], 25))
        participations = int(np.ceil(1.25 * cfg.rounds * k / cfg.sites))
        steps = participations * steps_per_round(n, cfg.dpsgd_batch, cfg.local_epochs)
        return noise_for_budget(steps, cfg.target_epsilon, cfg.delta,
                                q=min(1.0, cfg.dpsgd_batch / n))
    if cfg.accounting == "amplified":
        needed = noise_for_budget(cfg.rounds, cfg.target_epsilon, cfg.delta, q=k / cfg.sites)
    else:
        expected = int(np.ceil(1.25 * cfg.rounds * k / cfg.sites))
        needed = noise_for_budget(expected, cfg.target_epsilon, cfg.delta)
    return needed / float(np.sqrt(max(1.0 - cfg.dropout_prob, 0.05)))


def min_release(cfg: FedConfig, k_target: int) -> int:
    """Fewest survivors for which a round is released. Fixes the lowest noise a release can
    carry (sigma*sqrt(min_release/k_target)), which the budget guard and the abort rule share."""
    return max(cfg.min_cohort, int(np.ceil(0.6 * k_target)))


def _over_budget(cfg, acct, site, sigma, floor, q_g, sample_mode) -> bool:
    if sample_mode:
        n = len(local_data(site))
        return acct.would_exceed(site.site_id, sigma, min(1.0, cfg.dpsgd_batch / n),
                                 steps_per_round(n, cfg.dpsgd_batch, cfg.local_epochs))
    return acct.would_exceed(site.site_id, floor, q_g)


def server_lr(cfg: FedConfig, r: int) -> float:
    """Decaying server step: DP noise is a random walk, so later rounds move less."""
    if not cfg.dp:
        return 1.0
    return 1.0 / (1.0 + cfg.server_lr_decay * (r - 1))


def _f1(model: LoRAClassifier, split: Split, *ads: Adapter) -> float:
    return macro_f1(split.y, model.predict(split.x, *ads), N_CLASSES)


def run_federation(cfg: FedConfig, world: World | None = None, progress=None) -> FedResult:
    rng = np.random.default_rng(cfg.seed)
    world = world or make_world(cfg.sites, cfg.seed, cfg.dirichlet_alpha)
    model = build_model(world, cfg)
    g = Adapter.init(N_FEATURES, N_CLASSES, cfg.rank, rng)
    amp = cfg.accounting == "amplified"
    acct_part = PrivacyAccountant(delta=cfg.delta, budget=cfg.target_epsilon,
                                  orders=list(INT_ORDERS if cfg.dp_mode == "sample" else ORDERS))
    acct_amp = PrivacyAccountant(delta=cfg.delta, budget=cfg.target_epsilon,
                                 orders=list(INT_ORDERS))
    acct, acct_alt = (acct_amp, acct_part) if amp else (acct_part, acct_amp)
    audit = AuditLog()
    sigma = resolve_sigma(cfg, world) if cfg.dp else 0.0
    sample_mode = cfg.dp and cfg.dp_mode == "sample"
    history, excluded = [], {}

    for r in range(1, cfg.rounds + 1):
        # Budget guard uses the lowest noise a released round can carry (n = min_cohort). Under
        # amplified accounting the sampling rate rises as sites drop out, so iterate to a fixpoint.
        eligible = list(world.sites)
        while True:
            k_t = min(cfg.clients_per_round, len(eligible))
            q_g = k_t / max(len(eligible), 1) if amp else 1.0
            floor = sigma * float(np.sqrt(min_release(cfg, k_t) / max(k_t, 1)))
            keep = [s for s in eligible if not (cfg.dp and _over_budget(
                cfg, acct, s, sigma, floor, q_g, sample_mode))]
            if len(keep) == len(eligible):
                break
            eligible = keep
        for s in world.sites:
            if s not in eligible:
                excluded.setdefault(s.site_id, r)  # budget exhausted: excluded from now on
        k_target = min(cfg.clients_per_round, len(eligible))
        if k_target < cfg.min_cohort:
            audit.record("coordinator", r, "0" * 64, event="halt:min-cohort")
            break
        q = k_target / len(eligible)
        if cfg.sampling == "poisson":
            chosen = [s for s in eligible if rng.random() < q]
        else:
            chosen = [eligible[i] for i in rng.choice(len(eligible), k_target, replace=False)]
        k = len(chosen)
        if k < cfg.min_cohort:
            audit.record("coordinator", r, "0" * 64, event="round-aborted:small-cohort")
            history.append(_row(r, model, world, g, acct, acct_alt, k, 0, aborted=True))
            continue
        rc = RoundConfig(round=r, clients_per_round=k, local_epochs=cfg.local_epochs, lr=cfg.lr,
                         clip_norm=cfg.clip_norm, noise_multiplier=sigma,
                         strategy=cfg.strategy, mu=cfg.mu)
        ids = list(range(k))
        weights = np.full(k, 1.0 / k)
        if not cfg.dp:  # no DP: bucketed (coarsened) sizes weight the average
            raw = np.array([bucketed_weight(len(s.train)) for s in chosen])
            weights = raw / raw.sum()
        raw_deltas = []
        drops = set()
        for i, site in zip(ids, chosen):
            if sample_mode:  # per-example DP-SGD: the update is already record-level private
                delta, q_site = dpsgd_delta(
                    model, local_data(site), g, epochs=rc.local_epochs, lr=rc.lr,
                    batch=cfg.dpsgd_batch, clip=cfg.dpsgd_clip, sigma=sigma, rng=rng,
                    mu=rc.mu if rc.strategy == "fedprox" else 0.0, anchor=g)
                acct_part.step(site.site_id, sigma, r, q=q_site,
                               steps=steps_per_round(len(local_data(site)), cfg.dpsgd_batch,
                                                     rc.local_epochs))
            else:
                delta = local_update(model, site, g, epochs=rc.local_epochs, lr=rc.lr,
                                     strategy=rc.strategy, mu=rc.mu, rng=rng)
            if cfg.dp and not sample_mode:
                delta, _ = clip_update(delta, rc.clip_norm)
                # per-client std sigma*C/sqrt(target) => sum of n clients has std sigma*C*sqrt(n/target)
                delta = add_noise(delta, rc.clip_norm, sigma, k_target, rng)
            audit.record(site.site_id, r, update_sha256(delta))
            raw_deltas.append(delta)
            if rng.random() < cfg.dropout_prob:
                drops.add(i)
        survivors = [i for i in ids if i not in drops]
        if len(survivors) < min_release(cfg, k_target):
            audit.record("coordinator", r, "0" * 64, event="round-aborted:dropouts")
            history.append(_row(r, model, world, g, acct, acct_alt, k, len(drops), aborted=True))
            continue
        if cfg.dp and not sample_mode:  # a sum was released: charge it
            sigma_eff = sigma * float(np.sqrt(len(survivors) / k_target))
            for site in chosen:
                acct_part.step(site.site_id, sigma_eff, r)
            for site in eligible:
                acct_amp.step(site.site_id, sigma_eff, r, q=q)
        if cfg.aggregation == "product":
            # research option: server sees individual updates (no SecAgg)
            w = weights[survivors] / weights[survivors].sum()
            g = fed_avg_product(g, [raw_deltas[i] for i in survivors], w)
        else:
            try:
                total = secure_sum({i: raw_deltas[i] * weights[i] for i in ids}, session=r,
                                   threshold=max(cfg.min_cohort, k // 2 + 1),
                                   drop_before_upload=drops)
            except SecAggError as e:
                audit.record("coordinator", r, "0" * 64, event=f"round-aborted:secagg:{e}")
                history.append(_row(r, model, world, g, acct, acct_alt, k, len(drops), aborted=True))
                continue
            g = fed_avg_factors(g, server_lr(cfg, r) * total / weights[survivors].sum())
        history.append(_row(r, model, world, g, acct, acct_alt, k, len(drops)))
        if progress:
            progress(history[-1])

    personal = {}
    if cfg.personalize:
        for s in world.sites:
            personal[s.site_id] = train_personal(model, s, g, cfg, rng)
    for sid, r in excluded.items():
        audit.record(sid, r, "0" * 64, event="excluded:budget")
    return FedResult(cfg, world, model, g, personal, history, acct, acct_alt, audit, excluded)


def train_personal(model, site, g: Adapter, cfg: FedConfig, rng) -> Adapter:
    """Personal adapter on top of the frozen global one. Never leaves the site."""
    start = Adapter.init(N_FEATURES, N_CLASSES, cfg.rank, rng)
    return train_adapter(model, local_data(site), start, epochs=cfg.personal_epochs,
                         lr=cfg.lr, frozen=(g,), rng=rng)


def _row(r, model, world, g, acct, acct_alt, k, ndrop, aborted=False) -> dict:
    hf = [_f1(model, s.heldout, g) for s in world.sites]
    eps = [acct.epsilon(s.site_id) for s in world.sites]
    return {"round": r, "participants": k, "dropped": ndrop, "aborted": aborted,
            "global_f1": _f1(model, world.test, g), "site_f1_mean": float(np.mean(hf)),
            "eps_max": max(eps), "eps_alt_max": max(acct_alt.epsilon(s.site_id) for s in world.sites),
            "eps": dict(zip([s.site_id for s in world.sites], eps))}
