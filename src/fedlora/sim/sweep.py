"""Hyper-parameter sweeps over several seeds (BRAIN.md experiments 1-3)."""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from fedlora.config import FedConfig
from fedlora.coordinator.federation import run_federation
from fedlora.sim.runner import summarise


@dataclass
class SweepRow:
    params: dict
    gap_pts: float
    sites_improved_frac: float
    global_f1: float
    eps_max: float


def sweep(grid: dict[str, list], *, sites: int = 8, seeds: int = 3, dp: bool = True,
          base: dict | None = None) -> list[SweepRow]:
    """Run every combination in `grid` (FedConfig field -> values), averaged over seeds."""
    rows = []
    keys = list(grid)
    for combo in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, combo))
        gaps, imp, f1, eps = [], [], [], []
        for seed in range(seeds):
            kw = {"clients_per_round": max(3, round(0.9 * sites)), **(base or {}), **params}
            cfg = FedConfig(sites=sites, seed=seed, dp=dp, **kw)
            s = summarise(run_federation(cfg))
            gaps.append(s["gap_to_centralised_pts"])
            imp.append(s["sites_improved"] / sites)
            f1.append(s["last_round"]["global_f1"])
            eps.append(max(s["eps"].values()))
        rows.append(SweepRow(params, float(np.mean(gaps)), float(np.mean(imp)),
                             float(np.mean(f1)), float(np.max(eps))))
    return rows
