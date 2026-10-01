"""Load real note datasets (e.g. a locally licensed MIMIC-style export) as simulated hospitals.

Expected CSV: a text column and an integer label column (0..N_CLASSES-1, coded consistently
across all sites, e.g. an ICD chapter index), optionally a site column. Text is turned into
a hashed bag-of-words (N_FEATURES buckets, L1-normalised), the same representation the
simulator uses. No dataset is bundled or downloaded: credentialed corpora such as MIMIC
must be obtained by the user under their own data-use agreement, and raw text stays on the
machine that loads it.
"""
from __future__ import annotations

import csv
import re
import zlib
from pathlib import Path

import numpy as np

from fedlora.sim.data import N_CLASSES, N_FEATURES, Site, Split, World, site_name

_TOKEN = re.compile(r"[a-z0-9]+")


def vectorize(text: str) -> np.ndarray:
    v = np.zeros(N_FEATURES)
    for tok in _TOKEN.findall(text.lower()):
        v[zlib.crc32(tok.encode()) % N_FEATURES] += 1.0
    n = v.sum()
    return v / n if n else v


def read_csv(path: Path, text_col: str = "text", label_col: str = "label",
             site_col: str | None = None) -> tuple[np.ndarray, np.ndarray, list[str] | None]:
    xs, ys, sites = [], [], []
    with open(path, newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            y = int(row[label_col])
            if not 0 <= y < N_CLASSES:
                raise ValueError(f"row {i}: label {y} outside 0..{N_CLASSES - 1}")
            xs.append(vectorize(row[text_col]))
            ys.append(y)
            if site_col:
                sites.append(row[site_col])
    if not xs:
        raise ValueError(f"{path} has no rows")
    return np.stack(xs), np.array(ys), (sites if site_col else None)


def _canaries(rng, n: int) -> Split:
    return Split(rng.dirichlet(np.full(N_FEATURES, 0.3), size=n), rng.integers(0, N_CLASSES, n))


def world_from_arrays(x: np.ndarray, y: np.ndarray, n_sites: int, *, seed: int = 0,
                      dirichlet_alpha: float = 0.5, site_labels: list[str] | None = None,
                      public_frac: float = 0.1, test_frac: float = 0.15,
                      heldout_frac: float = 0.2, n_canaries: int = 10) -> World:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_pub, n_test = int(public_frac * len(y)), int(test_frac * len(y))
    pub, test, rest = idx[:n_pub], idx[n_pub:n_pub + n_test], idx[n_pub + n_test:]
    if site_labels is not None:  # pre-partitioned: one hospital per distinct site value
        names = sorted(set(site_labels))
        arr = np.array(site_labels)
        parts = [np.array([i for i in rest if arr[i] == nm]) for nm in names]
        site_ids = names
    else:
        site_ids = [site_name(i) for i in range(n_sites)]
        props = rng.dirichlet(np.full(n_sites, dirichlet_alpha), size=N_CLASSES)  # class -> site mix
        owner = np.empty(len(rest), dtype=int)
        for k, i in enumerate(rest):
            owner[k] = rng.choice(n_sites, p=props[y[i]])
        parts = [rest[owner == s] for s in range(n_sites)]
    sites = []
    for sid, part in zip(site_ids, parts):
        if len(part) < 10:
            raise ValueError(f"site {sid} has only {len(part)} notes; use fewer sites")
        part = rng.permutation(part)
        nh = max(1, int(heldout_frac * len(part)))
        sites.append(Site(sid, Split(x[part[nh:]], y[part[nh:]]), Split(x[part[:nh]], y[part[:nh]]),
                          _canaries(rng, n_canaries)))
    return World(np.zeros((N_CLASSES, N_FEATURES)), sites, Split(x[pub], y[pub]),
                 Split(x[test], y[test]))


def world_from_csv(path: Path, n_sites: int = 8, *, seed: int = 0, dirichlet_alpha: float = 0.5,
                   site_col: str | None = None, **cols) -> World:
    x, y, sites = read_csv(path, site_col=site_col, **cols)
    return world_from_arrays(x, y, n_sites, seed=seed, dirichlet_alpha=dirichlet_alpha,
                             site_labels=sites)


def build_world(cfg) -> World:
    """World for a FedConfig: the synthetic simulator or a local CSV split into hospitals."""
    if cfg.dataset.startswith("csv:"):
        return world_from_csv(Path(cfg.dataset[4:]), cfg.sites, seed=cfg.seed,
                              dirichlet_alpha=cfg.dirichlet_alpha)
    from fedlora.sim.data import make_world
    return make_world(cfg.sites, cfg.seed, cfg.dirichlet_alpha)


def csv_site_data(site_id: str, path: Path, public: Path | None = None):
    """Networked-client data factory: this site's own notes (and optional public notes)."""
    def factory(cfg):
        x, y, _ = read_csv(path)
        rng = np.random.default_rng([cfg.seed, zlib.crc32(site_id.encode())])
        order = rng.permutation(len(y))
        nh = max(1, int(0.2 * len(y)))
        h, t = order[:nh], order[nh:]
        site = Site(site_id, Split(x[t], y[t]), Split(x[h], y[h]), _canaries(rng, 10))
        if public is not None:
            px, py, _ = read_csv(public)
            pub = Split(px, py)
        else:  # no public corpus: the base stays near-zero and LoRA learns from scratch
            pub = Split(np.zeros((1, N_FEATURES)), np.zeros(1, dtype=int))
        return site, pub
    return factory
