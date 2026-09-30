"""Pairwise-masking secure aggregation with dropout recovery (FR-03).

Simplified Bonawitz et al. 2017: every pair (i, j) shares a seed (stands in for
a Diffie-Hellman agreement). Client i uploads  x_i + sum_{j>i} m_ij - sum_{j<i} m_ij
in fixed-point arithmetic mod 2^32, so the server learns only the sum once masks
cancel. If a client drops after masking, the survivors reveal their seeds with
the dropped client (stands in for Shamir share reconstruction) and the server
removes the orphaned masks. Omitted: self-masks / double masking, so a server
that lies about who dropped is out of scope for this simulation.
"""
from __future__ import annotations

import hashlib

import numpy as np

MOD_BITS = 32
MASK = (1 << MOD_BITS) - 1
SCALE = float(1 << 16)


def encode(x: np.ndarray) -> np.ndarray:
    return np.round(x * SCALE).astype(np.int64).astype(np.uint64) & np.uint64(MASK)


def decode(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.int64) & MASK
    v = np.where(v >= (1 << (MOD_BITS - 1)), v - (1 << MOD_BITS), v)
    return v.astype(np.float64) / SCALE


def pair_seed(session: int, i: int, j: int) -> int:
    a, b = min(i, j), max(i, j)
    h = hashlib.sha256(f"{session}:{a}:{b}".encode()).digest()
    return int.from_bytes(h[:8], "little")


def _prg(seed: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 1 << MOD_BITS, size=n, dtype=np.uint64)


def mask_update(x: np.ndarray, me: int, cohort: list[int], session: int) -> np.ndarray:
    out = encode(x)
    for j in cohort:
        if j == me:
            continue
        m = _prg(pair_seed(session, me, j), len(x))
        out = (out + m) & np.uint64(MASK) if me < j else (out - m) & np.uint64(MASK)
    return out


def aggregate(masked: dict[int, np.ndarray], cohort: list[int], session: int,
              min_cohort: int = 2) -> np.ndarray:
    """Server side: sum masked vectors of survivors, strip masks of dropped clients."""
    survivors = sorted(masked)
    if len(survivors) < min_cohort:
        raise ValueError(f"cohort {len(survivors)} below minimum {min_cohort}")
    n = len(next(iter(masked.values())))
    total = np.zeros(n, dtype=np.uint64)
    for v in masked.values():
        total = (total + v) & np.uint64(MASK)
    dropped = [c for c in cohort if c not in masked]
    for d in dropped:
        for s in survivors:
            m = _prg(pair_seed(session, s, d), n)
            # survivor s added +m if s<d else -m; remove that contribution
            total = (total - m) & np.uint64(MASK) if s < d else (total + m) & np.uint64(MASK)
    return decode(total)
