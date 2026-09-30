"""Fixed-point encoding used by secure aggregation (see protocol.py for the protocol).

Real-valued updates are scaled by 2^16 and embedded in Z_{2^32} so that masks cancel exactly.
"""
from __future__ import annotations

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
