"""JSON wire helpers: bytes <-> base64, numpy <-> base64."""
from __future__ import annotations

import base64
import json

import numpy as np


def b64e(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def b64d(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))


def arr_e(a: np.ndarray, dtype: str) -> str:
    return b64e(np.ascontiguousarray(a, dtype=dtype).tobytes())


def arr_d(s: str, dtype: str) -> np.ndarray:
    return np.frombuffer(b64d(s), dtype=dtype).copy()


def dumps(obj) -> bytes:
    return json.dumps(obj).encode("utf-8")


def loads(raw: bytes):
    return json.loads(raw.decode("utf-8"))
