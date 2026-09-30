"""Frozen NF4-quantised base plus LoRA adapter for a linear softmax classifier.

logits = (Q(W0) + (alpha/r) * B @ A) x + b0.  Only A and B are trained/shared.
Manual numpy gradients keep the simulation dependency-light.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# 16 NF4 code values (normal-float 4-bit, from the QLoRA paper).
NF4 = np.array(
    [-1.0, -0.6961928, -0.5250731, -0.3949175, -0.2844414, -0.1847734, -0.0910500, 0.0,
     0.0795803, 0.1609302, 0.2461123, 0.3379152, 0.4407098, 0.5626170, 0.7229568, 1.0]
)


def nf4_quantize(w: np.ndarray, block: int = 16) -> np.ndarray:
    """Block-wise NF4 quantise then dequantise (simulated 4-bit base)."""
    flat = w.reshape(-1)
    pad = (-len(flat)) % block
    padded = np.concatenate([flat, np.zeros(pad)]).reshape(-1, block)
    scale = np.abs(padded).max(1, keepdims=True)
    scale[scale == 0] = 1.0
    idx = np.abs(padded[..., None] / scale[..., None] - NF4).argmin(-1)
    deq = NF4[idx] * scale
    return deq.reshape(-1)[: len(flat)].reshape(w.shape)


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


@dataclass
class Adapter:
    A: np.ndarray  # (r, d_in)
    B: np.ndarray  # (d_out, r)

    @classmethod
    def init(cls, d_in: int, d_out: int, r: int, rng) -> "Adapter":
        return cls(rng.normal(0, 1 / np.sqrt(d_in), (r, d_in)), np.zeros((d_out, r)))

    def copy(self) -> "Adapter":
        return Adapter(self.A.copy(), self.B.copy())

    def flat(self) -> np.ndarray:
        return np.concatenate([self.A.ravel(), self.B.ravel()])

    @classmethod
    def from_flat(cls, v: np.ndarray, like: "Adapter") -> "Adapter":
        na = like.A.size
        return cls(v[:na].reshape(like.A.shape).copy(), v[na:].reshape(like.B.shape).copy())

    def n_params(self) -> int:
        return self.A.size + self.B.size


class LoRAClassifier:
    def __init__(self, w0: np.ndarray, b0: np.ndarray, alpha: float, rank: int, quantize=True):
        self.W = nf4_quantize(w0) if quantize else w0.copy()
        self.b = b0.copy()
        self.scale = alpha / rank

    def logits(self, x, *adapters: Adapter) -> np.ndarray:
        z = x @ self.W.T + self.b
        for ad in adapters:
            z = z + self.scale * (x @ ad.A.T) @ ad.B.T
        return z

    def loss(self, x, y, *adapters) -> float:
        p = softmax(self.logits(x, *adapters))
        return float(-np.log(p[np.arange(len(y)), y] + 1e-12).mean())

    def predict(self, x, *adapters) -> np.ndarray:
        return self.logits(x, *adapters).argmax(1)

    def grads(self, x, y, train: Adapter, *frozen: Adapter) -> tuple[np.ndarray, np.ndarray]:
        """Gradients (dA, dB) of mean CE wrt `train`; other adapters held fixed."""
        p = softmax(self.logits(x, train, *frozen))
        p[np.arange(len(y)), y] -= 1
        gW = self.scale * (p / len(y)).T @ x  # effective dL/d(BA), shape (d_out, d_in)
        return train.B.T @ gW, gW @ train.A.T


def pretrain_base(x, y, n_classes, epochs=200, lr=1.0, seed=0):
    """Fit a generic base on public data (stands in for the pretrained LLM)."""
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.01, (n_classes, x.shape[1]))
    b = np.zeros(n_classes)
    for _ in range(epochs):
        p = softmax(x @ W.T + b)
        p[np.arange(len(y)), y] -= 1
        g = p / len(y)
        W -= lr * g.T @ x
        b -= lr * g.sum(0)
    return W, b


def macro_f1(y, pred, n_classes) -> float:
    f = []
    for c in range(n_classes):
        tp = np.sum((pred == c) & (y == c))
        fp = np.sum((pred == c) & (y != c))
        fn = np.sum((pred != c) & (y == c))
        if tp + fp + fn == 0:
            continue
        f.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f)) if f else 0.0
