"""Global distillation into a compact student using synthetic prompts (FR-07).

Teacher = base + global adapter, optionally ensembled with each site's personal
adapter. Personal adapters never leave their site: each site labels the shared
*synthetic* prompts locally and only those soft labels are pooled. No real
notes are used or shared. Student = low-rank linear model U·V (no base).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fedlora.model import Adapter, LoRAClassifier, macro_f1, softmax
from fedlora.sim.data import N_CLASSES, N_FEATURES, Split

STUDENT_RANKS = {"1B": 4, "3B": 6, "7B": 8}  # simulation analogue of model sizes


@dataclass
class Student:
    U: np.ndarray  # (C, r)
    V: np.ndarray  # (r, d)
    b: np.ndarray

    def logits(self, x):
        return x @ self.V.T @ self.U.T + self.b

    def predict(self, x):
        return self.logits(x).argmax(1)

    def n_params(self) -> int:
        return self.U.size + self.V.size + self.b.size


def synthetic_prompts(n: int, seed: int = 0, pool: np.ndarray | None = None) -> np.ndarray:
    """Synthetic bag-of-words prompts; carries no patient data.

    With a public `pool`, prompts are random mixups of public notes (so they sit on the
    real input manifold); otherwise sparse Dirichlet samples.
    """
    rng = np.random.default_rng(seed)
    if pool is None:
        return rng.dirichlet(np.full(N_FEATURES, 0.3), size=n)
    a, b = rng.integers(0, len(pool), n), rng.integers(0, len(pool), n)
    lam = rng.beta(0.4, 0.4, size=(n, 1))
    mix = lam * pool[a] + (1 - lam) * pool[b]
    return mix / mix.sum(1, keepdims=True)


def teacher_probs(model: LoRAClassifier, x, g: Adapter, personal: dict | None,
                  T: float) -> np.ndarray:
    ps = [softmax(model.logits(x, g) / T)]
    for pm in (personal or {}).values():  # PersonalModel: labels computed at the owning site
        ps.append(softmax(model.logits(x, *pm.adapters(g)) / T))
    return np.mean(ps, axis=0)


def kd_loss(student_logits, teacher_p, T: float) -> float:
    q = softmax(student_logits / T)
    return float((teacher_p * (np.log(teacher_p + 1e-12) - np.log(q + 1e-12))).sum(1).mean())


def distill(model: LoRAClassifier, g: Adapter, personal: dict | None = None, *,
            student: str = "3B", n_prompts: int = 4000, T: float = 2.0, epochs: int = 300,
            lr: float = 0.5, seed: int = 0,
            pool: np.ndarray | None = None) -> tuple[Student, list[float]]:
    rank = STUDENT_RANKS.get(student, None)
    if rank is None:
        raise ValueError(f"unknown student {student!r}; choose from {sorted(STUDENT_RANKS)}")
    rng = np.random.default_rng(seed)
    x = synthetic_prompts(n_prompts, seed, pool)
    tp = teacher_probs(model, x, g, personal, T)
    s = Student(rng.normal(0, 0.3, (N_CLASSES, rank)), rng.normal(0, 0.3, (rank, N_FEATURES)),
                np.zeros(N_CLASSES))
    curve = []
    for _ in range(epochs):
        q = softmax(s.logits(x) / T)
        gz = (q - tp) / (len(x) * T) * T * T  # dKL/dlogits scaled by T^2
        h = x @ s.V.T
        dU, dV, db = gz.T @ h, (gz @ s.U).T @ x, gz.sum(0)
        s.U -= lr * dU * 10
        s.V -= lr * dV * 10
        s.b -= lr * db * 10
        curve.append(kd_loss(s.logits(x), tp, T))
    return s, curve


def evaluate_student(s: Student, split: Split) -> float:
    return macro_f1(split.y, s.predict(split.x), N_CLASSES)


def agreement(s: Student, model: LoRAClassifier, g: Adapter, split: Split) -> float:
    """Fraction of real held-out notes on which student and teacher agree."""
    return float((s.predict(split.x) == model.predict(split.x, g)).mean())
