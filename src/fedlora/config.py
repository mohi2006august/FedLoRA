"""Data models shared across components (TECHNICAL.md section 3)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RoundConfig(BaseModel):
    round: int
    clients_per_round: int
    local_epochs: int = 2
    lr: float = 0.5
    clip_norm: float = 1.0
    noise_multiplier: float = 1.0
    strategy: Literal["fedavg", "fedprox"] = "fedprox"
    mu: float = 0.01  # FedProx proximal term


class SiteReport(BaseModel):
    site_id: str
    round: int
    n_examples_bucket: str  # coarse bucket, never the exact count
    heldout_loss: float
    task_f1: float
    epsilon_spent: float
    update_sha256: str


class FedConfig(BaseModel):
    """Whole-federation configuration (fed.yaml)."""

    sites: int = 8
    rounds: int = 30
    clients_per_round: int = 6
    min_cohort: int = 3  # minimum cohort for SecAgg/DP (gradient-inversion defence)
    local_epochs: int = 2
    lr: float = 0.5
    rank: int = 4
    alpha: float = 8.0
    clip_norm: float = 1.0
    noise_multiplier: float = 0.8
    strategy: Literal["fedavg", "fedprox"] = "fedprox"
    mu: float = 0.01
    aggregation: Literal["factors", "product"] = "factors"
    target_epsilon: float = 8.0
    delta: float = 1e-5
    dropout_prob: float = 0.1
    seed: int = 0
    dp: bool = True
    personalize: bool = True
    personal_epochs: int = 5
    dirichlet_alpha: float = Field(0.5, description="non-IID label skew; lower = more skew")


def bucket_size(n: int) -> str:
    """Report only bucketed dataset sizes (CLAUDE.md gotcha)."""
    for hi in (100, 500, 1000, 5000):
        if n < hi:
            return f"<{hi}"
    return ">=5000"
