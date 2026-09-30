"""Data models shared across components (TECHNICAL.md section 3)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


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
    rounds: int = 10
    clients_per_round: int = 8
    min_cohort: int = 3  # minimum cohort for SecAgg/DP (gradient-inversion defence)
    local_epochs: int = 3
    lr: float = 1.0
    rank: int = 1
    alpha: float = 2.0
    clip_norm: float = 1.0
    noise_multiplier: float | None = Field(
        None, description="sigma; None = calibrate so expected participations fit target_epsilon")
    strategy: Literal["fedavg", "fedprox"] = "fedprox"
    mu: float = 0.01
    aggregation: Literal["factors", "product"] = "factors"
    target_epsilon: float = 8.0
    delta: float = 1e-5
    server_lr_decay: float = 0.5  # server step = 1 / (1 + decay * (round - 1))
    dropout_prob: float = 0.1
    seed: int = 0
    dp: bool = True
    personalize: bool = True
    personal_epochs: int = 20
    personal_rank: int | None = 2  # None = same as the global rank
    personal_select: bool = True  # pick (g_scale, epochs) on a local validation split
    personal_scales: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    dirichlet_alpha: float = Field(0.5, description="non-IID label skew; lower = more skew")
    dp_mode: Literal["client", "sample"] = Field(
        "client", description="client: clip+noise the whole update (user-level). "
        "sample: per-example DP-SGD (record-level), protects each note.")
    dpsgd_batch: int = 64
    dpsgd_clip: float = 1.0  # per-example gradient clip
    sampling: Literal["fixed", "poisson"] = "fixed"
    accounting: Literal["participation", "amplified"] = Field(
        "participation",
        description="participation: every participant pays full sigma (valid against the "
        "coordinator). amplified: credits Poisson sampling (valid only against an adversary "
        "that cannot see who was sampled).")

    @model_validator(mode="after")
    def _amplified_needs_poisson(self):
        if self.accounting == "amplified" and self.sampling != "poisson":
            raise ValueError("accounting='amplified' requires sampling='poisson'")
        if self.dp_mode == "sample" and self.accounting == "amplified":
            raise ValueError("dp_mode='sample' does not credit client-sampling amplification")
        return self


def bucket_size(n: int) -> str:
    """Report only bucketed dataset sizes (CLAUDE.md gotcha)."""
    for hi in (100, 500, 1000, 5000):
        if n < hi:
            return f"<{hi}"
    return ">=5000"
