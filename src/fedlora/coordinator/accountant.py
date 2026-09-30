"""RDP privacy accountant for the client-level (sampled) Gaussian mechanism (FR-04).

Each round a site's clipped update (sensitivity C) is released inside a sum with
Gaussian noise of std sigma*C. Un-sampled, the RDP curve is alpha/(2 sigma^2)
(Mironov 2017). When sites are Poisson-sampled with rate q the curve is the
sampled-Gaussian one for integer alpha (Mironov, Talwar, Zhang 2019):

    eps_alpha = log( sum_k C(alpha,k) (1-q)^(alpha-k) q^k exp((k^2-k)/(2 sigma^2)) ) / (alpha-1)

RDP composes additively and converts to (eps, delta) with the bound of
Balle et al. 2020 / Canonne et al. 2020.

Threat-model caveat: amplification by sampling only holds against an adversary
who does NOT learn which sites were sampled. The coordinator chooses the cohort,
so against the coordinator only the q = 1 ("participation") accounting is valid.
Both ledgers are kept; see federation.py.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

ORDERS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)) + [128, 256, 512]
INT_ORDERS = list(range(2, 64)) + [128, 256, 512]  # sampled Gaussian needs integer orders


def _log_binom(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _logsumexp(xs: list[float]) -> float:
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


@lru_cache(maxsize=None)
def sampled_gaussian_rdp_at(alpha: int, q: float, sigma: float) -> float:
    if q <= 0:
        return 0.0
    if q >= 1:
        return alpha / (2 * sigma**2)
    terms = [
        _log_binom(alpha, k) + (alpha - k) * math.log1p(-q) + k * math.log(q)
        + (k * k - k) / (2 * sigma**2)
        for k in range(alpha + 1)
    ]
    return _logsumexp(terms) / (alpha - 1)


def gaussian_rdp(sigma: float, steps: int = 1, orders=ORDERS, q: float = 1.0) -> list[float]:
    """RDP of `steps` compositions of the (Poisson-sampled, rate q) Gaussian mechanism."""
    if q >= 1.0:
        return [steps * a / (2 * sigma**2) for a in orders]
    if any(a != int(a) for a in orders):
        raise ValueError("sampled Gaussian RDP needs integer orders (use INT_ORDERS)")
    return [steps * sampled_gaussian_rdp_at(int(a), q, sigma) for a in orders]


def rdp_to_eps(rdp: list[float], delta: float, orders=ORDERS) -> float:
    best = math.inf
    for a, r in zip(orders, rdp):
        if a <= 1:
            continue
        eps = r + math.log1p(-1 / a) - (math.log(delta) + math.log(a)) / (a - 1)
        best = min(best, eps)
    return max(best, 0.0)


def epsilon_after(sigma: float, rounds: int, delta: float, q: float = 1.0) -> float:
    if rounds == 0:
        return 0.0
    if sigma <= 0:
        return math.inf
    orders = ORDERS if q >= 1.0 else INT_ORDERS
    return rdp_to_eps(gaussian_rdp(sigma, rounds, orders, q), delta, orders)


@dataclass
class PrivacyAccountant:
    delta: float = 1e-5
    budget: float = 8.0
    orders: list = field(default_factory=lambda: list(ORDERS))
    rdp: dict[str, list[float]] = field(default_factory=dict)
    rounds: dict[str, int] = field(default_factory=dict)
    ledger: list[dict] = field(default_factory=list)

    def _inc(self, sigma: float, q: float, steps: int = 1) -> list[float]:
        if sigma <= 0:
            return [math.inf] * len(self.orders)
        return gaussian_rdp(sigma, steps, self.orders, q)

    def step(self, site_id: str, sigma: float, round_no: int, q: float = 1.0,
             steps: int = 1) -> float:
        """Charge `steps` compositions of the (sampled) Gaussian mechanism."""
        cur = self.rdp.setdefault(site_id, [0.0] * len(self.orders))
        self.rdp[site_id] = [c + i for c, i in zip(cur, self._inc(sigma, q, steps))]
        self.rounds[site_id] = self.rounds.get(site_id, 0) + 1
        eps = self.epsilon(site_id)
        self.ledger.append({"site": site_id, "round": round_no, "epsilon": eps})
        return eps

    def epsilon(self, site_id: str) -> float:
        if site_id not in self.rdp:
            return 0.0
        return rdp_to_eps(self.rdp[site_id], self.delta, self.orders)

    def would_exceed(self, site_id: str, sigma: float, q: float = 1.0, steps: int = 1) -> bool:
        """True if one more charge would push the site past budget."""
        if sigma <= 0:
            return True
        cur = self.rdp.get(site_id, [0.0] * len(self.orders))
        nxt = [c + i for c, i in zip(cur, self._inc(sigma, q, steps))]
        return rdp_to_eps(nxt, self.delta, self.orders) > self.budget

    def exhausted(self, site_id: str) -> bool:
        return self.epsilon(site_id) >= self.budget


def noise_for_budget(rounds: int, target_eps: float, delta: float, q: float = 1.0) -> float:
    """Smallest sigma (bisection) with eps(rounds, q) <= target_eps.

    `rounds` counts compositions of the mechanism (rounds for client-level DP, optimiser steps
    for DP-SGD).
    """
    lo, hi = 0.05, 200.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if epsilon_after(mid, rounds, delta, q) > target_eps:
            lo = mid
        else:
            hi = mid
    return hi
