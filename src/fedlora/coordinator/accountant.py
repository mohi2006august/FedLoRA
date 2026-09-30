"""RDP privacy accountant for the client-level Gaussian mechanism (FR-04).

Each round a site's clipped update (sensitivity C) is released with noise
std sigma*C, an RDP curve of alpha/(2 sigma^2) per round (Mironov 2017). We
compose additively and convert to (eps, delta) with the improved bound of
Balle et al. 2020 / Canonne et al. 2020. Amplification by client sampling is
deliberately NOT credited (conservative).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

ORDERS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)) + [128, 256, 512]


def gaussian_rdp(sigma: float, steps: int = 1, orders=ORDERS) -> list[float]:
    return [steps * a / (2 * sigma**2) for a in orders]


def rdp_to_eps(rdp: list[float], delta: float, orders=ORDERS) -> float:
    best = math.inf
    for a, r in zip(orders, rdp):
        if a <= 1:
            continue
        eps = r + math.log1p(-1 / a) - (math.log(delta) + math.log(a)) / (a - 1)
        best = min(best, eps)
    return max(best, 0.0)


def epsilon_after(sigma: float, rounds: int, delta: float) -> float:
    if rounds == 0:
        return 0.0
    if sigma <= 0:
        return math.inf
    return rdp_to_eps(gaussian_rdp(sigma, rounds), delta)


@dataclass
class PrivacyAccountant:
    delta: float = 1e-5
    budget: float = 8.0
    rdp: dict[str, list[float]] = field(default_factory=dict)
    rounds: dict[str, int] = field(default_factory=dict)
    ledger: list[dict] = field(default_factory=list)

    def step(self, site_id: str, sigma: float, round_no: int) -> float:
        cur = self.rdp.setdefault(site_id, [0.0] * len(ORDERS))
        inc = gaussian_rdp(sigma) if sigma > 0 else [math.inf] * len(ORDERS)
        self.rdp[site_id] = [c + i for c, i in zip(cur, inc)]
        self.rounds[site_id] = self.rounds.get(site_id, 0) + 1
        eps = self.epsilon(site_id)
        self.ledger.append({"site": site_id, "round": round_no, "epsilon": eps})
        return eps

    def epsilon(self, site_id: str) -> float:
        if site_id not in self.rdp:
            return 0.0
        return rdp_to_eps(self.rdp[site_id], self.delta)

    def would_exceed(self, site_id: str, sigma: float) -> bool:
        """True if one more participation would push the site past budget."""
        if sigma <= 0:
            return True
        cur = self.rdp.get(site_id, [0.0] * len(ORDERS))
        nxt = [c + i for c, i in zip(cur, gaussian_rdp(sigma))]
        return rdp_to_eps(nxt, self.delta) > self.budget

    def exhausted(self, site_id: str) -> bool:
        return self.epsilon(site_id) >= self.budget


def noise_for_budget(rounds: int, target_eps: float, delta: float) -> float:
    """Smallest sigma (bisection) with eps(rounds) <= target_eps."""
    lo, hi = 0.05, 200.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if epsilon_after(mid, rounds, delta) > target_eps:
            lo = mid
        else:
            hi = mid
    return hi
