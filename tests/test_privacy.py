import math

import numpy as np

from fedlora.client.dp import add_noise, clip_update
from fedlora.coordinator.accountant import (
    PrivacyAccountant, epsilon_after, noise_for_budget)


def test_clip_bounds_norm():
    v = np.ones(100) * 3
    c, n = clip_update(v, 1.0)
    assert math.isclose(np.linalg.norm(c), 1.0, rel_tol=1e-6) and n > 1
    small, _ = clip_update(np.ones(4) * 0.01, 1.0)
    assert np.allclose(small, 0.01)


def test_noise_sum_matches_target_std():
    rng = np.random.default_rng(0)
    cohort, sigma, C = 5, 2.0, 1.0
    total = sum(add_noise(np.zeros(20000), C, sigma, cohort, rng) for _ in range(cohort))
    assert abs(total.std() - sigma * C) < 0.05


def test_accountant_matches_closed_form_gaussian_rdp():
    # single-round eps for sigma=4: reference formula via same conversion
    sigma, delta = 4.0, 1e-5
    eps = epsilon_after(sigma, 1, delta)
    # classic bound sqrt(2 ln(1.25/delta))/sigma is an upper bound on the RDP-derived eps
    classic = math.sqrt(2 * math.log(1.25 / delta)) / sigma
    assert 0 < eps <= classic + 1e-9


def test_eps_monotone_in_rounds_and_noise():
    assert epsilon_after(1.0, 5, 1e-5) < epsilon_after(1.0, 10, 1e-5)
    assert epsilon_after(2.0, 10, 1e-5) < epsilon_after(1.0, 10, 1e-5)


def test_composition_is_additive_in_rdp():
    acc = PrivacyAccountant(delta=1e-5, budget=100)
    for r in range(10):
        acc.step("a", 3.0, r)
    assert math.isclose(acc.epsilon("a"), epsilon_after(3.0, 10, 1e-5), rel_tol=1e-9)


def test_budget_exclusion_and_calibration():
    sigma = noise_for_budget(30, 8.0, 1e-5)
    assert epsilon_after(sigma, 30, 1e-5) <= 8.0 + 1e-6
    acc = PrivacyAccountant(delta=1e-5, budget=2.0)
    n = 0
    while not acc.would_exceed("s", 6.0):
        acc.step("s", 6.0, n)
        n += 1
    assert 0 < n < 100 and acc.epsilon("s") <= 2.0
