import numpy as np
import pytest

from fedlora.secagg.masking import aggregate, mask_update


def _run(cohort, drop, n=50, seed=0):
    rng = np.random.default_rng(seed)
    xs = {i: rng.normal(0, 0.1, n) for i in cohort}
    masked = {i: mask_update(xs[i], i, cohort, session=7) for i in cohort if i not in drop}
    return xs, masked


def test_sum_recovered_without_dropout():
    cohort = [0, 1, 2, 3, 4]
    xs, masked = _run(cohort, [])
    got = aggregate(masked, cohort, 7)
    assert np.allclose(got, sum(xs.values()), atol=1e-3)


def test_individual_upload_looks_random():
    cohort = [0, 1, 2]
    xs, masked = _run(cohort, [])
    # a single masked vector is uncorrelated with the plaintext
    a = masked[0].astype(np.float64)
    assert abs(np.corrcoef(a, xs[0])[0, 1]) < 0.4


def test_dropout_recovery():
    cohort = [0, 1, 2, 3, 4, 5]
    xs, masked = _run(cohort, drop=[1, 4])
    got = aggregate(masked, cohort, 7)
    expect = sum(xs[i] for i in cohort if i not in (1, 4))
    assert np.allclose(got, expect, atol=1e-3)


def test_min_cohort_enforced():
    cohort = [0, 1, 2]
    _, masked = _run(cohort, drop=[1, 2])
    with pytest.raises(ValueError):
        aggregate(masked, cohort, 7, min_cohort=2)
