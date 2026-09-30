import numpy as np

from fedlora.client.dpsgd import dpsgd_delta, per_example_grads, steps_per_round
from fedlora.config import FedConfig
from fedlora.coordinator.federation import build_model, run_federation
from fedlora.model import Adapter
from fedlora.sim.data import N_CLASSES, N_FEATURES, Split, make_world


def _setup(rank=2):
    w = make_world(3, seed=0)
    cfg = FedConfig(rank=rank, alpha=2.0 * rank)
    m = build_model(w, cfg)
    rng = np.random.default_rng(0)
    ad = Adapter.init(N_FEATURES, N_CLASSES, rank, rng)
    ad.B = rng.normal(0, 0.1, ad.B.shape)
    return w, m, ad, rng


def test_per_example_grads_average_to_batch_grad():
    w, m, ad, _ = _setup()
    s = w.sites[0].train
    g = per_example_grads(m, s.x[:16], s.y[:16], ad)
    dA, dB = m.grads(s.x[:16], s.y[:16], ad)
    assert np.allclose(g.mean(0), np.concatenate([dA.ravel(), dB.ravel()]))


def test_noiseless_unclipped_full_batch_equals_gradient_descent():
    w, m, ad, rng = _setup()
    s = w.sites[0].train
    data = Split(s.x[:32], s.y[:32])
    delta, q = dpsgd_delta(m, data, ad, epochs=1, lr=0.3, batch=32, clip=1e9, sigma=0.0, rng=rng)
    dA, dB = m.grads(data.x, data.y, ad)
    assert q == 1.0
    assert np.allclose(delta, -0.3 * np.concatenate([dA.ravel(), dB.ravel()]))


def test_noise_scale_and_clipping():
    w, m, ad, rng = _setup()
    data = Split(w.sites[0].train.x[:64], w.sites[0].train.y[:64])
    sigma, clip, lr, B = 2.0, 0.5, 1.0, 64
    # zero out the signal with a tiny clip so the delta is (almost) pure noise
    deltas = np.stack([dpsgd_delta(m, data, ad, epochs=1, lr=lr, batch=B, clip=clip, sigma=sigma,
                                   rng=np.random.default_rng(i))[0] for i in range(300)])
    expected_std = lr * sigma * clip / B
    assert abs(deltas.std(0).mean() - expected_std) / expected_std < 0.15
    assert steps_per_round(64, 64, 3) == 3


def test_sample_level_federation_stays_in_budget():
    cfg = FedConfig(dp_mode="sample", rounds=6, clients_per_round=7, personalize=False)
    r = run_federation(cfg)
    assert all(r.accountant.epsilon(s.site_id) <= cfg.target_epsilon for s in r.world.sites)
    assert r.history[-1]["global_f1"] > 0.5
    assert max(r.accountant.epsilon(s.site_id) for s in r.world.sites) > 0


def test_sample_mode_rejects_amplified():
    import pytest
    with pytest.raises(ValueError):
        FedConfig(dp_mode="sample", sampling="poisson", accounting="amplified")
