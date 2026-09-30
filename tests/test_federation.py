import numpy as np

from fedlora.config import FedConfig
from fedlora.coordinator.federation import run_federation
from fedlora.coordinator.strategies import fed_avg_product, refactor
from fedlora.model import Adapter


def test_federation_improves_over_first_round_without_dp():
    r = run_federation(FedConfig(dp=False, rounds=20, rank=2, alpha=4.0, local_epochs=2, lr=0.5,
                                 personalize=False))
    assert r.history[-1]["global_f1"] >= r.history[0]["global_f1"] - 0.01
    assert r.history[-1]["global_f1"] > 0.9
    assert r.audit.verify_chain()


def test_dp_respects_budget_and_records_ledger():
    cfg = FedConfig(rounds=6, sites=8, personalize=False, target_epsilon=8.0)
    r = run_federation(cfg)
    assert all(r.accountant.epsilon(s.site_id) <= cfg.target_epsilon for s in r.world.sites)
    assert r.accountant.ledger


def test_budget_exhaustion_excludes_sites_and_halts():
    cfg = FedConfig(rounds=20, sites=8, noise_multiplier=1.0, target_epsilon=4.0,
                    personalize=False, min_cohort=3)
    r = run_federation(cfg)
    assert r.excluded  # sites dropped after exhausting eps
    assert len(r.history) < 20
    assert all(r.accountant.epsilon(s.site_id) <= 4.0 for s in r.world.sites)


def test_dropouts_do_not_break_rounds():
    r = run_federation(FedConfig(dp=False, rounds=6, dropout_prob=0.4, personalize=False))
    assert any(h["dropped"] for h in r.history)
    assert np.isfinite(r.global_adapter.flat()).all()


def test_personal_adapters_stay_local_and_help_small_sites():
    r = run_federation(FedConfig(dp=False, rounds=6, personalize=True, personal_epochs=8))
    from fedlora.model import macro_f1
    from fedlora.sim.data import N_CLASSES
    gains = []
    for s in r.world.sites:
        g = macro_f1(s.heldout.y, r.model.predict(s.heldout.x, r.global_adapter), N_CLASSES)
        p = macro_f1(s.heldout.y, r.model.predict(s.heldout.x, r.global_adapter,
                                                  r.personal[s.site_id]), N_CLASSES)
        gains.append(p - g)
    assert np.mean(gains) > -0.05


def test_product_aggregation_is_faithful():
    rng = np.random.default_rng(0)
    g = Adapter.init(16, 8, 2, rng)
    ds = [rng.normal(0, 0.1, g.flat().shape) for _ in range(3)]
    w = np.full(3, 1 / 3)
    out = fed_avg_product(g, ds, w)
    true = sum(wi * (lambda a: a.B @ a.A)(Adapter.from_flat(g.flat() + d, g)) for wi, d in zip(w, ds))
    err_prod = np.linalg.norm(out.B @ out.A - true)
    fac = Adapter.from_flat(g.flat() + sum(ds) / 3, g)
    err_fac = np.linalg.norm(fac.B @ fac.A - true)
    assert err_prod <= err_fac + 1e-9
    assert refactor(true, g).A.shape == g.A.shape
