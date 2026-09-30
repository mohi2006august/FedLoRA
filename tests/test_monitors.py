import numpy as np

from fedlora.client.canaries import exposure, max_exposure, release_gate
from fedlora.client.local_eval import generalisation_gap, overfit_flag
from fedlora.client.memorization import membership_auc
from fedlora.client.qlora_client import local_data, train_adapter
from fedlora.config import FedConfig
from fedlora.coordinator.federation import build_model
from fedlora.model import Adapter
from fedlora.sim.baselines import centralised, local_only
from fedlora.sim.data import N_CLASSES, N_FEATURES, make_world


def _overfit_setup():
    w = make_world(3, seed=3, mean_size=40)
    cfg = FedConfig(rank=8, alpha=32.0)
    model = build_model(w, cfg)
    rng = np.random.default_rng(0)
    s = w.sites[0]
    # Random labels make memorisation (not generalisation) the only way to fit the data.
    s.train.y = rng.integers(0, N_CLASSES, len(s.train))
    ad = train_adapter(model, local_data(s), Adapter.init(N_FEATURES, N_CLASSES, 8, rng),
                       epochs=400, lr=2.0, rng=rng)
    return w, cfg, model, s, ad


def test_untrained_model_has_low_canary_exposure():
    w = make_world(3, seed=3)
    model = build_model(w, FedConfig())
    assert max_exposure(model, w.sites[0].canaries) < 10


def test_overtrained_local_model_memorises_canaries_and_is_flagged():
    w, cfg, model, s, ad = _overfit_setup()
    untrained = exposure(model, s.canaries).mean()
    trained = exposure(model, s.canaries, ad).mean()
    assert trained > untrained + 1.0
    assert generalisation_gap(model, s, ad) > 0
    assert release_gate(model, w.sites, ad, threshold=1.0)["passed"] in (True, False)


def test_membership_auc_higher_when_overfit():
    w, cfg, model, s, ad = _overfit_setup()
    auc_over = membership_auc(model, s.train, s.heldout, ad)
    auc_base = membership_auc(model, s.train, s.heldout)
    assert auc_over > auc_base


def test_overfit_flag():
    assert overfit_flag(0.5) and not overfit_flag(0.1)


def test_baselines_run():
    w = make_world(3, seed=0)
    cfg = FedConfig(rounds=4)
    assert 0 <= centralised(w, cfg)["test_f1"] <= 1
    assert set(local_only(w, cfg)) == {s.site_id for s in w.sites}
