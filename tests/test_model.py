import numpy as np

from fedlora.model import Adapter, LoRAClassifier, nf4_quantize
from fedlora.sim.data import N_CLASSES, N_FEATURES, make_world


def test_nf4_is_4bit_and_close():
    w = np.random.default_rng(0).normal(size=(8, 64))
    q = nf4_quantize(w)
    assert q.shape == w.shape
    block = q.reshape(-1, 16)[0]
    assert len(np.unique(block)) <= 16
    assert np.abs(q - w).mean() < 0.15


def test_lora_gradients_match_numeric():
    rng = np.random.default_rng(1)
    m = LoRAClassifier(rng.normal(size=(N_CLASSES, N_FEATURES)), np.zeros(N_CLASSES), 8, 4)
    ad = Adapter.init(N_FEATURES, N_CLASSES, 4, rng)
    ad.B = rng.normal(0, 0.1, ad.B.shape)
    x = rng.random((10, N_FEATURES))
    y = rng.integers(0, N_CLASSES, 10)
    dA, dB = m.grads(x, y, ad)
    eps = 1e-6
    for arr, g in ((ad.A, dA), (ad.B, dB)):
        i = (0, 1)
        arr[i] += eps
        hi = m.loss(x, y, ad)
        arr[i] -= 2 * eps
        lo = m.loss(x, y, ad)
        arr[i] += eps
        assert abs((hi - lo) / (2 * eps) - g[i]) < 1e-5


def test_world_is_non_iid_and_has_canaries():
    w = make_world(4, seed=0)
    assert len(w.sites) == 4
    assert all(len(s.canaries) > 0 for s in w.sites)
    hist = [np.bincount(s.train.y, minlength=N_CLASSES) / len(s.train) for s in w.sites]
    assert np.abs(hist[0] - hist[1]).sum() > 0.2
