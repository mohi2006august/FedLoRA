import numpy as np

from fedlora.personalize.embeddings import _pos_index, federated_embeddings, infonce
from fedlora.sim.data import make_world


def test_infonce_gradient_matches_numeric():
    rng = np.random.default_rng(0)
    E = rng.normal(0, 0.3, (4, 12))
    x = rng.random((10, 12))
    y = rng.integers(0, 3, 10)
    pos = _pos_index(y, rng)
    _, g = infonce(E, x, pos)
    eps = 1e-6
    for i in [(0, 0), (2, 5)]:
        E[i] += eps
        hi = infonce(E, x, pos)[0]
        E[i] -= 2 * eps
        lo = infonce(E, x, pos)[0]
        E[i] += eps
        assert abs((hi - lo) / (2 * eps) - g[i]) < 1e-5


def test_federated_embeddings_improve_retrieval():
    w = make_world(4, seed=0)
    _, curve = federated_embeddings(w, rounds=8, dim=2, lr=3.0, epochs=3, clip_norm=50.0)
    assert curve[-1] > curve[0] + 0.1
