import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fedlora.config import FedConfig  # noqa: E402
from fedlora.coordinator.federation import build_model, run_federation  # noqa: E402
from fedlora.model import Adapter, LoRAClassifier  # noqa: E402
from fedlora.sim.data import N_CLASSES, N_FEATURES, make_world  # noqa: E402
from fedlora.torch_backend import (  # noqa: E402
    TorchLoRAClassifier, from_peft, load_peft_adapter, save_peft_adapter, to_peft)


def _setup(rank=3):
    w = make_world(3, seed=0)
    m = build_model(w, FedConfig(rank=rank, alpha=2.0 * rank))
    rng = np.random.default_rng(0)
    ad = Adapter.init(N_FEATURES, N_CLASSES, rank, rng)
    ad.B = rng.normal(0, 0.2, ad.B.shape)
    frozen = Adapter.init(N_FEATURES, N_CLASSES, rank, rng)
    frozen.B = rng.normal(0, 0.2, frozen.B.shape)
    s = w.sites[0].train
    return m, ad, frozen, s.x[:32], s.y[:32]


def test_autograd_gradients_match_numpy_gradients():
    m, ad, frozen, x, y = _setup()
    t = TorchLoRAClassifier(m.W, m.b, alpha=m.scale * 3, rank=3, quantize=False)
    for args in ((), (frozen,)):
        dA, dB = m.grads(x, y, ad, *args)
        tA, tB = t.grads(x, y, ad, *args)
        assert np.allclose(dA, tA, atol=1e-10) and np.allclose(dB, tB, atol=1e-10)


def test_torch_backend_is_a_drop_in_for_federation(monkeypatch):
    import fedlora.coordinator.federation as fed
    cfg = FedConfig(dp=False, rounds=3, personalize=False, rank=2, alpha=4.0, lr=0.5, local_epochs=1)
    ref = run_federation(cfg)
    monkeypatch.setattr(fed, "LoRAClassifier", TorchLoRAClassifier)
    got = run_federation(cfg)
    assert isinstance(got.model, TorchLoRAClassifier)
    assert np.allclose(ref.global_adapter.flat(), got.global_adapter.flat(), atol=1e-8)


def test_peft_layer_computes_the_same_logits():
    pytest.importorskip("peft")
    m, ad, _, x, _ = _setup()
    peft_net = to_peft(m, ad)
    ours = m.logits(x, ad)
    theirs = peft_net(torch.as_tensor(x, dtype=torch.float64)).detach().numpy()
    assert np.allclose(ours, theirs, atol=1e-9)
    back = from_peft(peft_net)
    assert np.allclose(back.A, ad.A) and np.allclose(back.B, ad.B)


def test_adapter_roundtrips_through_peft_checkpoint_files(tmp_path):
    pytest.importorskip("peft")
    m, ad, _, x, _ = _setup()
    save_peft_adapter(m, ad, tmp_path)
    assert (tmp_path / "adapter_config.json").exists()
    assert (tmp_path / "adapter_model.safetensors").exists()
    loaded = load_peft_adapter(m, tmp_path)
    assert np.allclose(m.logits(x, loaded), m.logits(x, ad), atol=1e-6)
