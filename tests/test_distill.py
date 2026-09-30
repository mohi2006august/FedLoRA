import numpy as np
import pytest

from fedlora.config import FedConfig
from fedlora.coordinator.federation import run_federation
from fedlora.distill.distiller import agreement, distill, evaluate_student


def test_distillation_reduces_kd_loss_and_keeps_quality():
    r = run_federation(FedConfig(dp=False, rounds=15, local_epochs=2, lr=0.5, personalize=False))
    s, curve = distill(r.model, r.global_adapter, student="3B", epochs=200,
                    pool=r.world.public.x)
    assert curve[-1] < curve[0] * 0.5
    teacher_f1 = r.history[-1]["global_f1"]
    assert evaluate_student(s, r.world.test) > teacher_f1 - 0.25
    assert agreement(s, r.model, r.global_adapter, r.world.test) > 0.6
    # compact: fewer params than base + adapter
    assert s.n_params() < r.model.W.size + r.global_adapter.n_params() + r.model.b.size


def test_unknown_student_rejected():
    r = run_federation(FedConfig(dp=False, rounds=1, personalize=False))
    with pytest.raises(ValueError):
        distill(r.model, r.global_adapter, student="70B")
