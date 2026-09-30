"""Membership-inference test: loss-threshold attack AUC (FR-06). 0.5 = no leakage."""
from __future__ import annotations

import numpy as np

from fedlora.model import Adapter, LoRAClassifier, softmax
from fedlora.sim.data import Split


def per_example_loss(model: LoRAClassifier, split: Split, *adapters: Adapter) -> np.ndarray:
    p = softmax(model.logits(split.x, *adapters))
    return -np.log(p[np.arange(len(split)), split.y] + 1e-12)


def membership_auc(model: LoRAClassifier, members: Split, non_members: Split,
                   *adapters: Adapter) -> float:
    """AUC of 'lower loss => member' (Mann-Whitney U formulation)."""
    lm = per_example_loss(model, members, *adapters)
    ln = per_example_loss(model, non_members, *adapters)
    greater = (ln[None, :] > lm[:, None]).mean()
    ties = (ln[None, :] == lm[:, None]).mean()
    return float(greater + 0.5 * ties)
