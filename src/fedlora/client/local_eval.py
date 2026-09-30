"""Local held-out evaluation and overfitting monitor (FR-06). Only aggregates leave the site."""
from __future__ import annotations

import numpy as np

from fedlora.config import SiteReport, bucket_size
from fedlora.model import Adapter, LoRAClassifier, macro_f1
from fedlora.sim.data import N_CLASSES, Site


def evaluate(model: LoRAClassifier, split, *adapters: Adapter) -> tuple[float, float]:
    loss = model.loss(split.x, split.y, *adapters)
    f1 = macro_f1(split.y, model.predict(split.x, *adapters), N_CLASSES)
    return loss, f1


def generalisation_gap(model: LoRAClassifier, site: Site, *adapters: Adapter) -> float:
    """Held-out loss minus training loss; a growing gap signals overfitting."""
    return model.loss(site.heldout.x, site.heldout.y, *adapters) - model.loss(
        site.train.x, site.train.y, *adapters)


def overfit_flag(gap: float, threshold: float = 0.35) -> bool:
    return gap > threshold


def site_report(model: LoRAClassifier, site: Site, round_no: int, epsilon: float,
                sha: str, *adapters: Adapter) -> SiteReport:
    loss, f1 = evaluate(model, site.heldout, *adapters)
    return SiteReport(site_id=site.site_id, round=round_no,
                      n_examples_bucket=bucket_size(len(site.train)),
                      heldout_loss=loss, task_f1=f1, epsilon_spent=epsilon, update_sha256=sha)
