"""End-to-end simulation: federate, baseline, monitor, release-gate, persist artefacts."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from fedlora.client.canaries import release_gate
from fedlora.client.local_eval import generalisation_gap, overfit_flag
from fedlora.client.memorization import membership_auc
from fedlora.compliance.report import dashboard, model_card, save_json
from fedlora.config import FedConfig
from fedlora.coordinator.federation import FedResult, run_federation
from fedlora.model import macro_f1
from fedlora.sim.baselines import centralised, local_only
from fedlora.sim.data import N_CLASSES, Split


def summarise(res: FedResult, progress=None) -> dict:
    cfg, w, m, g = res.cfg, res.world, res.model, res.global_adapter
    ep = cfg.rounds * cfg.local_epochs
    cen = centralised(w, cfg, ep)
    loc = local_only(w, cfg, ep)
    per_site_f1 = {}
    for s in w.sites:
        ads = (g, res.personal[s.site_id]) if s.site_id in res.personal else (g,)
        per_site_f1[s.site_id] = macro_f1(s.heldout.y, m.predict(s.heldout.x, *ads), N_CLASSES)
    improved = sum(per_site_f1[s] > loc[s]["heldout_f1"] for s in per_site_f1)
    gate = release_gate(m, w.sites, g)
    pooled = Split(np.concatenate([s.train.x for s in w.sites]),
                   np.concatenate([s.train.y for s in w.sites]))
    held = Split(np.concatenate([s.heldout.x for s in w.sites]),
                 np.concatenate([s.heldout.y for s in w.sites]))
    overfit = {}
    for s in w.sites:
        gap = generalisation_gap(m, s, g)
        overfit[s.site_id] = {"gap": gap, "flag": overfit_flag(gap)}
    fed_f1 = res.history[-1]["global_f1"] if res.history else 0.0
    return {
        "config": cfg.model_dump(), "sites": cfg.sites, "rounds_planned": cfg.rounds,
        "rounds_run": len(res.history), "last_round": res.history[-1] if res.history else
        {"round": 0, "participants": 0, "global_f1": 0.0},
        "eps": {s.site_id: res.accountant.epsilon(s.site_id) for s in w.sites},
        "eps_alt": {s.site_id: res.accountant_alt.epsilon(s.site_id) for s in w.sites},
        "accounting": cfg.accounting,
        "budget": cfg.target_epsilon, "delta": cfg.delta, "excluded": res.excluded,
        "centralised_f1": cen["test_f1"],
        "gap_to_centralised_pts": 100 * (cen["test_f1"] - fed_f1),
        "local_only_mean_f1": float(np.mean([v["heldout_f1"] for v in loc.values()])),
        "site_f1_final": per_site_f1, "sites_improved": int(improved),
        "canary": gate, "mia_auc": membership_auc(m, pooled, held, g), "overfit": overfit,
    }


def run_and_save(cfg: FedConfig, out: Path, progress=None) -> tuple[FedResult, dict]:
    res = run_federation(cfg, progress=progress)
    summary = summarise(res)
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "summary.json", summary)
    save_json(out / "ledger.json", {"budget": cfg.target_epsilon, "delta": cfg.delta,
                                    "entries": res.accountant.ledger,
                                    "eps": summary["eps"], "eps_alt": summary["eps_alt"],
                                    "accounting": cfg.accounting, "excluded": res.excluded})
    (out / "sites").mkdir(exist_ok=True)
    for sid, ad in res.personal.items():  # simulation only: stays in the site's own folder
        np.savez(out / "sites" / f"{sid}.npz", A=ad.A, B=ad.B)
    res.audit.export(str(out / "audit.jsonl"))
    np.savez(out / "global_adapter.npz", A=res.global_adapter.A, B=res.global_adapter.B)
    (out / "MODEL_CARD.md").write_text(model_card(summary), encoding="utf-8")
    (out / "dashboard.txt").write_text(dashboard(summary), encoding="utf-8")
    return res, summary
