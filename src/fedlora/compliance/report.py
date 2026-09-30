"""Federation dashboard, privacy ledger and model card (DESIGN.md screens)."""
from __future__ import annotations

import json
from pathlib import Path

BAR = 6


def _bar(eps: float, budget: float) -> str:
    filled = min(BAR, int(round(BAR * eps / budget))) if budget > 0 else BAR
    return "▇" * filled + "░" * (BAR - filled)


def privacy_ledger(eps: dict[str, float], budget: float, delta: float) -> str:
    """Per-site epsilon vs budget; amber warning marker at >= 70% (never colour alone)."""
    lines = [f"Privacy ε (δ={delta:g})   budget {budget:.1f}"]
    for site, e in sorted(eps.items()):
        warn = " ⚠ nearly exhausted" if e >= 0.7 * budget else ""
        lines.append(f"  {site:<12} ε {e:5.2f}  {_bar(e, budget)}{warn}")
    return "\n".join(lines)


def dashboard(summary: dict) -> str:
    h = summary["last_round"]
    lines = [
        f"Round {h['round']}/{summary['rounds_planned']}   sites {h['participants']}/"
        f"{summary['sites']} participating   global F1 {h['global_f1']:.3f} "
        f"(local-only avg {summary['local_only_mean_f1']:.3f})",
        privacy_ledger(summary["eps"], summary["budget"], summary["delta"]),
    ]
    for site, gap in summary["overfit"].items():
        if gap["flag"]:
            lines.append(f"Overfitting: held-out loss gap {gap['gap']:.2f} at {site} → "
                         "rely on personal adapter only")
    c = summary["canary"]
    mark = "✔" if c["passed"] else "✘ BLOCKED"
    lines.append(f"Canary exposure: max {c['max_bits']:.1f} bits {mark} (< {c['threshold']:.0f})")
    lines.append(f"Gap to centralised: {summary['gap_to_centralised_pts']:.1f} pts "
                 f"(target ≤ 3)   sites improved vs local-only: "
                 f"{summary['sites_improved']}/{summary['sites']}")
    return "\n".join(lines)


def model_card(summary: dict) -> str:
    c = summary["canary"]
    rows = "\n".join(f"| {s} | {e:.2f} |" for s, e in sorted(summary["eps"].items()))
    return f"""# Model Card: FedLoRA Health global adapter

**Intended use:** assist clinician review (summarisation / ICD coding). Not for autonomous
clinical decisions.

## Training
- Federated LoRA (rank {summary['config']['rank']}) over {summary['sites']} sites, strategy
  `{summary['config']['strategy']}`, aggregation `{summary['config']['aggregation']}`.
- Secure aggregation with cohort minimum {summary['config']['min_cohort']}; DP:
  {'on' if summary['config']['dp'] else 'OFF'}.

## Quality (simulation)
- Global test F1: {summary['last_round']['global_f1']:.3f}
- Centralised baseline F1: {summary['centralised_f1']:.3f}
  (gap {summary['gap_to_centralised_pts']:.1f} pts)
- Sites improved vs local-only: {summary['sites_improved']}/{summary['sites']}

## Privacy (δ={summary['delta']:g}, budget ε={summary['budget']})
| Site | ε spent |
|---|---|
{rows}

## Memorisation
- Max canary exposure: {c['max_bits']:.2f} bits (threshold {c['threshold']:.0f}) -
  {'PASSED' if c['passed'] else 'FAILED, release blocked'}
- Membership-inference AUC (0.5 = none): {summary['mia_auc']:.3f}

## Limitations
- Synthetic-data simulation; real clinical validation required before deployment.
- Per-round privacy amplification by sampling is not credited (conservative).
- Factor-wise aggregation approximates averaging BA.
"""


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
