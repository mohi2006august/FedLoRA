"""`fed` command line (TECHNICAL.md section 4)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import typer
import yaml
from rich.console import Console

from fedlora.compliance.report import dashboard, privacy_ledger
from fedlora.config import FedConfig

app = typer.Typer(help="FedLoRA Health: privacy-preserving federated LoRA.", no_args_is_help=True)
privacy = typer.Typer(help="Privacy accounting.")
coordinator = typer.Typer(help="Federation coordinator.")
client = typer.Typer(help="Site client.")
app.add_typer(privacy, name="privacy")
app.add_typer(coordinator, name="coordinator")
app.add_typer(client, name="client")


@app.callback()
def _main() -> None:
    # Windows consoles default to cp1252, which cannot print the epsilon / bar glyphs.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


console = Console(highlight=False)
RUNS = Path("runs/latest")


def _run(cfg: FedConfig, out: Path) -> dict:
    from fedlora.sim.runner import run_and_save

    def tick(row):
        console.print(f"round {row['round']:>3}  n={row['participants']}  dropped={row['dropped']}  "
                      f"F1={row['global_f1']:.3f}  eps_max={row['eps_max']:.2f}")

    _, summary = run_and_save(cfg, out, progress=tick)
    console.print()
    console.print(dashboard(summary))
    console.print(f"\nArtefacts written to {out}/")
    return summary


@app.command()
def sim(sites: int = typer.Option(8, help="Number of synthetic hospitals."),
        dataset: str = typer.Option("clinical-notes-synthetic", help="Simulation dataset."),
        rounds: int = 10, clients_per_round: int = typer.Option(
            0, help="Sites per round; 0 = 90% of sites (larger cohorts help DP)."),
        epsilon: float = 8.0, dp: bool = True, personalize: bool = True,
        strategy: str = "fedprox", aggregation: str = "factors", seed: int = 0,
        out: Path = RUNS) -> None:
    """Simulated federation (synthetic notes split into hospitals)."""
    if dataset not in ("clinical-notes-synthetic", "clinical-notes-public"):
        raise typer.BadParameter("only the synthetic simulator is bundled")
    cohort = clients_per_round or max(3, int(round(0.9 * sites)))
    cfg = FedConfig(sites=sites, rounds=rounds, clients_per_round=cohort, target_epsilon=epsilon, dp=dp,
                    personalize=personalize, strategy=strategy, aggregation=aggregation,
                    seed=seed)
    _run(cfg, out)


@coordinator.command("start")
def coordinator_start(config: Path = typer.Option(..., help="fed.yaml"), out: Path = RUNS) -> None:
    """Run the coordinator from a config file (in-process simulated transport)."""
    cfg = FedConfig(**yaml.safe_load(config.read_text(encoding="utf-8")))
    _run(cfg, out)


@client.command("join")
def client_join(site: str = typer.Option(..., help="Site id, e.g. hospital-a")) -> None:
    """Validate site environment. Networked transport (Flower + mTLS) is not implemented yet."""
    env = {k: os.environ.get(k) for k in ("FED_COORDINATOR_URL", "FED_CERT_PATH")}
    missing = [k for k, v in env.items() if not v]
    console.print(f"site: {site}")
    if missing:
        console.print(f"missing environment: {', '.join(missing)}")
    console.print("Networked client transport is not implemented; use `fed sim` or "
                  "`fed coordinator start`.")
    raise typer.Exit(code=2)


@privacy.command("report")
def privacy_report(run: Path = RUNS) -> None:
    """Cumulative epsilon per site and audit verification."""
    f = run / "ledger.json"
    if not f.exists():
        console.print("No run found; run `fed sim` first.")
        raise typer.Exit(code=1)
    led = json.loads(f.read_text(encoding="utf-8"))
    console.print(privacy_ledger(led["eps"], led["budget"], led["delta"]))
    if led["excluded"]:
        console.print("Excluded (budget exhausted): " +
                      ", ".join(f"{s}@round {r}" for s, r in led["excluded"].items()))
    a = run / "audit.jsonl"
    if a.exists():
        from fedlora.compliance.audit import AuditLog
        log = AuditLog([json.loads(x) for x in a.read_text(encoding="utf-8").splitlines()])
        ok = log.verify_chain()
        console.print(f"Audit log: {len(log.entries)} entries, chain/signatures "
                      f"{'valid' if ok else 'INVALID'}")
        if not ok:
            raise typer.Exit(code=1)


@app.command()
def distill(student: str = typer.Option("3B", help="Student size: 1B, 3B or 7B."),
            run: Path = RUNS) -> None:
    """Distil the global (+ personal) adapters into a compact student."""
    from fedlora.coordinator.federation import build_model
    from fedlora.distill.distiller import agreement, distill as do_distill, evaluate_student
    from fedlora.model import Adapter
    from fedlora.sim.data import make_world

    f = run / "summary.json"
    if not f.exists():
        console.print("No run found; run `fed sim` first.")
        raise typer.Exit(code=1)
    cfg = FedConfig(**json.loads(f.read_text(encoding="utf-8"))["config"])
    world = make_world(cfg.sites, cfg.seed, cfg.dirichlet_alpha)
    model = build_model(world, cfg)
    z = np.load(run / "global_adapter.npz")
    g = Adapter(z["A"], z["B"])
    personal = {}
    sites_dir = run / "sites"
    for p in sites_dir.glob("*.npz") if sites_dir.exists() else []:
        d = np.load(p)
        personal[p.stem] = Adapter(d["A"], d["B"])
    s, curve = do_distill(model, g, personal, student=student, seed=cfg.seed,
                          pool=world.public.x)
    console.print(f"student {student}: {s.n_params()} params, KD loss {curve[0]:.3f} -> "
                  f"{curve[-1]:.3f}")
    console.print(f"student test F1 {evaluate_student(s, world.test):.3f}; agreement with "
                  f"teacher {agreement(s, model, g, world.test):.3f}")
    np.savez(run / f"student_{student}.npz", U=s.U, V=s.V, b=s.b)


if __name__ == "__main__":
    app()
