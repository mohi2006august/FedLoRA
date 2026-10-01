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
        dataset: str = typer.Option("synthetic", help="synthetic | csv:<path> (text,label columns)"),
        rounds: int = 10, clients_per_round: int = typer.Option(
            0, help="Sites per round; 0 = 90% of sites (larger cohorts help DP)."),
        epsilon: float = 8.0, dp: bool = True, personalize: bool = True,
        strategy: str = "fedprox", aggregation: str = "factors", seed: int = 0,
        dp_mode: str = typer.Option("client", help="client (update-level) | sample (DP-SGD)"),
        sampling: str = typer.Option("fixed", help="fixed | poisson"),
        accounting: str = typer.Option("participation", help="participation | amplified"),
        out: Path = RUNS) -> None:
    """Simulated federation (synthetic notes split into hospitals)."""
    if dataset in ("clinical-notes-synthetic", "clinical-notes-public"):  # spec aliases
        dataset = "synthetic"
    if dataset != "synthetic" and not dataset.startswith("csv:"):
        raise typer.BadParameter("dataset must be 'synthetic' or 'csv:<path>'")
    cohort = clients_per_round or max(3, int(round(0.9 * sites)))
    cfg = FedConfig(dataset=dataset, sites=sites, rounds=rounds, clients_per_round=cohort, target_epsilon=epsilon, dp=dp,
                    personalize=personalize, strategy=strategy, aggregation=aggregation,
                    seed=seed, sampling=sampling, accounting=accounting, dp_mode=dp_mode)
    _run(cfg, out)


def _floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",")]


@app.command()
def sweep(sites: int = 8, seeds: int = 3, dp: bool = True,
          lr: str = typer.Option("1.0", help="Comma-separated values."),
          local_epochs: str = typer.Option("3", help="Comma-separated values."),
          rounds: str = typer.Option("10", help="Comma-separated values."),
          clip_norm: str = typer.Option("1.0", help="Comma-separated values."),
          server_lr_decay: str = typer.Option("0.5", help="Comma-separated values."),
          mu: str = typer.Option("0.01", help="Comma-separated values."),
          dp_mode: str = "client", sampling: str = "fixed",
          accounting: str = "participation") -> None:
    """Grid search averaged over seeds; reports gap to centralised and sites improved."""
    from fedlora.sim.sweep import sweep as run_sweep

    grid = {"lr": _floats(lr), "local_epochs": [int(v) for v in _floats(local_epochs)],
            "rounds": [int(v) for v in _floats(rounds)], "clip_norm": _floats(clip_norm),
            "server_lr_decay": _floats(server_lr_decay), "mu": _floats(mu)}
    base = {"dp_mode": dp_mode, "sampling": sampling, "accounting": accounting}
    for row in run_sweep(grid, sites=sites, seeds=seeds, dp=dp, base=base):
        console.print(f"{row.params}  gap {row.gap_pts:5.1f} pts  improved "
                      f"{100 * row.sites_improved_frac:3.0f}%  F1 {row.global_f1:.3f}  "
                      f"eps_max {row.eps_max:.2f}")


@coordinator.command("start")
def coordinator_start(config: Path = typer.Option(..., help="fed.yaml"), out: Path = RUNS) -> None:
    """Run the coordinator from a config file (in-process simulated transport)."""
    cfg = FedConfig(**yaml.safe_load(config.read_text(encoding="utf-8")))
    _run(cfg, out)


@app.command("certs")
def certs_init(out: Path = typer.Option(Path("certs"), help="Directory for the private PKI."),
               sites: str = typer.Option(..., help="Comma-separated site ids."),
               hosts: str = typer.Option("localhost,127.0.0.1", help="Coordinator SAN hosts.")) -> None:
    """Create a CA, a coordinator certificate and one certificate per site (mTLS)."""
    from fedlora.transport.certs import init_pki

    ids = [x.strip() for x in sites.split(",") if x.strip()]
    init_pki(out, ids, tuple(h.strip() for h in hosts.split(",")))
    console.print(f"PKI written to {out}/ for {len(ids)} sites. Keep *.key private; never commit them.")


@coordinator.command("serve")
def coordinator_serve(config: Path = typer.Option(..., help="fed.yaml"),
                      certs_dir: Path = typer.Option(None, "--certs", envvar="FED_CERT_PATH"),
                      host: str = "0.0.0.0", port: int = 8443, out: Path = RUNS,
                      phase_timeout: float = 60.0) -> None:
    """Run the real networked coordinator (HTTPS, mutual TLS, secure aggregation)."""
    from fedlora.compliance.report import save_json
    from fedlora.transport.server import Coordinator

    if certs_dir is None:
        console.print("Set --certs or FED_CERT_PATH (see `fed certs`).")
        raise typer.Exit(code=2)
    cfg = FedConfig(**yaml.safe_load(config.read_text(encoding="utf-8")))
    coord = Coordinator(cfg, certs_dir, host=host, port=port, phase_timeout=phase_timeout).start()
    console.print(f"coordinator listening on https://{host}:{coord.port} (mTLS required)")
    coord.wait()
    coord.drain()
    coord.stop()
    save_json(out / "history.json", coord.history)
    save_json(out / "ledger.json", {
        "budget": cfg.target_epsilon, "delta": cfg.delta, "accounting": "participation",
        "dp_mode": "client", "entries": coord.acct.ledger, "excluded": coord.excluded,
        "eps": {s: coord.acct.epsilon(s) for s in sorted(coord.present)}, "eps_alt": {}})
    coord.audit.export(str(out / "audit.jsonl"))
    console.print(f"done: {len(coord.history)} rounds; artefacts in {out}/")


@client.command("join")
def client_join(site: str = typer.Option(..., envvar="SITE_ID", help="Site id, e.g. hospital-a"),
                coordinator_url: str = typer.Option(None, "--coordinator",
                                                    envvar="FED_COORDINATOR_URL"),
                certs_dir: Path = typer.Option(None, "--certs", envvar="FED_CERT_PATH"),
                data: str = typer.Option("sim", help="'sim' (synthetic) or csv:<path> (local notes)")
                ) -> None:
    """Join the federation as a site. Data is read locally and never transmitted."""
    from fedlora.transport.client import SiteClient, sim_data

    missing = [n for n, v in (("FED_COORDINATOR_URL / --coordinator", coordinator_url),
                              ("FED_CERT_PATH / --certs", certs_dir)) if not v]
    if missing:
        console.print(f"missing: {', '.join(missing)}")
        raise typer.Exit(code=2)
    if data == "sim":
        factory = sim_data(site)
    else:
        from fedlora.sim.loaders import csv_site_data
        factory = csv_site_data(site, Path(data.removeprefix("csv:")))
    try:
        res = SiteClient(site, coordinator_url, certs_dir, factory).run()
    except ConnectionError as e:
        console.print(f"error: {e}")
        raise typer.Exit(code=1)
    console.print(f"{site}: took part in {res.rounds_participated} rounds; "
                  f"coordinator-reported eps {res.epsilon:.2f}")


@privacy.command("report")
def privacy_report(run: Path = RUNS) -> None:
    """Cumulative epsilon per site and audit verification."""
    f = run / "ledger.json"
    if not f.exists():
        console.print("No run found; run `fed sim` first.")
        raise typer.Exit(code=1)
    led = json.loads(f.read_text(encoding="utf-8"))
    console.print(privacy_ledger(led["eps"], led["budget"], led["delta"]))
    if led.get("dp_mode") == "sample":
        console.print("Guarantee: record-level (DP-SGD) - each patient note is protected; "
                      "a site's participation is not hidden.")
    elif led.get("accounting") == "amplified":
        worst = max(led["eps_alt"].values())
        console.print(f"Accounting: amplified (Poisson sampling credited; valid only if sampling is "
                      f"hidden from the adversary). Coordinator-visible worst-case eps: {worst:.2f}")
    elif led.get("eps_alt") and led.get("dp_mode") != "sample":
        console.print(f"Accounting: participation (valid against the coordinator). "
                      f"If sampling were hidden, worst-site eps would be "
                      f"{max(led['eps_alt'].values()):.2f}")
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
    from fedlora.personalize.local import PersonalModel
    from fedlora.sim.loaders import build_world

    f = run / "summary.json"
    if not f.exists():
        console.print("No run found; run `fed sim` first.")
        raise typer.Exit(code=1)
    cfg = FedConfig(**json.loads(f.read_text(encoding="utf-8"))["config"])
    world = build_world(cfg)
    model = build_model(world, cfg)
    z = np.load(run / "global_adapter.npz")
    g = Adapter(z["A"], z["B"])
    personal = {}
    sites_dir = run / "sites"
    for p in sites_dir.glob("*.npz") if sites_dir.exists() else []:
        d = np.load(p)
        personal[p.stem] = PersonalModel(Adapter(d["A"], d["B"]), float(d["g_scale"]))
    s, curve = do_distill(model, g, personal, student=student, seed=cfg.seed,
                          pool=world.public.x)
    console.print(f"student {student}: {s.n_params()} params, KD loss {curve[0]:.3f} -> "
                  f"{curve[-1]:.3f}")
    console.print(f"student test F1 {evaluate_student(s, world.test):.3f}; agreement with "
                  f"teacher {agreement(s, model, g, world.test):.3f}")
    np.savez(run / f"student_{student}.npz", U=s.U, V=s.V, b=s.b)


if __name__ == "__main__":
    app()
