import json

from typer.testing import CliRunner

from fedlora.cli import app
from fedlora.compliance.audit import AuditLog, update_sha256
from fedlora.compliance.report import privacy_ledger
from fedlora.config import bucket_size
import numpy as np

runner = CliRunner()


def test_sim_privacy_report_and_distill_roundtrip(tmp_path):
    out = tmp_path / "run"
    r = runner.invoke(app, ["sim", "--sites", "4", "--rounds", "3", "--out", str(out)])
    assert r.exit_code == 0, r.output
    for f in ("summary.json", "ledger.json", "audit.jsonl", "MODEL_CARD.md", "global_adapter.npz"):
        assert (out / f).exists()
    r = runner.invoke(app, ["privacy", "report", "--run", str(out)])
    assert r.exit_code == 0 and "valid" in r.output
    r = runner.invoke(app, ["distill", "--student", "1B", "--run", str(out)])
    assert r.exit_code == 0, r.output


def test_no_raw_text_or_exact_sizes_in_artefacts(tmp_path):
    out = tmp_path / "run"
    runner.invoke(app, ["sim", "--sites", "4", "--rounds", "2", "--out", str(out)])
    blob = (out / "summary.json").read_text() + (out / "audit.jsonl").read_text()
    assert "n_examples" not in blob  # only bucketed sizes may ever be reported
    assert json.loads((out / "summary.json").read_text())["canary"]["threshold"] == 10


def test_tampered_audit_chain_detected():
    log = AuditLog()
    log.record("hospital-a", 1, update_sha256(np.ones(3)))
    log.record("hospital-b", 1, update_sha256(np.zeros(3)))
    assert log.verify_chain()
    log.entries[0]["sha256"] = "0" * 64
    assert not log.verify_chain()


def test_bucketing_and_ledger_warning():
    assert bucket_size(37) == "<100" and bucket_size(7000) == ">=5000"
    assert "⚠" in privacy_ledger({"a": 7.0, "b": 1.0}, 8.0, 1e-5)


def test_client_join_refuses_without_transport():
    assert runner.invoke(app, ["client", "join", "--site", "hospital-a"]).exit_code == 2


def test_sweep_command_runs():
    r = runner.invoke(app, ["sweep", "--sites", "4", "--seeds", "1", "--rounds", "2,3"])
    assert r.exit_code == 0, r.output
    assert r.output.count("gap") == 2
