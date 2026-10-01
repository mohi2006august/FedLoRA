import csv

import numpy as np
import pytest
from typer.testing import CliRunner

from fedlora.cli import app
from fedlora.config import FedConfig
from fedlora.sim.data import N_CLASSES, N_FEATURES
from fedlora.sim.loaders import csv_site_data, read_csv, vectorize, world_from_csv

VOCAB = [["chest", "pain", "troponin", "ecg", "stemi"], ["cough", "fever", "infiltrate", "xray", "lung"],
         ["insulin", "glucose", "diabetes", "a1c", "thirst"], ["fracture", "femur", "ortho", "cast", "fall"]]


def _write(path, n=600, site=False, seed=0):
    rng = np.random.default_rng(seed)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "label"] + (["site"] if site else []))
        for i in range(n):
            y = int(rng.integers(0, 4))
            words = list(rng.choice(VOCAB[y], 8)) + list(rng.choice(sum(VOCAB, []), 3))
            row = [" ".join(words), y] + ([f"h{i % 4}"] if site else [])
            w.writerow(row)


def test_vectorize_is_hashed_normalised_and_deterministic():
    v = vectorize("Chest pain, chest pain; ECG: ST elevation")
    assert v.shape == (N_FEATURES,) and abs(v.sum() - 1) < 1e-9
    assert np.array_equal(v, vectorize("Chest pain, chest pain; ECG: ST elevation"))
    assert vectorize("").sum() == 0


def test_rejects_labels_outside_range(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("text,label\nhello,99\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_csv(p)


def test_world_from_csv_partitions_and_has_no_overlap(tmp_path):
    _write(tmp_path / "n.csv")
    w = world_from_csv(tmp_path / "n.csv", n_sites=4, seed=1)
    assert len(w.sites) == 4 and all(len(s.train) > 10 for s in w.sites)
    total = sum(len(s.train) + len(s.heldout) for s in w.sites) + len(w.public) + len(w.test)
    assert total == 600  # every note used exactly once
    h = [np.bincount(s.train.y, minlength=N_CLASSES) / len(s.train) for s in w.sites]
    assert np.abs(h[0] - h[1]).sum() > 0.1  # label-skewed hospitals


def test_pre_partitioned_site_column(tmp_path):
    _write(tmp_path / "n.csv", site=True)
    from fedlora.sim.loaders import read_csv as rc, world_from_arrays
    x, y, sites = rc(tmp_path / "n.csv", site_col="site")
    w = world_from_arrays(x, y, 0, site_labels=sites)
    assert [s.site_id for s in w.sites] == ["h0", "h1", "h2", "h3"]


def test_sim_runs_on_csv_dataset(tmp_path):
    _write(tmp_path / "n.csv", n=900)
    r = CliRunner().invoke(app, ["sim", "--dataset", f"csv:{tmp_path / 'n.csv'}", "--sites", "4",
                                 "--rounds", "3", "--no-dp", "--out", str(tmp_path / "run")])
    assert r.exit_code == 0, r.output
    assert (tmp_path / "run" / "summary.json").exists()
    # distill reloads the same CSV world from the stored config
    r = CliRunner().invoke(app, ["distill", "--student", "1B", "--run", str(tmp_path / "run")])
    assert r.exit_code == 0, r.output


def test_networked_client_factory_reads_only_local_file(tmp_path):
    _write(tmp_path / "local.csv", n=200)
    site, public = csv_site_data("hospital-a", tmp_path / "local.csv")(FedConfig())
    assert len(site.train) + len(site.heldout) == 200 and len(public) == 1
