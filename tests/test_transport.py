import ssl
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from fedlora.config import FedConfig
from fedlora.sim.data import site_name
from fedlora.transport import certs
from fedlora.transport.client import SiteClient, sim_data
from fedlora.transport.server import Coordinator


def _cfg(**kw):
    base = dict(sites=5, clients_per_round=5, rounds=3, personalize=False, dp=True,
                dropout_prob=0.0, min_cohort=3, local_epochs=1)
    base.update(kw)
    return FedConfig(**base)


def _run(tmp_path, cfg, fail=None, phase_timeout=10.0):
    ids = [site_name(i) for i in range(cfg.sites)]
    certs.init_pki(tmp_path, ids)
    coord = Coordinator(cfg, tmp_path, phase_timeout=phase_timeout, join_timeout=20,
                        report_timeout=2).start()
    url = f"https://127.0.0.1:{coord.port}"
    results, errors = {}, []

    def work(sid):
        try:
            c = SiteClient(sid, url, tmp_path, sim_data(sid), fail_at=(fail or {}).get(sid))
            results[sid] = c.run()
        except Exception as e:  # noqa: BLE001
            errors.append((sid, repr(e)))

    ts = [threading.Thread(target=work, args=(s,)) for s in ids]
    [t.start() for t in ts]
    assert coord.wait(timeout=120), "federation did not finish"
    [t.join(timeout=30) for t in ts]
    coord.stop()
    return coord, results, errors


def test_networked_federation_end_to_end_with_mtls(tmp_path):
    cfg = _cfg()
    coord, results, errors = _run(tmp_path, cfg)
    assert not errors, errors
    assert len(coord.history) == 3 and not any(h["aborted"] for h in coord.history)
    final = coord.g.flat()
    assert np.isfinite(final).all() and np.abs(final).sum() > 0
    for res in results.values():  # every site ends with exactly the coordinator's adapter
        assert np.allclose(res.global_adapter.flat(), final)
        assert res.rounds_participated == 3
    assert coord.audit.verify_chain()
    signed = [e for e in coord.audit.entries if e["event"] == "update-signed"]
    assert len(signed) == 15  # 5 sites x 3 rounds, each with a verified Ed25519 signature
    assert len(coord.reports[3]) == 5 and all(r.epsilon_spent > 0 for r in coord.reports[3].values())
    assert all(coord.acct.epsilon(s) <= cfg.target_epsilon for s in results)


def test_dropout_mid_protocol_is_recovered_over_the_network(tmp_path):
    cfg = _cfg(rounds=2)
    victim = site_name(0)
    coord, results, errors = _run(tmp_path, cfg, fail={victim: "masked"}, phase_timeout=3.0)
    assert not errors, errors
    assert len(coord.history) == 2 and not any(h["aborted"] for h in coord.history)
    assert all(h["survivors"] == 4 for h in coord.history)
    assert results[victim].rounds_participated == 0
    assert np.isfinite(coord.g.flat()).all()


def test_connection_without_client_certificate_is_refused(tmp_path):
    cfg = _cfg(rounds=1)
    certs.init_pki(tmp_path, [site_name(i) for i in range(cfg.sites)])
    coord = Coordinator(cfg, tmp_path, join_timeout=1).start()
    try:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.load_verify_locations(tmp_path / "ca.pem")  # trusts the server, presents no cert
        with pytest.raises((urllib.error.URLError, ssl.SSLError, ConnectionError, OSError)):
            urllib.request.urlopen(f"https://127.0.0.1:{coord.port}/config", context=ctx, timeout=5).read()
    finally:
        coord.stop()


def test_certificate_from_a_foreign_ca_is_refused(tmp_path):
    cfg = _cfg(rounds=1)
    good, rogue = tmp_path / "good", tmp_path / "rogue"
    ids = [site_name(i) for i in range(cfg.sites)]
    certs.init_pki(good, ids)
    certs.init_pki(rogue, ids)  # a different CA minting a cert with a valid-looking CN
    coord = Coordinator(cfg, good, join_timeout=1).start()
    try:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.load_verify_locations(good / "ca.pem")
        ctx.load_cert_chain(rogue / f"{ids[0]}.pem", rogue / f"{ids[0]}.key")
        with pytest.raises((urllib.error.URLError, ssl.SSLError, ConnectionError, OSError)):
            urllib.request.urlopen(f"https://127.0.0.1:{coord.port}/config", context=ctx, timeout=5).read()
    finally:
        coord.stop()


def test_site_cert_verification_checks_cn_and_issuer(tmp_path):
    certs.init_pki(tmp_path, ["hospital-a", "hospital-b"])
    ca = certs.load_cert((tmp_path / "ca.pem").read_bytes())
    pem = (tmp_path / "hospital-a.pem").read_bytes()
    assert len(certs.verify_site_cert(pem, ca, "hospital-a")) == 32
    with pytest.raises(ValueError):
        certs.verify_site_cert(pem, ca, "hospital-b")  # impersonation: wrong CN
    other = tmp_path / "other"
    certs.init_pki(other, ["hospital-a"])
    with pytest.raises(Exception):
        certs.verify_site_cert((other / "hospital-a.pem").read_bytes(), ca, "hospital-a")
