"""Networked coordinator: HTTPS with mutual TLS, round orchestration, SecAgg relay (FR-01..03).

The coordinator never sees training data, raw updates or private keys. It sees only masked
vectors, encrypted shares and aggregate metrics. Clients are authenticated by the CN of their
CA-signed certificate (not by anything they claim in a request body).

Endpoints (JSON):
  GET  /config                      -> FedConfig (public hyper-parameters)
  GET  /round?after=N               -> long-poll for round > N (cohort, certs, global adapter)
  POST /round/{r}/{phase}           -> submit and block until the phase closes, returns result
        phases: advert | shares | masked | sigs | unmask
  POST /round/{r}/report            -> SiteReport (aggregate metrics only)
"""
from __future__ import annotations

import ssl
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from fedlora.compliance.audit import AuditLog
from fedlora.config import FedConfig, SiteReport
from fedlora.coordinator.accountant import ORDERS, PrivacyAccountant
from fedlora.coordinator.federation import min_release, resolve_sigma, server_lr
from fedlora.model import Adapter
from fedlora.secagg.protocol import Advert, SecAggError, SecAggServer
from fedlora.sim.data import N_CLASSES, N_FEATURES
from fedlora.transport import certs
from fedlora.transport.wire import arr_d, arr_e, b64d, b64e, dumps, loads

PHASES = ("advert", "shares", "masked", "sigs", "unmask")


@dataclass
class Phase:
    sub: dict = field(default_factory=dict)  # site -> payload
    out: dict = field(default_factory=dict)  # site -> response
    closed: bool = False


@dataclass
class RoundState:
    r: int
    cohort: list[str]
    sigma: float
    k_target: int
    threshold: int
    abort: str | None = None
    phases: dict = field(default_factory=lambda: {p: Phase() for p in PHASES})
    secagg: SecAggServer | None = None
    done: bool = False


class Coordinator:
    def __init__(self, cfg: FedConfig, cert_dir: Path, *, host: str = "127.0.0.1", port: int = 0,
                 phase_timeout: float = 20.0, join_timeout: float = 30.0,
                 report_timeout: float = 3.0):
        if cfg.dp_mode != "client":
            raise ValueError("networked mode supports client-level DP only")
        self.cfg, self.cert_dir = cfg, cert_dir
        self.phase_timeout, self.join_timeout, self.report_timeout = (
            phase_timeout, join_timeout, report_timeout)
        self.ca = certs.load_cert((cert_dir / "ca.pem").read_bytes())
        self.cv = threading.Condition()
        self.sigma = resolve_sigma(cfg) if cfg.dp else 0.0
        self.g = Adapter.init(N_FEATURES, N_CLASSES, cfg.rank, np.random.default_rng(cfg.seed))
        self.acct = PrivacyAccountant(delta=cfg.delta, budget=cfg.target_epsilon,
                                      orders=list(ORDERS))
        self.audit = AuditLog()
        self.history: list[dict] = []
        self.reports: dict[int, dict[str, SiteReport]] = {}
        self.present: dict[str, float] = {}
        self.site_pems: dict[str, bytes] = {}
        self.current: RoundState | None = None
        self.finished = False
        self.excluded: dict[str, int] = {}
        self._rng = np.random.default_rng(cfg.seed + 7)
        self._http = self._make_server(host, port)
        self.port = self._http.server_address[1]
        self.threads: list[threading.Thread] = []

    # ---- lifecycle -------------------------------------------------------------------------
    def _make_server(self, host: str, port: int) -> ThreadingHTTPServer:
        coord = self

        class Handler(_Handler):
            c = coord

        srv = ThreadingHTTPServer((host, port), Handler)
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.minimum_version = ssl.TLSVersion.TLSv1_3
        ctx.load_cert_chain(self.cert_dir / "coordinator.pem", self.cert_dir / "coordinator.key")
        ctx.verify_mode = ssl.CERT_REQUIRED
        ctx.load_verify_locations(self.cert_dir / "ca.pem")
        srv.socket = ctx.wrap_socket(srv.socket, server_side=True, do_handshake_on_connect=False)
        srv.daemon_threads = True
        return srv

    def start(self) -> "Coordinator":
        for target in (self._http.serve_forever, self._run):
            t = threading.Thread(target=target, daemon=True)
            t.start()
            self.threads.append(t)
        return self

    def wait(self, timeout: float | None = None) -> bool:
        with self.cv:
            return self.cv.wait_for(lambda: self.finished, timeout)

    def stop(self) -> None:
        self._http.shutdown()
        self._http.server_close()

    # ---- round loop ------------------------------------------------------------------------
    def _eligible(self, sigma_floor: float, k_t: int) -> list[str]:
        out = []
        for s in sorted(self.present):
            if self.cfg.dp and self.acct.would_exceed(s, sigma_floor):
                self.excluded.setdefault(s, len(self.history) + 1)
                continue
            out.append(s)
        return out

    def _await_cohort(self) -> list[str] | None:
        deadline = time.time() + self.join_timeout
        with self.cv:
            while True:
                fresh = {s for s, t in self.present.items() if time.time() - t < 15}
                k_t = min(self.cfg.clients_per_round, len(fresh))
                floor = self.sigma * float(np.sqrt(min_release(self.cfg, max(k_t, 1)) / max(k_t, 1)))
                el = [s for s in self._eligible(floor, k_t) if s in fresh]
                if len(el) >= max(self.cfg.min_cohort, 1):
                    k = min(self.cfg.clients_per_round, len(el))
                    return sorted(self._rng.choice(el, k, replace=False).tolist())
                if time.time() > deadline:
                    return None
                self.cv.wait(0.2)

    def _wait_phase(self, rs: RoundState, name: str, expected: set[str]) -> None:
        deadline = time.time() + self.phase_timeout
        with self.cv:
            ph = rs.phases[name]
            while not expected <= set(ph.sub) and time.time() < deadline and not rs.abort:
                self.cv.wait(0.05)

    def _close(self, rs: RoundState, name: str) -> None:
        rs.phases[name].closed = True
        self.cv.notify_all()

    def _abort(self, rs: RoundState, reason: str) -> None:
        rs.abort = reason
        for ph in rs.phases.values():
            ph.closed = True
        self.audit.record("coordinator", rs.r, "0" * 64, event=f"round-aborted:{reason}")
        self.cv.notify_all()

    def _run(self) -> None:
        cfg = self.cfg
        for r in range(1, cfg.rounds + 1):
            cohort = self._await_cohort()
            if cohort is None:
                self.audit.record("coordinator", r, "0" * 64, event="halt:no-cohort")
                break
            k = len(cohort)
            rs = RoundState(r, cohort, self.sigma, k, max(cfg.min_cohort, k // 2 + 1))
            with self.cv:
                self.current = rs
                self.cv.notify_all()
            try:
                self._do_round(rs)
            except SecAggError as e:
                with self.cv:
                    self._abort(rs, f"secagg:{e}")
                    self.history.append({"round": r, "participants": k, "aborted": True})
            with self.cv:
                rs.done = True
                self.cv.notify_all()
            self._wait_reports(rs)
        with self.cv:
            self.finished = True
            self.cv.notify_all()
        time.sleep(0.5)  # let final long-polls return {"done": true}

    def _do_round(self, rs: RoundState) -> None:
        cohort, uid = rs.cohort, {s: i for i, s in enumerate(rs.cohort)}
        sa = rs.secagg = SecAggServer(rs.r, rs.threshold, self.g.n_params())
        # 0 advert
        self._wait_phase(rs, "advert", set(cohort))
        with self.cv:
            subs = rs.phases["advert"].sub
            adverts = sa.collect_adverts([_advert(uid[s], p) for s, p in subs.items()])
            common = {"adverts": [_advert_json(a) for a in adverts]}
            rs.phases["advert"].out = {s: common for s in subs}
            self._close(rs, "advert")
        # 1 shares
        self._wait_phase(rs, "shares", {s for s in cohort if s in rs.phases["advert"].sub})
        with self.cv:
            subs = rs.phases["shares"].sub
            inbox = sa.relay_shares({uid[s]: {int(v): b64d(ct) for v, ct in p["cts"].items()}
                                     for s, p in subs.items()})
            for s in subs:
                rs.phases["shares"].out[s] = {
                    "u2": sa.u2, "inbox": {str(u): b64e(ct) for u, ct in inbox[uid[s]].items()}}
            self._close(rs, "shares")
        u2_sites = {cohort[i] for i in sa.u2}
        # 2 masked inputs
        self._wait_phase(rs, "masked", u2_sites)
        with self.cv:
            subs = rs.phases["masked"].sub
            for s, p in subs.items():  # verify + audit-log the signed upload hash
                self._audit_upload(s, rs.r, p)
            u3 = sa.collect_masked({uid[s]: arr_d(p["y"], "<u4").astype(np.uint64)
                                    for s, p in subs.items()})
            for s in subs:
                rs.phases["masked"].out[s] = {"u3": u3}
            self._close(rs, "masked")
        # 3 survivor signatures
        u3_sites = {cohort[i] for i in u3}
        self._wait_phase(rs, "sigs", u3_sites)
        with self.cv:
            subs = rs.phases["sigs"].sub
            sigs = {str(uid[s]): p["sig"] for s, p in subs.items()}
            for s in subs:
                rs.phases["sigs"].out[s] = {"u3": u3, "u2": sa.u2, "sigs": sigs}
            self._close(rs, "sigs")
        # 4 unmasking
        self._wait_phase(rs, "unmask", set(rs.phases["sigs"].sub))
        with self.cv:
            subs = rs.phases["unmask"].sub
            replies = {uid[s]: {int(v): (kind, int(share)) for v, (kind, share) in p["reply"].items()}
                       for s, p in subs.items()}
            n_surv = len(u3)
            if n_surv < min_release(self.cfg, rs.k_target):
                raise SecAggError("too few survivors to release")
            total = sa.finish(replies)  # sum over survivors of delta_i / k_target
            self.g = Adapter.from_flat(
                self.g.flat() + server_lr(self.cfg, rs.r) * total * (rs.k_target / n_surv), self.g)
            if self.cfg.dp:
                sigma_eff = self.sigma * float(np.sqrt(n_surv / rs.k_target))
                for s in cohort:
                    self.acct.step(s, sigma_eff, rs.r)
            out = {"global": arr_e(self.g.flat(), "<f8")}
            for s in subs:
                rs.phases["unmask"].out[s] = out
            self.history.append({"round": rs.r, "participants": len(cohort), "survivors": n_surv,
                                 "aborted": False,
                                 "eps_max": max((self.acct.epsilon(s) for s in self.present),
                                                default=0.0)})
            self._close(rs, "unmask")

    def _audit_upload(self, site: str, r: int, payload: dict) -> None:
        import hashlib
        sha = hashlib.sha256(b64d(payload["y"])).hexdigest()
        pub = Ed25519PublicKey.from_public_bytes(
            certs.verify_site_cert(self.site_pems[site], self.ca, site))
        pub.verify(b64d(payload["upload_sig"]), f"upload|{site}|{r}|{sha}".encode())
        self.audit.record_signed(site, r, sha, payload["upload_sig"])

    def _wait_reports(self, rs: RoundState) -> None:
        deadline = time.time() + self.report_timeout
        with self.cv:
            want = set(rs.phases["unmask"].sub)
            while want - set(self.reports.get(rs.r, {})) and time.time() < deadline:
                self.cv.wait(0.05)

    # ---- request handling (called from handler threads) --------------------------------------
    def config_json(self) -> dict:
        return {"cfg": self.cfg.model_dump(mode="json"), "sigma": self.sigma}

    def poll_round(self, site: str, pem: bytes, after: int, wait: float = 25.0) -> dict:
        deadline = time.time() + wait
        with self.cv:
            self.site_pems[site] = pem
            while True:
                self.present[site] = time.time()
                if self.finished:
                    return {"done": True, "global": arr_e(self.g.flat(), "<f8"),
                            "eps": self.acct.epsilon(site)}
                rs = self.current
                if rs is not None and rs.r > after:
                    if site not in rs.cohort:
                        return {"round": rs.r, "participant": False}
                    return {"round": rs.r, "participant": True, "cohort": rs.cohort,
                            "certs": {s: b64e(self.site_pems[s]) for s in rs.cohort
                                      if s in self.site_pems},
                            "global": arr_e(self.g.flat(), "<f8"), "sigma": rs.sigma,
                            "k_target": rs.k_target, "threshold": rs.threshold,
                            "clip": self.cfg.clip_norm, "lr": self.cfg.lr,
                            "epochs": self.cfg.local_epochs, "mu": self.cfg.mu,
                            "strategy": self.cfg.strategy, "dp": self.cfg.dp}
                if time.time() > deadline:
                    return {"round": after, "participant": False, "idle": True}
                self.cv.wait(0.2)

    def submit(self, site: str, r: int, phase: str, payload: dict) -> dict:
        with self.cv:
            rs = self.current
            if rs is None or rs.r != r or site not in rs.cohort:
                return {"abort": "not in this round"}
            ph = rs.phases[phase]
            if ph.closed and site not in ph.sub:
                return {"abort": "late"}
            ph.sub[site] = payload
            self.cv.notify_all()
            while not ph.closed:
                self.cv.wait(0.05)
            if rs.abort:
                return {"abort": rs.abort}
            return ph.out.get(site, {"abort": "excluded"})

    def report(self, site: str, r: int, payload: dict) -> dict:
        rep = SiteReport(**payload)
        if rep.site_id != site:
            return {"error": "site mismatch"}
        with self.cv:
            rep.epsilon_spent = self.acct.epsilon(site)  # ledger lives only at the coordinator
            self.reports.setdefault(r, {})[site] = rep
            self.cv.notify_all()
        return {"ok": True}


def _advert(uid: int, p: dict) -> Advert:
    return Advert(uid, b64d(p["c_pk"]), b64d(p["s_pk"]), b64d(p["sig"]))


def _advert_json(a: Advert) -> dict:
    return {"uid": a.uid, "c_pk": b64e(a.c_pk), "s_pk": b64e(a.s_pk), "sig": b64e(a.sig)}


class _Handler(BaseHTTPRequestHandler):
    c: Coordinator
    protocol_version = "HTTP/1.1"

    def setup(self):
        self.request.settimeout(60)
        self.request.do_handshake()  # mTLS: fails here without a CA-signed client certificate
        super().setup()

    def log_message(self, *a):  # structured logging lives in the audit log, not stderr noise
        pass

    def _site(self) -> tuple[str, bytes]:
        der = self.connection.getpeercert(binary_form=True)
        pem = certs.x509.load_der_x509_certificate(der).public_bytes(serialization.Encoding.PEM)
        cn = certs.cert_cn(certs.load_cert(pem))
        certs.verify_site_cert(pem, self.c.ca, cn)  # belt and braces on top of TLS verification
        return cn, pem

    def _send(self, code: int, obj) -> None:
        body = dumps(obj)
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        try:
            site, pem = self._site()
            path, _, query = self.path.partition("?")
            if path == "/config":
                return self._send(200, self.c.config_json())
            if path == "/round":
                after = int(dict(q.split("=") for q in query.split("&") if q).get("after", 0))
                return self._send(200, self.c.poll_round(site, pem, after))
            self._send(404, {"error": "not found"})
        except Exception as e:  # noqa: BLE001 - report to the client, never crash the server
            self._send(400, {"error": f"{type(e).__name__}: {e}"})

    def do_POST(self):
        try:
            site, _ = self._site()
            n = int(self.headers.get("Content-Length", 0))
            payload = loads(self.rfile.read(n)) if n else {}
            parts = self.path.strip("/").split("/")
            if len(parts) == 3 and parts[0] == "round":
                r, phase = int(parts[1]), parts[2]
                if phase == "report":
                    return self._send(200, self.c.report(site, r, payload))
                if phase in PHASES:
                    return self._send(200, self.c.submit(site, r, phase, payload))
            self._send(404, {"error": "not found"})
        except Exception as e:  # noqa: BLE001
            self._send(400, {"error": f"{type(e).__name__}: {e}"})
