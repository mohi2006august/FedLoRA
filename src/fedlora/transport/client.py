"""Site client for the networked federation (outbound-only HTTPS connections, mTLS).

Everything that touches data - local training, clipping, DP noise, evaluation, personalisation -
runs here. Only a masked, noised update, encrypted key shares and aggregate metrics leave.
"""
from __future__ import annotations

import hashlib
import ssl
import urllib.error
import urllib.request
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from fedlora.client.dp import add_noise, clip_update
from fedlora.client.local_eval import evaluate
from fedlora.client.qlora_client import local_update
from fedlora.config import FedConfig, SiteReport, bucket_size
from fedlora.model import Adapter, LoRAClassifier, pretrain_base
from fedlora.personalize.local import PersonalModel, personalize_site
from fedlora.secagg.protocol import Advert, SecAggClient, SecAggError
from fedlora.sim.data import N_CLASSES, Site, Split, make_world
from fedlora.transport import certs
from fedlora.transport.wire import arr_d, arr_e, b64d, b64e, dumps, loads

DataFactory = Callable[[FedConfig], tuple[Site, Split]]


def sim_data(site_id: str) -> DataFactory:
    """Local data for a simulated site: regenerated deterministically from the shared seed."""
    def factory(cfg: FedConfig) -> tuple[Site, Split]:
        world = make_world(cfg.sites, cfg.seed, cfg.dirichlet_alpha)
        return next(s for s in world.sites if s.site_id == site_id), world.public
    return factory


@dataclass
class SiteResult:
    global_adapter: Adapter
    personal: PersonalModel | None
    rounds_participated: int
    epsilon: float


class SiteClient:
    def __init__(self, site_id: str, url: str, cert_dir: Path, data: DataFactory, *,
                 fail_at: str | None = None, request_timeout: float = 90.0):
        self.site_id, self.url, self.data, self.fail_at = site_id, url.rstrip("/"), data, fail_at
        self.timeout = request_timeout
        bundle = certs.Bundle.for_party(cert_dir, site_id)
        self.ca = certs.load_cert(bundle.ca.read_bytes())
        self.pem = bundle.cert.read_bytes()
        self.identity = certs.load_key(bundle.key)
        self.ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        self.ctx.minimum_version = ssl.TLSVersion.TLSv1_3
        self.ctx.load_verify_locations(bundle.ca)
        self.ctx.load_cert_chain(bundle.cert, bundle.key)
        self.participated = 0

    # ---- transport -------------------------------------------------------------------------
    def _call(self, method: str, path: str, body: dict | None = None) -> dict:
        req = urllib.request.Request(self.url + path, method=method,
                                     data=dumps(body) if body is not None else None,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, context=self.ctx, timeout=self.timeout) as r:
                return loads(r.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"{path}: HTTP {e.code} {e.read().decode()[:200]}") from e

    # ---- main loop -------------------------------------------------------------------------
    def run(self) -> SiteResult:
        cfgj = self._call("GET", "/config")
        cfg = FedConfig(**cfgj["cfg"])
        site, public = self.data(cfg)
        w0, b0 = pretrain_base(public.x, public.y, N_CLASSES, seed=cfg.seed)
        model = LoRAClassifier(w0, b0, cfg.alpha, cfg.rank, quantize=True)
        after = 0
        while True:
            msg = self._call("GET", f"/round?after={after}")
            if msg.get("done"):
                g = _adapter(msg["global"], cfg)
                rng = np.random.default_rng([cfg.seed, zlib.crc32(self.site_id.encode())])
                pm = personalize_site(model, site, g, cfg, rng) if cfg.personalize else None
                return SiteResult(g, pm, self.participated, float(msg.get("eps", 0.0)))
            if msg.get("idle"):
                continue
            after = msg["round"]
            if msg["participant"]:
                try:
                    self._round(cfg, model, site, msg)
                except SecAggError as e:
                    print(f"[{self.site_id}] round {after} aborted: {e}")

    def _round(self, cfg: FedConfig, model: LoRAClassifier, site: Site, msg: dict) -> None:
        r, cohort = msg["round"], msg["cohort"]
        uid = cohort.index(self.site_id)
        pubs = {i: certs.verify_site_cert(b64d(msg["certs"][s]), self.ca, s)
                for i, s in enumerate(cohort)}
        g = _adapter(msg["global"], cfg)
        rng = np.random.default_rng([cfg.seed, r, zlib.crc32(self.site_id.encode())])
        delta = local_update(model, site, g, epochs=msg["epochs"], lr=msg["lr"],
                             strategy=msg["strategy"], mu=msg["mu"], rng=rng)
        if msg["dp"]:
            delta, _ = clip_update(delta, msg["clip"])
            delta = add_noise(delta, msg["clip"], msg["sigma"], msg["k_target"], rng)
        x = delta / msg["k_target"]  # uniform weight; sizes are never transmitted
        sa = SecAggClient(uid, self.identity, pubs, r, msg["threshold"])
        base = f"/round/{r}/"
        if self.fail_at == "advert":
            return
        a = sa.advertise()
        res = self._call("POST", base + "advert", {"c_pk": b64e(a.c_pk), "s_pk": b64e(a.s_pk),
                                                   "sig": b64e(a.sig)})
        _check(res)
        adverts = [Advert(d["uid"], b64d(d["c_pk"]), b64d(d["s_pk"]), b64d(d["sig"]))
                   for d in res["adverts"]]
        if self.fail_at == "shares":
            return
        cts = sa.share_keys(adverts)
        res = self._call("POST", base + "shares", {"cts": {str(v): b64e(c) for v, c in cts.items()}})
        _check(res)
        sa.receive_shares({int(u): b64d(c) for u, c in res["inbox"].items()})
        u2 = res["u2"]
        if self.fail_at == "masked":
            return
        y = sa.masked_input(x, u2)
        y_bytes = y.astype("<u4").tobytes()
        sha = hashlib.sha256(y_bytes).hexdigest()
        up_sig = self.identity.sign(f"upload|{self.site_id}|{r}|{sha}".encode())
        res = self._call("POST", base + "masked", {"y": b64e(y_bytes), "upload_sig": b64e(up_sig)})
        _check(res)
        u3 = res["u3"]
        if self.fail_at == "sigs":
            return
        res = self._call("POST", base + "sigs", {"sig": b64e(sa.sign_survivors(u3))})
        _check(res)
        if self.fail_at == "unmask":
            return
        reply = sa.unmask(res["u3"], {int(k): b64d(v) for k, v in res["sigs"].items()}, res["u2"])
        res = self._call("POST", base + "unmask", {"reply": {str(v): [k, str(s)]
                                                           for v, (k, s) in reply.items()}})
        _check(res)
        new_g = _adapter(res["global"], cfg)
        self.participated += 1
        loss, f1 = evaluate(model, site.heldout, new_g)
        rep = SiteReport(site_id=self.site_id, round=r, n_examples_bucket=bucket_size(len(site.train)),
                         heldout_loss=loss, task_f1=f1, epsilon_spent=0.0, update_sha256=sha)
        self._call("POST", base + "report", rep.model_dump())


def _adapter(b64: str, cfg: FedConfig) -> Adapter:
    from fedlora.sim.data import N_FEATURES
    like = Adapter(np.zeros((cfg.rank, N_FEATURES)), np.zeros((N_CLASSES, cfg.rank)))
    return Adapter.from_flat(arr_d(b64, "<f8"), like)


def _check(res: dict) -> None:
    if "abort" in res:
        raise SecAggError(f"coordinator: {res['abort']}")
