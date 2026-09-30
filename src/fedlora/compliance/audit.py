"""Signed, hash-chained audit log of every update (NFR: updates signed and audited).

Signatures are HMAC-SHA256 with a per-site key taken from the environment
(FED_SIGNING_KEY_<SITE>) or derived deterministically in simulation only.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
from dataclasses import dataclass, field

import numpy as np


def update_sha256(delta: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(delta, dtype=np.float64).tobytes()).hexdigest()


def site_key(site_id: str) -> bytes:
    env = os.environ.get("FED_SIGNING_KEY_" + site_id.upper().replace("-", "_"))
    if env:
        return env.encode()
    return hashlib.sha256(f"sim-only:{site_id}".encode()).digest()  # simulation only


def sign(site_id: str, round_no: int, sha: str) -> str:
    msg = f"{site_id}|{round_no}|{sha}".encode()
    return hmac.new(site_key(site_id), msg, hashlib.sha256).hexdigest()


def verify(site_id: str, round_no: int, sha: str, sig: str) -> bool:
    return hmac.compare_digest(sign(site_id, round_no, sha), sig)


@dataclass
class AuditLog:
    entries: list[dict] = field(default_factory=list)

    def record(self, site_id: str, round_no: int, sha: str, event: str = "update") -> dict:
        prev = self.entries[-1]["chain"] if self.entries else "0" * 64
        e = {"site": site_id, "round": round_no, "event": event, "sha256": sha,
             "sig": sign(site_id, round_no, sha)}
        e["chain"] = hashlib.sha256((prev + json.dumps(e, sort_keys=True)).encode()).hexdigest()
        self.entries.append(e)
        return e

    def verify_chain(self) -> bool:
        prev = "0" * 64
        for e in self.entries:
            body = {k: v for k, v in e.items() if k != "chain"}
            if hashlib.sha256((prev + json.dumps(body, sort_keys=True)).encode()).hexdigest() != e["chain"]:
                return False
            if e["event"] == "update" and not verify(e["site"], e["round"], e["sha256"], e["sig"]):
                return False
            prev = e["chain"]
        return True

    def export(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for e in self.entries:
                f.write(json.dumps(e) + "\n")
