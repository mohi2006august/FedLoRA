"""Secure aggregation with double masking, key agreement and secret sharing (FR-03).

Follows Bonawitz et al. 2017 (semi-honest server core plus the signed-survivor-list
consistency check). Rounds:

 0 AdvertiseKeys   each client sends two X25519 public keys (c: share encryption, s: mask
                   agreement), signed with its long-term Ed25519 identity key.
 1 ShareKeys       each client draws a self-mask seed b_u, Shamir-shares b_u and its s secret
                   key among all participants (threshold t), encrypting each share (AES-GCM)
                   to its recipient under the c-key agreement. The server only relays.
 2 MaskedInput     y_u = x_u + PRG(b_u) + sum_{v>u} PRG(s_uv) - sum_{v<u} PRG(s_uv)  (mod 2^32)
 3 Consistency     survivors sign the list U3 of users whose y reached the server.
 4 Unmasking       a client that sees >= t valid signatures on the same U3 returns, for every
                   dropped v, its share of v's s key, and for every survivor u, its share of
                   b_u - never both for one user. The server rebuilds and removes the masks.

Residual limits: PRG is AES-CTR keyed by the seed (sound); fixed-point encoding mod 2^32 is
shared with the rest of the code base; the server is assumed not to collude with >= t clients;
identity keys are distributed out of band (the `identity_pubs` registry).
"""
from __future__ import annotations

import json
import os
import secrets
from dataclasses import dataclass, field

import numpy as np
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey, Ed25519PublicKey)
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from fedlora.secagg.masking import MASK, decode, encode

PRIME = 2**521 - 1  # Mersenne prime > 2^256 (secrets are 32-byte strings)
_RAW = dict(encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)


class SecAggError(RuntimeError):
    """Protocol abort (too few survivors, bad signature, inconsistent survivor lists...)."""


# ---- Shamir secret sharing over GF(PRIME) ---------------------------------------------------
def shamir_split(secret: int, threshold: int, xs: list[int]) -> dict[int, int]:
    if not 0 <= secret < PRIME:
        raise ValueError("secret out of field range")
    coeffs = [secret] + [secrets.randbelow(PRIME) for _ in range(threshold - 1)]
    out = {}
    for x in xs:
        y = 0
        for c in reversed(coeffs):  # Horner
            y = (y * x + c) % PRIME
        out[x] = y
    return out


def shamir_combine(points: dict[int, int]) -> int:
    total = 0
    for xi, yi in points.items():
        num = den = 1
        for xj in points:
            if xj != xi:
                num = num * (-xj) % PRIME
                den = den * (xi - xj) % PRIME
        total = (total + yi * num * pow(den, -1, PRIME)) % PRIME
    return total


# ---- primitives ------------------------------------------------------------------------------
def prg(seed: bytes, n: int) -> np.ndarray:
    """n uint64 values in [0, 2^32) from AES-256-CTR keystream."""
    enc = Cipher(algorithms.AES(seed), modes.CTR(b"\x00" * 16)).encryptor()
    stream = enc.update(b"\x00" * (4 * n))
    return np.frombuffer(stream, dtype="<u4").astype(np.uint64)


def _kdf(secret: bytes, info: bytes) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=info).derive(secret)


def _signed_advert(session: int, uid: int, c_pk: bytes, s_pk: bytes) -> bytes:
    return b"advert|%d|%d|" % (session, uid) + c_pk + s_pk


def _survivor_msg(session: int, u3: list[int]) -> bytes:
    return b"survivors|%d|" % session + json.dumps(sorted(u3)).encode()


def _int_bytes(x: int) -> bytes:
    return x.to_bytes(66, "big")


@dataclass
class Advert:
    uid: int
    c_pk: bytes
    s_pk: bytes
    sig: bytes


class SecAggClient:
    def __init__(self, uid: int, identity: Ed25519PrivateKey, identity_pubs: dict[int, bytes],
                 session: int, threshold: int):
        self.uid, self.session, self.t = uid, session, threshold
        self._identity = identity
        self._pubs = identity_pubs
        self._c_sk = X25519PrivateKey.generate()
        self._s_sk_bytes = os.urandom(32)
        self._s_sk = X25519PrivateKey.from_private_bytes(self._s_sk_bytes)
        self._b = os.urandom(32)
        self._peers: dict[int, Advert] = {}
        self._held: dict[int, dict] = {}  # sender -> {"b": share, "s": share}
        self._signed_u3: list[int] | None = None

    # round 0
    def advertise(self) -> Advert:
        c_pk = self._c_sk.public_key().public_bytes(**_RAW)
        s_pk = self._s_sk.public_key().public_bytes(**_RAW)
        sig = self._identity.sign(_signed_advert(self.session, self.uid, c_pk, s_pk))
        return Advert(self.uid, c_pk, s_pk, sig)

    # round 1
    def share_keys(self, adverts: list[Advert]) -> dict[int, bytes]:
        if len(adverts) < self.t:
            raise SecAggError("too few participants advertised")
        for a in adverts:
            try:
                Ed25519PublicKey.from_public_bytes(self._pubs[a.uid]).verify(
                    a.sig, _signed_advert(self.session, a.uid, a.c_pk, a.s_pk))
            except (InvalidSignature, KeyError) as e:
                raise SecAggError(f"bad advert from {a.uid}") from e
        self._peers = {a.uid: a for a in adverts}
        xs = sorted(self._peers)
        b_sh = shamir_split(int.from_bytes(self._b, "big"), self.t, [x + 1 for x in xs])
        s_sh = shamir_split(int.from_bytes(self._s_sk_bytes, "big"), self.t, [x + 1 for x in xs])
        out = {}
        for v in xs:
            key = _kdf(self._c_sk.exchange(X25519PublicKey.from_public_bytes(self._peers[v].c_pk)),
                       b"fedlora-secagg-share")
            nonce = os.urandom(12)
            body = _int_bytes(b_sh[v + 1]) + _int_bytes(s_sh[v + 1])
            ad = b"%d|%d|%d" % (self.session, self.uid, v)
            out[v] = nonce + AESGCM(key).encrypt(nonce, body, ad)
        return out

    def receive_shares(self, cts: dict[int, bytes]) -> None:
        for u, ct in cts.items():
            key = _kdf(self._c_sk.exchange(X25519PublicKey.from_public_bytes(self._peers[u].c_pk)),
                       b"fedlora-secagg-share")
            body = AESGCM(key).decrypt(ct[:12], ct[12:], b"%d|%d|%d" % (self.session, u, self.uid))
            self._held[u] = {"b": int.from_bytes(body[:66], "big"),
                             "s": int.from_bytes(body[66:], "big")}

    # round 2
    def _pair_seed(self, v: int) -> bytes:
        shared = self._s_sk.exchange(X25519PublicKey.from_public_bytes(self._peers[v].s_pk))
        a, b = sorted((self.uid, v))
        return _kdf(shared, b"fedlora-secagg-mask|%d|%d|%d" % (self.session, a, b))

    def masked_input(self, x: np.ndarray, u2: list[int]) -> np.ndarray:
        y = encode(x)
        y = (y + prg(self._b, len(x))) & np.uint64(MASK)
        for v in u2:
            if v == self.uid:
                continue
            m = prg(self._pair_seed(v), len(x))
            y = (y + m) & np.uint64(MASK) if self.uid < v else (y - m) & np.uint64(MASK)
        return y

    # round 3
    def sign_survivors(self, u3: list[int]) -> bytes:
        self._signed_u3 = sorted(u3)
        return self._identity.sign(_survivor_msg(self.session, u3))

    # round 4
    def unmask(self, u3: list[int], sigs: dict[int, bytes], u2: list[int]) -> dict[int, tuple]:
        if self._signed_u3 != sorted(u3):
            raise SecAggError("server announced a survivor list this client did not sign")
        good = 0
        for w, sig in sigs.items():
            try:
                Ed25519PublicKey.from_public_bytes(self._pubs[w]).verify(
                    sig, _survivor_msg(self.session, u3))
                good += 1
            except (InvalidSignature, KeyError):
                continue
        if good < self.t:
            raise SecAggError("fewer than t valid survivor signatures")
        out = {}
        for v in u2:
            if v not in self._held:
                continue
            # never reveal both shares of one user: b for survivors, s for dropped
            out[v] = ("b", self._held[v]["b"]) if v in u3 else ("s", self._held[v]["s"])
        return out


@dataclass
class SecAggServer:
    session: int
    threshold: int
    n_values: int
    u2: list[int] = field(default_factory=list)
    adverts: dict[int, Advert] = field(default_factory=dict)
    masked: dict[int, np.ndarray] = field(default_factory=dict)

    def collect_adverts(self, adverts: list[Advert]) -> list[Advert]:
        if len(adverts) < self.threshold:
            raise SecAggError("too few participants advertised")
        self.adverts = {a.uid: a for a in adverts}
        return list(adverts)

    def relay_shares(self, sent: dict[int, dict[int, bytes]]) -> dict[int, dict[int, bytes]]:
        """sent[u][v] = ciphertext u->v; returns inbox[v][u]. Users who sent nothing drop out."""
        if len(sent) < self.threshold:
            raise SecAggError("too few participants completed share distribution")
        self.u2 = sorted(sent)
        inbox: dict[int, dict[int, bytes]] = {v: {} for v in self.u2}
        for u, per in sent.items():
            for v, ct in per.items():
                if v in inbox:
                    inbox[v][u] = ct
        return inbox

    def collect_masked(self, masked: dict[int, np.ndarray]) -> list[int]:
        if len(masked) < self.threshold:
            raise SecAggError("too few masked inputs")
        self.masked = masked
        return sorted(masked)

    def finish(self, replies: dict[int, dict[int, tuple]]) -> np.ndarray:
        """Combine unmasking replies into the plain sum over survivors (float vector)."""
        if len(replies) < self.threshold:
            raise SecAggError("too few unmasking replies")
        u3 = sorted(self.masked)
        dropped = [v for v in self.u2 if v not in u3]
        b_pts: dict[int, dict[int, int]] = {}
        s_pts: dict[int, dict[int, int]] = {}
        for w, per in replies.items():
            for v, (kind, share) in per.items():
                (b_pts if kind == "b" else s_pts).setdefault(v, {})[w + 1] = share
        total = np.zeros(self.n_values, dtype=np.uint64)
        for y in self.masked.values():
            total = (total + y) & np.uint64(MASK)
        for u in u3:  # remove self-masks
            if len(b_pts.get(u, {})) < self.threshold:
                raise SecAggError(f"cannot reconstruct self-mask of {u}")
            seed = shamir_combine(dict(list(b_pts[u].items())[: self.threshold])).to_bytes(32, "big")
            total = (total - prg(seed, self.n_values)) & np.uint64(MASK)
        for v in dropped:  # remove orphaned pairwise masks
            if len(s_pts.get(v, {})) < self.threshold:
                raise SecAggError(f"cannot reconstruct key of dropped user {v}")
            sk_bytes = shamir_combine(dict(list(s_pts[v].items())[: self.threshold])).to_bytes(32, "big")
            sk = X25519PrivateKey.from_private_bytes(sk_bytes)
            for u in u3:
                shared = sk.exchange(X25519PublicKey.from_public_bytes(self.adverts[u].s_pk))
                a, b = sorted((u, v))
                seed = _kdf(shared, b"fedlora-secagg-mask|%d|%d|%d" % (self.session, a, b))
                m = prg(seed, self.n_values)
                # survivor u added +m if u < v else -m; undo it
                total = (total - m) & np.uint64(MASK) if u < v else (total + m) & np.uint64(MASK)
        return decode(total)


def new_identities(ids: list[int]) -> tuple[dict[int, Ed25519PrivateKey], dict[int, bytes]]:
    keys = {i: Ed25519PrivateKey.generate() for i in ids}
    return keys, {i: k.public_key().public_bytes(**_RAW) for i, k in keys.items()}


def secure_sum(xs: dict[int, np.ndarray], *, session: int, threshold: int,
               identities: dict[int, Ed25519PrivateKey] | None = None,
               drop_before_share: set[int] = frozenset(), drop_before_upload: set[int] = frozenset(),
               drop_before_unmask: set[int] = frozenset()) -> np.ndarray:
    """Run all five rounds in-process. `drop_*` simulate a client vanishing before that round.

    Returns the sum of the vectors of clients whose masked input reached the server.
    """
    ids = sorted(xs)
    n = len(next(iter(xs.values())))
    if identities is None:
        identities, pubs = new_identities(ids)
    else:
        pubs = {i: k.public_key().public_bytes(**_RAW) for i, k in identities.items()}
    clients = {i: SecAggClient(i, identities[i], pubs, session, threshold) for i in ids}
    server = SecAggServer(session, threshold, n)
    adverts = server.collect_adverts([c.advertise() for c in clients.values()])
    sent = {i: c.share_keys(adverts) for i, c in clients.items() if i not in drop_before_share}
    inbox = server.relay_shares(sent)
    for v, cts in inbox.items():
        clients[v].receive_shares(cts)
    masked = {i: clients[i].masked_input(xs[i], server.u2)
              for i in server.u2 if i not in drop_before_upload}
    u3 = server.collect_masked(masked)
    sigs = {i: clients[i].sign_survivors(u3) for i in u3 if i not in drop_before_unmask}
    replies = {i: clients[i].unmask(u3, sigs, server.u2) for i in sigs}
    return server.finish(replies)
