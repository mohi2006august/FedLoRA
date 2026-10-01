"""Private PKI for mTLS: one CA, one coordinator certificate, one certificate per site.

Site certificates carry Ed25519 keys. The same key signs the SecAgg adverts and survivor
lists, so the CA-verified certificate *is* the identity registry (no extra out-of-band step).
Private keys are written with mode 0600 where the platform supports it. Secrets only ever
come from these files / environment (FED_CERT_PATH), never from source.
"""
from __future__ import annotations

import datetime as dt
import ipaddress
import os
from dataclasses import dataclass
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

COORDINATOR_CN = "fedlora-coordinator"


def _name(cn: str) -> x509.Name:
    return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)])


def _build(subject: str, key: Ed25519PrivateKey, issuer_name: x509.Name, issuer_key, *,
           ca: bool = False, usage=None, sans: list | None = None, days: int = 365):
    now = dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=5)
    b = (x509.CertificateBuilder().subject_name(_name(subject)).issuer_name(issuer_name)
         .public_key(key.public_key()).serial_number(x509.random_serial_number())
         .not_valid_before(now).not_valid_after(now + dt.timedelta(days=days))
         .add_extension(x509.BasicConstraints(ca=ca, path_length=None), critical=True))
    if usage:
        b = b.add_extension(x509.ExtendedKeyUsage(usage), critical=False)
    if sans:
        b = b.add_extension(x509.SubjectAlternativeName(sans), critical=False)
    return b.sign(issuer_key, None)  # Ed25519 signs without a separate hash


def _write(path: Path, data: bytes, secret: bool = False) -> None:
    path.write_bytes(data)
    if secret:
        try:
            os.chmod(path, 0o600)
        except OSError:  # e.g. Windows
            pass


def _key_pem(key) -> bytes:
    return key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                             serialization.NoEncryption())


def init_pki(out: Path, site_ids: list[str], hosts: tuple[str, ...] = ("localhost", "127.0.0.1"),
             days: int = 365) -> None:
    out.mkdir(parents=True, exist_ok=True)
    ca_key = Ed25519PrivateKey.generate()
    ca = _build("fedlora-ca", ca_key, _name("fedlora-ca"), ca_key, ca=True, days=days)
    _write(out / "ca.pem", ca.public_bytes(serialization.Encoding.PEM))
    _write(out / "ca.key", _key_pem(ca_key), secret=True)
    sans = []
    for h in hosts:
        try:
            sans.append(x509.IPAddress(ipaddress.ip_address(h)))
        except ValueError:
            sans.append(x509.DNSName(h))
    k = Ed25519PrivateKey.generate()
    c = _build(COORDINATOR_CN, k, ca.subject, ca_key, usage=[ExtendedKeyUsageOID.SERVER_AUTH],
               sans=sans, days=days)
    _write(out / "coordinator.pem", c.public_bytes(serialization.Encoding.PEM))
    _write(out / "coordinator.key", _key_pem(k), secret=True)
    for sid in site_ids:
        k = Ed25519PrivateKey.generate()
        c = _build(sid, k, ca.subject, ca_key, usage=[ExtendedKeyUsageOID.CLIENT_AUTH], days=days)
        _write(out / f"{sid}.pem", c.public_bytes(serialization.Encoding.PEM))
        _write(out / f"{sid}.key", _key_pem(k), secret=True)


def load_cert(pem: bytes) -> x509.Certificate:
    return x509.load_pem_x509_certificate(pem)


def load_key(path: Path) -> Ed25519PrivateKey:
    return serialization.load_pem_private_key(path.read_bytes(), password=None)


def cert_cn(cert: x509.Certificate) -> str:
    return cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value


def verify_site_cert(pem: bytes, ca: x509.Certificate, expect_cn: str) -> bytes:
    """Check issuer, validity and CN; return the raw Ed25519 public key (SecAgg identity)."""
    cert = load_cert(pem)
    cert.verify_directly_issued_by(ca)
    now = dt.datetime.now(dt.timezone.utc)
    if not (cert.not_valid_before_utc <= now <= cert.not_valid_after_utc):
        raise ValueError("certificate not currently valid")
    if cert_cn(cert) != expect_cn:
        raise ValueError(f"certificate CN {cert_cn(cert)!r} != {expect_cn!r}")
    return cert.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


@dataclass
class Bundle:
    """Paths one party needs for mTLS."""
    ca: Path
    cert: Path
    key: Path

    @classmethod
    def for_party(cls, directory: Path, name: str) -> "Bundle":
        return cls(directory / "ca.pem", directory / f"{name}.pem", directory / f"{name}.key")
