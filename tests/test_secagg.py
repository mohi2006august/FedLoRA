import numpy as np
import pytest

from fedlora.secagg.masking import decode, encode
from fedlora.secagg.protocol import (
    PRIME, SecAggClient, SecAggError, SecAggServer, new_identities, prg, secure_sum,
    shamir_combine, shamir_split)


def _xs(n_clients=6, n=40, seed=0):
    rng = np.random.default_rng(seed)
    return {i: rng.normal(0, 0.1, n) for i in range(n_clients)}


def test_shamir_roundtrip_and_threshold():
    secret = 2**255 + 12345
    sh = shamir_split(secret, 4, list(range(1, 9)))
    assert shamir_combine({x: sh[x] for x in (2, 5, 7, 8)}) == secret
    assert shamir_combine({x: sh[x] for x in (1, 2, 3)}) != secret  # t-1 shares learn nothing
    assert all(0 <= v < PRIME for v in sh.values())


def test_fixed_point_roundtrip():
    x = np.array([0.5, -1.25, 3.0])
    assert np.allclose(decode(encode(x)), x, atol=1e-4)


def test_prg_deterministic_and_seed_dependent():
    a, b = prg(b"\x01" * 32, 100), prg(b"\x01" * 32, 100)
    assert (a == b).all() and not (a == prg(b"\x02" * 32, 100)).all()
    assert a.max() < 2**32


def test_sum_without_dropout():
    xs = _xs()
    assert np.allclose(secure_sum(xs, session=1, threshold=4), sum(xs.values()), atol=1e-3)


def test_dropout_before_upload_is_recovered():
    xs = _xs(8)
    got = secure_sum(xs, session=2, threshold=5, drop_before_upload={1, 4})
    assert np.allclose(got, sum(v for i, v in xs.items() if i not in (1, 4)), atol=1e-3)


def test_dropout_before_share_excludes_client_entirely():
    xs = _xs(8)
    got = secure_sum(xs, session=3, threshold=5, drop_before_share={0})
    assert np.allclose(got, sum(v for i, v in xs.items() if i != 0), atol=1e-3)


def test_dropout_before_unmask_still_works_with_threshold_survivors():
    xs = _xs(8)
    got = secure_sum(xs, session=4, threshold=5, drop_before_upload={2}, drop_before_unmask={0, 1})
    assert np.allclose(got, sum(v for i, v in xs.items() if i != 2), atol=1e-3)


def test_too_many_dropouts_abort():
    xs = _xs(8)
    with pytest.raises(SecAggError):
        secure_sum(xs, session=5, threshold=5, drop_before_upload={0, 1, 2, 3})
    with pytest.raises(SecAggError):
        secure_sum(xs, session=6, threshold=5, drop_before_unmask={0, 1, 2, 3})


def _setup(n_clients=5, threshold=3, n=12, session=9):
    ids = list(range(n_clients))
    idk, pubs = new_identities(ids)
    clients = {i: SecAggClient(i, idk[i], pubs, session, threshold) for i in ids}
    server = SecAggServer(session, threshold, n)
    adverts = server.collect_adverts([c.advertise() for c in clients.values()])
    inbox = server.relay_shares({i: c.share_keys(adverts) for i, c in clients.items()})
    for v, cts in inbox.items():
        clients[v].receive_shares(cts)
    return ids, clients, server


def test_masked_upload_reveals_nothing_about_plaintext():
    ids, clients, server = _setup(n=400)
    x = np.full(400, 0.25)
    y = clients[0].masked_input(x, server.u2).astype(np.float64)
    assert abs(np.corrcoef(y, np.arange(400))[0, 1]) < 0.2
    assert abs(y.mean() / 2**32 - 0.5) < 0.1  # uniform over Z_{2^32}


def test_client_refuses_survivor_list_it_did_not_sign():
    ids, clients, server = _setup()
    masked = {i: clients[i].masked_input(np.ones(12), server.u2) for i in ids}
    u3 = server.collect_masked(masked)
    sigs = {i: clients[i].sign_survivors(u3) for i in u3}
    lie = u3[:-1]  # malicious server claims the last client dropped, to learn its pairwise key
    with pytest.raises(SecAggError):
        clients[0].unmask(lie, sigs, server.u2)


def test_client_requires_t_valid_signatures():
    ids, clients, server = _setup(n_clients=5, threshold=4)
    masked = {i: clients[i].masked_input(np.ones(12), server.u2) for i in ids}
    u3 = server.collect_masked(masked)
    sigs = {i: clients[i].sign_survivors(u3) for i in u3}
    few = {i: sigs[i] for i in list(sigs)[:2]}
    with pytest.raises(SecAggError):
        clients[0].unmask(u3, few, server.u2)


def test_forged_advert_is_rejected():
    ids = [0, 1, 2]
    idk, pubs = new_identities(ids)
    clients = {i: SecAggClient(i, idk[i], pubs, 1, 2) for i in ids}
    adverts = [c.advertise() for c in clients.values()]
    adverts[1].s_pk = b"\x00" * 32  # man-in-the-middle swaps a key
    with pytest.raises(SecAggError):
        clients[0].share_keys(adverts)


def test_no_user_has_both_shares_revealed():
    ids, clients, server = _setup()
    masked = {i: clients[i].masked_input(np.ones(12), server.u2) for i in ids if i != 4}
    u3 = server.collect_masked(masked)
    sigs = {i: clients[i].sign_survivors(u3) for i in u3}
    reply = clients[0].unmask(u3, sigs, server.u2)
    assert reply[4][0] == "s" and all(reply[u][0] == "b" for u in u3)
