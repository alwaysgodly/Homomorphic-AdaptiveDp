"""Run with:  python tests/test_he_secure_agg.py   (or pytest)"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import he_secure_agg as he

AUTH = he.KeyAuthority(key_bits=1024)   # small key => fast tests (NOT for real use)
PK = AUTH.public_key


def _fedavg_via_he(vectors, counts, slots=None):
    ups = [he.encrypt_update(PK, v, slots=slots) for v in vectors]
    agg, total, _ = he.aggregate_ciphertexts(PK, ups, counts)
    out, _ = AUTH.decrypt_average(agg, total, ups[0].n_params, ups[0].slots)
    return out


def test_weighted_fedavg_matches_numpy():
    rng = np.random.default_rng(1)
    vs = [rng.normal(0, 0.5, 100) for _ in range(5)]
    counts = [274, 297, 158, 417, 291]
    expected = sum(c * v for c, v in zip(counts, vs)) / sum(counts)
    got = _fedavg_via_he(vs, counts)
    assert np.max(np.abs(got - expected)) < 2 * 2 ** -he.FRAC_BITS, np.max(np.abs(got - expected))


def test_negative_and_zero_values():
    v = np.array([-3.25, 0.0, 2.5, -0.00001, 1e-5])
    got = _fedavg_via_he([v, v], [10, 30])
    assert np.allclose(got, v, atol=2 ** -he.FRAC_BITS)


def test_naive_equals_packed():
    rng = np.random.default_rng(2)
    vs = [rng.normal(0, 1, 40) for _ in range(3)]
    counts = [5, 7, 11]
    assert np.allclose(_fedavg_via_he(vs, counts, slots=1), _fedavg_via_he(vs, counts), atol=1e-9)


def test_packing_uses_expected_slots_and_length_not_multiple_of_slots():
    s = he.slots_per_ciphertext(PK)
    assert s == 15  # (1024-2)//64
    v = np.linspace(-1, 1, s * 3 + 4)   # ragged last ciphertext
    got = _fedavg_via_he([v], [1])
    assert len(got) == len(v) and np.allclose(got, v, atol=2 ** -he.FRAC_BITS)


def test_encryption_is_randomised_and_server_cannot_read():
    v = np.array([0.5] * 10)
    a = he.encrypt_update(PK, v).ciphertexts
    b = he.encrypt_update(PK, v).ciphertexts
    assert a != b                      # same plaintext, different ciphertext
    assert all(c > PK.n for c in a)    # looks like a big random number mod n^2, not the plaintext


def test_additive_homomorphism_raw():
    c1, c2 = PK.raw_encrypt(1234), PK.raw_encrypt(4321)
    s = (c1 * c2) % PK.nsquare
    assert AUTH._private_key.raw_decrypt(s) == 1234 + 4321
    assert AUTH._private_key.raw_decrypt(pow(c1, 7, PK.nsquare)) == 7 * 1234


def test_clipping_reported():
    v = np.array([1e6, -1e6, 0.1])
    up = he.encrypt_update(PK, v)
    assert up.clipped == 2


def test_flatten_roundtrip():
    w = {"W1": np.random.randn(64, 64), "b1": np.random.randn(1, 64),
         "W2": np.random.randn(64, 10), "b2": np.random.randn(1, 10)}
    flat = he.flatten_weights(w)
    assert flat.size == 4810
    back = he.unflatten_weights(flat, w)
    assert all(np.array_equal(w[k], back[k]) for k in w)


def test_incompatible_updates_rejected():
    a = he.encrypt_update(PK, np.zeros(20)); b = he.encrypt_update(PK, np.zeros(40))
    try:
        he.aggregate_ciphertexts(PK, [a, b], [1, 1]); assert False
    except ValueError:
        pass


if __name__ == "__main__":
    fns = [f for n, f in sorted(globals().items()) if n.startswith("test_")]
    for f in fns:
        f(); print("PASS", f.__name__)
    print(f"\n{len(fns)}/{len(fns)} tests passed")
