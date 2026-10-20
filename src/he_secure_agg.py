"""
Homomorphic-encryption secure aggregation (Paillier) for Federated Averaging
============================================================================

Roles
-----
KeyAuthority   Generates the Paillier key pair, publishes the PUBLIC key, keeps
               the PRIVATE key. Only ever decrypts the final aggregate.
Client         Encrypts its (optionally DP-noised) weight vector with the
               public key:        encrypt_update(public_key, vector)
Aggregator     (the FL server) holds ONLY the public key. It combines
               ciphertexts without being able to read any of them:
                                   aggregate_ciphertexts(public_key, ...)

How FedAvg is done on ciphertexts
---------------------------------
Paillier is additively homomorphic:
    E(a) * E(b) mod n^2  =  E(a + b)        (add two encrypted values)
    E(a) ^ k    mod n^2  =  E(k * a)        (multiply by a plaintext integer k)
FedAvg needs  w = sum_k (n_k / N) * w_k.  The aggregator computes
    C = prod_k  E(w_k) ^ n_k   =  E( sum_k n_k * w_k )
and the authority decrypts C and divides by N. Nobody ever decrypts an
individual client's update.

Encoding
--------
Floats -> fixed-point integers:  q = round(w * 2^FRAC_BITS) + BIAS
BIAS makes every value non-negative so that several values can be packed into
one ciphertext ("batching"): SLOT_BITS bits per value, so one 2048-bit
ciphertext carries 31 weights instead of 1. After weighting by n_k and summing
the bias becomes BIAS*N, which the authority subtracts.

Known limitations (state these honestly in the viva)
----------------------------------------------------
* Trust: if the aggregator and the key authority collude, they can decrypt
  individual updates. Fixing this needs threshold Paillier / multi-party keys.
* No proof that a client encrypted a *sane* update (a malicious client could
  encrypt garbage). Range proofs are out of scope.
* Fixed-point precision is 2^-FRAC_BITS (~1.5e-5): decrypted FedAvg differs
  from float FedAvg by at most that rounding error.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np
from phe import paillier

try:  # phe uses gmpy2 automatically when installed (much faster)
    from phe.util import mulmod, powmod
except Exception:  # pragma: no cover
    def powmod(a, b, c):
        return pow(a, b, c)

    def mulmod(a, b, c):
        return (a * b) % c

# ----------------------------------------------------------------------------
# Encoding parameters
# ----------------------------------------------------------------------------
FRAC_BITS = 16                       # fixed-point fractional bits
VALUE_BITS = 32                      # bits for one biased, quantised value
SLOT_BITS = 64                       # bits reserved per value inside a ciphertext
BIAS = 1 << (VALUE_BITS - 1)         # makes quantised values non-negative
SCALE = 1 << FRAC_BITS
MAX_ABS = (BIAS - 1) / SCALE         # |w| must be below ~32768
MAX_TOTAL_SAMPLES = 1 << (SLOT_BITS - VALUE_BITS - 1)   # headroom for weighting


def slots_per_ciphertext(public_key) -> int:
    """How many weights fit in one ciphertext (plaintext must stay < n/3)."""
    return max(1, (public_key.n.bit_length() - 2) // SLOT_BITS)


# ----------------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------------
@dataclass
class EncryptedUpdate:
    """What a client sends to the aggregator."""
    ciphertexts: List[int]
    n_params: int
    slots: int
    clipped: int = 0                          # values clipped to +-MAX_ABS
    encrypt_seconds: float = 0.0
    key_bits: int = 0

    @property
    def nbytes(self) -> int:
        # a Paillier ciphertext lives mod n^2 -> 2 * key_bits bits
        return len(self.ciphertexts) * (2 * self.key_bits // 8)


@dataclass
class AggregateTimings:
    aggregate_seconds: float = 0.0
    decrypt_seconds: float = 0.0
    extras: Dict[str, float] = field(default_factory=dict)


# ----------------------------------------------------------------------------
# Client side
# ----------------------------------------------------------------------------
def _quantise(vec: np.ndarray):
    vec = np.asarray(vec, dtype=np.float64).ravel()
    clipped = int(np.sum(np.abs(vec) > MAX_ABS))
    vec = np.clip(vec, -MAX_ABS, MAX_ABS)
    q = np.rint(vec * SCALE).astype(np.int64) + BIAS
    assert q.min() >= 0 and q.max() < (1 << VALUE_BITS)
    return q, clipped


def encrypt_update(public_key, vector: np.ndarray, slots: int | None = None) -> EncryptedUpdate:
    """Encrypt a flat weight vector. ``slots=1`` gives the naive 1-value-per-ciphertext mode."""
    start = time.perf_counter()
    slots = slots or slots_per_ciphertext(public_key)
    if slots > slots_per_ciphertext(public_key):
        raise ValueError("too many slots for this key size")
    q, clipped = _quantise(vector)
    n_params = len(q)

    cts: List[int] = []
    for i in range(0, n_params, slots):
        chunk = q[i:i + slots]
        packed = 0
        for j, v in enumerate(chunk):
            packed |= int(v) << (SLOT_BITS * j)
        cts.append(int(public_key.raw_encrypt(packed)))

    return EncryptedUpdate(
        ciphertexts=cts, n_params=n_params, slots=slots, clipped=clipped,
        encrypt_seconds=time.perf_counter() - start,
        key_bits=public_key.n.bit_length(),
    )


# ----------------------------------------------------------------------------
# Aggregator side (public key only!)
# ----------------------------------------------------------------------------
def aggregate_ciphertexts(public_key, updates: Sequence[EncryptedUpdate],
                          sample_counts: Sequence[int]):
    """
    Compute  prod_k  C_k ^ n_k  (mod n^2)  =  E( sum_k n_k * packed_k ).
    Uses only the public key; the aggregator cannot read any update.
    Returns (aggregate_ciphertexts, total_samples, seconds).
    """
    start = time.perf_counter()
    if len(updates) != len(sample_counts) or not updates:
        raise ValueError("need one sample count per update")
    total = int(sum(sample_counts))
    if total >= MAX_TOTAL_SAMPLES:
        raise ValueError("total sample count too large for slot headroom")
    n_ct = len(updates[0].ciphertexts)
    if any(len(u.ciphertexts) != n_ct or u.n_params != updates[0].n_params
           or u.slots != updates[0].slots for u in updates):
        raise ValueError("clients sent incompatible updates")

    nsq = public_key.nsquare
    out: List[int] = []
    for idx in range(n_ct):
        acc = 1                                   # E(0) up to randomness
        for upd, n_k in zip(updates, sample_counts):
            acc = mulmod(acc, powmod(upd.ciphertexts[idx], int(n_k), nsq), nsq)
        out.append(int(acc))
    return out, total, time.perf_counter() - start


# ----------------------------------------------------------------------------
# Key authority
# ----------------------------------------------------------------------------
class KeyAuthority:
    """Holds the private key; decrypts ONLY aggregates."""

    def __init__(self, key_bits: int = 2048):
        t = time.perf_counter()
        self.public_key, self._private_key = paillier.generate_paillier_keypair(n_length=key_bits)
        self.keygen_seconds = time.perf_counter() - t
        self.key_bits = key_bits

    def decrypt_average(self, agg_ciphertexts: Sequence[int], total_samples: int,
                        n_params: int, slots: int) -> tuple[np.ndarray, float]:
        """Decrypt the weighted sum and return the FedAvg vector (float64)."""
        start = time.perf_counter()
        mask = (1 << SLOT_BITS) - 1
        vals: List[int] = []
        for ct in agg_ciphertexts:
            packed = int(self._private_key.raw_decrypt(ct))
            for j in range(slots):
                vals.append((packed >> (SLOT_BITS * j)) & mask)
        vals = vals[:n_params]
        arr = np.array([v - BIAS * total_samples for v in vals], dtype=np.float64)
        return arr / (total_samples * SCALE), time.perf_counter() - start


# ----------------------------------------------------------------------------
# Helpers to move between the project's weight dicts and flat vectors
# ----------------------------------------------------------------------------
WEIGHT_KEYS = ("W1", "b1", "W2", "b2")


def flatten_weights(weights: Dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([weights[k].ravel() for k in WEIGHT_KEYS])


def unflatten_weights(vec: np.ndarray, template: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out, pos = {}, 0
    for k in WEIGHT_KEYS:
        size = template[k].size
        out[k] = np.asarray(vec[pos:pos + size]).reshape(template[k].shape).copy()
        pos += size
    return out
