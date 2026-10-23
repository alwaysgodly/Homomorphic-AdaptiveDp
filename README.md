# Homomorphic Encryption Secure Aggregation for Federated Learning

Part of the project **Privacy Preserving in FL using Homomorphic Encryption and Differential Privacy (adaptive budgets)**.
This document covers the **homomorphic encryption (HE)** half: how it works, what was built, measured results, and how to run it.

> When you create the new repo, rename this file to `README.md` (or merge it in). The old `README.md` in this folder belongs to the baseline FL project.

---

## 1. What it does

In normal federated learning (FedAvg) the server receives every client's model weights in the clear. Those weights can leak training data (gradient inversion attacks).

With this module:

1. Each client **encrypts** its model update before sending it.
2. The server **averages the encrypted updates without decrypting them**.
3. Only the final **aggregate** is decrypted. No individual client's update is ever readable by the server.
4. The decrypted result matches plain FedAvg (max difference 7.6e-6, which is fixed-point rounding).

## 2. How it works

**Roles**

| Role | Holds | Does |
|---|---|---|
| Key authority | private key | generates the Paillier key pair, publishes the public key, decrypts **only the aggregate** |
| Clients | public key | train locally, optionally apply the DP hook, encrypt the update |
| Aggregator (FL server) | public key only | combines ciphertexts, cannot read any of them |

**The math.** Paillier encryption is additively homomorphic:

- `E(a) * E(b) mod n^2 = E(a + b)` (multiplying ciphertexts adds the hidden values)
- `E(a)^k mod n^2 = E(k * a)` (raising to a plain integer multiplies the hidden value)

FedAvg needs `w = sum_k (n_k / N) * w_k`. The server computes
`C = prod_k E(w_k)^(n_k) = E(sum_k n_k * w_k)`, and the authority decrypts `C` and divides by `N`.

**Encoding.** Floats become fixed-point integers: `q = round(w * 2^16) + 2^31`. The bias makes every value non-negative so several can share one ciphertext.

**Packing (batching).** One 2048-bit ciphertext carries 31 weights (64 bits per slot) instead of 1. For our 4,810-parameter model that is 156 ciphertexts per client instead of 4,810.

## 3. Project layout

```
src/
  fl_server.py        baseline FedAvg server (plain; used as the correctness check)
  fl_client.py        client: local training of a small neural network
  data_utils.py       dataset loading and IID / non-IID splitting
  he_secure_agg.py    NEW: Paillier secure aggregation (KeyAuthority, encrypt_update, aggregate_ciphertexts)
  dp_hooks.py         NEW: DP hook interface (placeholder; the DP sub-team replaces it)
demos/
  he_demo.py          NEW: end-to-end terminal demo
  he_benchmark.py     NEW: naive vs packed timings
  dashboard_server.py NEW: live dashboard server
  dashboard.html      NEW: dashboard page
tests/
  test_he_secure_agg.py   NEW: 9 tests
results/              plots, JSON and logs produced by the demos
```

## 4. Setup

Python 3.10 or newer is required.

```
pip install -r requirements.txt
```

`gmpy2` is optional and makes Paillier much faster. If it fails to install (common on Windows), delete the `gmpy2` line from `requirements.txt`. Everything still works, only slower.

## 5. How to run

Run everything from the project root.

**Step 1: run the tests**

```
python tests/test_he_secure_agg.py
```

Expected: `9/9 tests passed`.

**Step 2: run the live dashboard (best for presenting)**

```
python demos/dashboard_server.py
```

Open http://localhost:8000, pick a key size and number of rounds, and press **Start training**. Use the **1024-bit** key for a live demo (under a minute). The **2048-bit** key takes about 3 minutes. To use another port: `python demos/dashboard_server.py 8765`.

The dashboard shows the four pipeline steps, per-client cards, encrypted vs plain accuracy, the ciphertext the server actually receives, a privacy-budget panel, and a per-round table. All values are streamed from a real run, so nothing on it is hardcoded.

**Step 3: run the terminal demo**

```
python demos/he_demo.py                     # 5 clients, 10 rounds, 2048-bit key
python demos/he_demo.py --key-bits 1024     # faster
python demos/he_demo.py --rounds 3          # fewer rounds
python demos/he_demo.py --iid               # IID data split
python demos/he_demo.py --dp-noise 0.01     # turn on the DP placeholder hook
python demos/he_demo.py --no-plot           # skip the PNG
```

Each round it prints the encrypted-aggregation accuracy next to plain FedAvg, the maximum weight difference between them, timings, and a sample of what the server actually sees (ciphertext hex; encrypting the same data twice gives a different ciphertext).
Outputs are saved to `results/he_demo_results.json` and `results/he_demo_results.png`.

**Step 4: run the benchmark**

```
python demos/he_benchmark.py
```

Takes about 1.5 minutes and writes `results/he_benchmark.json`.

## 6. Measured results

Five clients, 4,810-parameter model, run on the build machine. Your timings will differ with hardware and `gmpy2`.

| Measure | Naive (1 weight per ciphertext) | Packed (31 weights per ciphertext) |
|---|---|---|
| Encryption per client, 2048-bit | about 60 s | about 1.9 s |
| Ciphertexts per client | 4,810 | 156 |
| Upload per client per round | about 2.46 MB | about 0.08 MB |

- Packing gives about 33x faster encryption and about 31x smaller uploads.
- Encrypted accuracy equals plain FedAvg accuracy (74.2% after 10 rounds in the latest run).
- Per round: about 0.06 s server aggregation and about 0.6 s decryption at 2048-bit.
- A packed encrypted upload is about 2.1x the size of a plain update (0.080 MB vs 0.038 MB).

## 7. Plugging in differential privacy

The DP sub-team replaces `ClipAndNoise` in `src/dp_hooks.py`. The pipeline calls it on every client before encryption:

```python
hook(client_id, global_vec, local_vec) -> flat_vector
```

Optionally, add this method to feed the dashboard's privacy-budget panel after every round:

```python
budget_report() -> {client_id: {"epsilon_total": float, "epsilon_used": float}}
```

The bundled `ClipAndNoise` is only a placeholder. It clips and adds Gaussian noise but makes **no epsilon guarantee**.

## 8. API summary (`src/he_secure_agg.py`)

| Function / class | Purpose |
|---|---|
| `KeyAuthority(key_bits)` | makes the key pair; `.public_key`; `.decrypt_average(...)` |
| `encrypt_update(public_key, vector, slots=None)` | client side; returns an `EncryptedUpdate` (`slots=1` is the naive mode) |
| `aggregate_ciphertexts(public_key, updates, sample_counts)` | server side; weighted product of ciphertexts |
| `flatten_weights` / `unflatten_weights` | convert between weight dictionaries and flat vectors |

## 9. Limitations (state these honestly)

- **Trust:** if the aggregator and the key authority collude, they can decrypt individual updates. Fixing this needs threshold or multi-party Paillier.
- **No range proofs:** nothing proves a client encrypted a sane update.
- **Simulation:** clients, server and authority run in one process, not on a real network.
- **Precision:** fixed-point step is about 1.5e-5, and values must stay below about 32,768 in magnitude (larger values are clipped and counted).
- **Key size:** 1024-bit is for demos only; use 2048-bit or larger for anything realistic.
- The non-IID split is unseeded, so accuracy varies between runs.

## 10. Troubleshooting

- **`ModuleNotFoundError: phe`**: run `pip install -r requirements.txt`.
- **`gmpy2` fails to install**: remove it from `requirements.txt`; it is optional.
- **Port already in use**: pass another port, e.g. `python demos/dashboard_server.py 8765`.
- **Dashboard says a run is already in progress**: wait for the current run to finish.
- **2048-bit run feels slow**: switch to the 1024-bit key for live demos.
