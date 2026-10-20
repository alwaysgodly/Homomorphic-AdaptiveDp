"""
Real (measured) cost of Paillier secure aggregation for our 4,810-parameter model.
Run:  python demos/he_benchmark.py            (~1.5 min)
Writes results/he_benchmark.json
"""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import he_secure_agg as he

N_PARAMS = 4810
COUNTS = [274, 297, 158, 417, 291]          # client sizes from the real run
RESULTS = os.path.join(HERE, "..", "results")


def bench(key_bits, slots_mode):
    auth = he.KeyAuthority(key_bits)
    pk = auth.public_key
    slots = 1 if slots_mode == "naive" else None
    rng = np.random.default_rng(0)
    vecs = [rng.normal(0, 0.1, N_PARAMS) for _ in COUNTS]

    ups = [he.encrypt_update(pk, v, slots=slots) for v in vecs]
    agg, total, t_agg = he.aggregate_ciphertexts(pk, ups, COUNTS)
    out, t_dec = auth.decrypt_average(agg, total, N_PARAMS, ups[0].slots)

    expected = sum(c * v for c, v in zip(COUNTS, vecs)) / total
    return {
        "key_bits": key_bits, "mode": slots_mode,
        "slots_per_ciphertext": ups[0].slots,
        "ciphertexts_per_client": len(ups[0].ciphertexts),
        "encrypt_s_per_client": float(np.mean([u.encrypt_seconds for u in ups])),
        "aggregate_s_server": t_agg, "decrypt_s_authority": t_dec,
        "upload_MB_per_client": ups[0].nbytes / 1e6,
        "plaintext_MB_per_client": N_PARAMS * 8 / 1e6,
        "max_abs_error_vs_float_fedavg": float(np.max(np.abs(out - expected))),
    }


if __name__ == "__main__":
    rows = []
    print(f"{'key':>5} {'mode':>7} {'cts':>5} {'enc/client':>11} {'aggregate':>10} {'decrypt':>9} {'upload MB':>10} {'max err':>9}")
    for bits in (1024, 2048):
        for mode in ("naive", "packed"):
            r = bench(bits, mode); rows.append(r)
            print(f"{bits:>5} {mode:>7} {r['ciphertexts_per_client']:>5} {r['encrypt_s_per_client']:>10.2f}s "
                  f"{r['aggregate_s_server']:>9.2f}s {r['decrypt_s_authority']:>8.2f}s "
                  f"{r['upload_MB_per_client']:>10.3f} {r['max_abs_error_vs_float_fedavg']:>9.1e}", flush=True)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "he_benchmark.json"), "w") as f:
        json.dump({"rows": rows}, f, indent=2)
    print("\nplaintext update size:", rows[0]["plaintext_MB_per_client"], "MB per client")
