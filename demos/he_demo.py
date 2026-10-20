"""
End-to-end federated learning with Paillier homomorphic-encryption secure aggregation.

  clients  : train locally (existing FederatedClient) -> [DP hook] -> encrypt with PUBLIC key
  server   : holds only the public key, multiplies ciphertexts (FedAvg on encrypted data)
  authority: holds the private key, decrypts ONLY the aggregate

Every round we also compute plain FedAvg with the project's own server code on the very same
client updates and report the difference, so correctness is checked, not assumed.

Usage:
  python demos/he_demo.py                      # 5 clients, 10 rounds, 2048-bit key (~3 min)
  python demos/he_demo.py --key-bits 1024      # faster, demo-grade key
  python demos/he_demo.py --rounds 3           # quick check
  python demos/he_demo.py --dp-noise 0.01      # plug in the placeholder DP hook
"""
import argparse, contextlib, io, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from fl_server import FederatedServer
from fl_clients import FederatedClient
from data_uitls import load_digits_dataset, preprocess_data, partition_data_federated
import he_secure_agg as he
from dp_hooks import NoPrivacy, ClipAndNoise

EVENTS_PATH = os.path.join(HERE, "..", "results", "live_events.jsonl")


def emit(type_, **kw):
    """Append one event for the live dashboard (the dashboard only ever displays these)."""
    rec = {"t": time.time(), "type": type_}
    rec.update(kw)
    with open(EVENTS_PATH, "a") as f:
        f.write(json.dumps(rec) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--key-bits", type=int, default=2048)
    ap.add_argument("--iid", action="store_true")
    ap.add_argument("--dp-noise", type=float, default=0.0, help="placeholder DP hook: Gaussian std (0 = off)")
    ap.add_argument("--dp-clip", type=float, default=1.0)
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(EVENTS_PATH), exist_ok=True)
    open(EVENTS_PATH, "w").close()          # fresh event log for this run
    emit("start", clients=a.clients, rounds=a.rounds, key_bits=a.key_bits, iid=a.iid, dp_noise=a.dp_noise)

    print("=" * 78)
    print("FEDERATED LEARNING + PAILLIER HOMOMORPHIC-ENCRYPTION SECURE AGGREGATION")
    print("=" * 78)

    # ---- data + parties -------------------------------------------------------
    X, y = load_digits_dataset()
    Xp, yp = preprocess_data(X, y, n_classes=10)
    data = partition_data_federated(Xp, yp, n_clients=a.clients, iid=a.iid)
    server = FederatedServer({"input_size": 64, "hidden_size": 64, "output_size": 10}, security_enabled=False)
    with contextlib.redirect_stdout(io.StringIO()):
        clients = [FederatedClient(cid, d["X_train"], d["y_train"], data["X_test"], data["y_test"])
                   for cid, d in data["client_data"].items()]
    sizes = [c.data_size for c in clients]
    print(f"{a.clients} clients, {'IID' if a.iid else 'Non-IID'}, samples per client: {sizes}")

    authority = he.KeyAuthority(a.key_bits)
    pk = authority.public_key
    print(f"Key authority: Paillier {a.key_bits}-bit key generated in {authority.keygen_seconds:.2f}s; "
          f"public key sent to clients + server, private key stays with the authority")
    print(f"Packing: {he.slots_per_ciphertext(pk)} weights per ciphertext "
          f"(model has {sum(v.size for v in server.get_global_model().values())} parameters)")
    hook = ClipAndNoise(a.dp_clip, a.dp_noise) if a.dp_noise > 0 else NoPrivacy()
    print(f"Privacy hook before encryption: {type(hook).__name__}"
          + (f" (clip={a.dp_clip}, noise_std={a.dp_noise}) [PLACEHOLDER - no epsilon claim]" if a.dp_noise > 0 else ""))

    def accuracy(weights):
        clients[0].receive_global_model(weights)
        return float(clients[0].evaluate(data["X_test"], data["y_test"]))

    history = []
    template = server.get_global_model()
    n_params = int(sum(v.size for v in template.values()))
    emit("setup", client_ids=[c.client_id for c in clients], sizes=sizes, n_params=n_params,
         slots=he.slots_per_ciphertext(pk), keygen_s=authority.keygen_seconds, key_bits=a.key_bits,
         hook=type(hook).__name__, plain_MB=n_params * 8 / 1e6)
    print(f"\nInitial global accuracy: {accuracy(template):.4f}\n")
    print(f"{'rnd':>3} {'acc (HE)':>9} {'acc (plain)':>11} {'max|HE-plain|':>14} "
          f"{'enc/client':>10} {'server agg':>10} {'decrypt':>8} {'upload/client':>13}")

    for rnd in range(1, a.rounds + 1):
        gw = server.get_global_model()
        gvec = he.flatten_weights(gw)
        emit("round_start", round=rnd)

        # ---- CLIENT SIDE ------------------------------------------------------
        enc_updates, sent_plain, enc_times = [], [], []
        for c in clients:
            c.receive_global_model(gw)
            emit("client_training", round=rnd, client=c.client_id, samples=c.data_size)
            t_train = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                c.local_training(epochs=3, learning_rate=0.1, batch_size=32)
            t_train = time.perf_counter() - t_train
            sent = hook(c.client_id, gvec, he.flatten_weights(c.local_weights))
            sent_plain.append(sent)                      # kept ONLY to verify correctness
            emit("client_encrypting", round=rnd, client=c.client_id)
            u = he.encrypt_update(pk, sent)
            enc_updates.append(u); enc_times.append(u.encrypt_seconds)
            emit("client_done", round=rnd, client=c.client_id, samples=c.data_size, train_s=t_train,
                 encrypt_s=u.encrypt_seconds, n_ciphertexts=len(u.ciphertexts), upload_bytes=u.nbytes,
                 clipped=u.clipped, true_preview=[round(float(x), 5) for x in sent[:3]],
                 ct_preview=hex(u.ciphertexts[0])[:40] + "...")

        if rnd == 1:                                     # show what the server actually receives
            u0 = enc_updates[0]
            again = he.encrypt_update(pk, sent_plain[0]).ciphertexts[0]
            print(f"\n  [what the server sees, round 1, {clients[0].client_id}]")
            print(f"   true first 3 weights : {np.round(sent_plain[0][:3], 5).tolist()}")
            print(f"   ciphertext #0 (hex)  : {hex(u0.ciphertexts[0])[:70]}...  ({u0.key_bits*2//8} bytes)")
            print(f"   same data re-encrypted: {hex(again)[:70]}...  -> different ciphertext "
                  f"({'randomised' if again != u0.ciphertexts[0] else 'SAME?!'})\n")

        # ---- SERVER SIDE (public key only) ------------------------------------
        emit("server_aggregating", round=rnd)
        agg, total, t_agg = he.aggregate_ciphertexts(pk, enc_updates, sizes)
        emit("server_aggregated", round=rnd, seconds=t_agg, agg_preview=hex(agg[0])[:40] + "...")
        emit("authority_decrypting", round=rnd)

        # ---- KEY AUTHORITY (decrypts the aggregate only) ----------------------
        avg, t_dec = authority.decrypt_average(agg, total, u0.n_params, u0.slots)
        emit("authority_decrypted", round=rnd, seconds=t_dec)
        he_weights = he.unflatten_weights(avg, template)

        # ---- correctness check vs the project's own plaintext FedAvg -----------
        plain_weights = server.aggregate_updates([he.unflatten_weights(v, template) for v in sent_plain], sizes)
        err = float(np.max(np.abs(avg - he.flatten_weights(plain_weights))))
        acc_plain = accuracy(plain_weights)
        server.global_weights = he_weights
        server.round_number += 1
        acc_he = accuracy(he_weights)

        row = dict(round=rnd, acc_he=acc_he, acc_plain=acc_plain, max_err=err,
                   enc_s_per_client=float(np.mean(enc_times)), agg_s=t_agg, dec_s=t_dec,
                   upload_MB_per_client=u0.nbytes / 1e6, clipped=int(sum(u.clipped for u in enc_updates)))
        history.append(row)
        emit("round_done", **row)
        if hasattr(hook, "budget_report"):
            emit("dp_budget", round=rnd, report=hook.budget_report())
        print(f"{rnd:>3} {acc_he:>9.4f} {acc_plain:>11.4f} {err:>14.2e} {row['enc_s_per_client']:>9.2f}s "
              f"{t_agg:>9.3f}s {t_dec:>7.2f}s {row['upload_MB_per_client']:>11.3f}MB", flush=True)

    # ---- summary ------------------------------------------------------------------
    worst = max(h["max_err"] for h in history)
    plain_MB = template and sum(v.size for v in template.values()) * 8 / 1e6
    print("\n" + "=" * 78)
    print(f"Final accuracy with encrypted aggregation : {history[-1]['acc_he']:.4f}")
    print(f"Plain FedAvg on the same updates          : {history[-1]['acc_plain']:.4f}")
    print(f"Worst decrypted-vs-plain weight difference: {worst:.2e}  (fixed-point step is {2**-he.FRAC_BITS:.1e})")
    print(f"Upload per client per round: {history[0]['upload_MB_per_client']:.3f} MB encrypted vs {plain_MB:.3f} MB plain "
          f"({history[0]['upload_MB_per_client']/plain_MB:.1f}x)")
    print(f"Per-round HE cost: encrypt {np.mean([h['enc_s_per_client'] for h in history]):.2f}s/client "
          f"(clients work in parallel), aggregate {np.mean([h['agg_s'] for h in history]):.2f}s, "
          f"decrypt {np.mean([h['dec_s'] for h in history]):.2f}s")
    print("=" * 78)

    emit("finished", final_acc_he=history[-1]["acc_he"], final_acc_plain=history[-1]["acc_plain"],
         worst_err=worst, upload_MB=history[0]["upload_MB_per_client"], plain_MB=plain_MB)
    out_dir = os.path.join(HERE, "..", "results"); os.makedirs(out_dir, exist_ok=True)
    meta = dict(clients=a.clients, rounds=a.rounds, key_bits=a.key_bits, iid=a.iid,
                dp_noise=a.dp_noise, slots=he.slots_per_ciphertext(pk), history=history)
    with open(os.path.join(out_dir, "he_demo_results.json"), "w") as f:
        json.dump(meta, f, indent=2)
    if not a.no_plot:
        plot(history, a, os.path.join(out_dir, "he_demo_results.png"))
        print("Saved results/he_demo_results.json and results/he_demo_results.png")


def plot(history, a, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1a1a19", "#6b6a63", "#e3e2dc"
    r = [h["round"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), facecolor="white")
    for x in ax:
        x.set_facecolor("white"); x.grid(axis="y", color=GRID, lw=0.8); x.set_axisbelow(True)
        for s in ("top", "right"): x.spines[s].set_visible(False)
        for s in ("left", "bottom"): x.spines[s].set_color(GRID)
        x.tick_params(colors=MUTED)
    # panel 1: accuracy, both series (they should overlap -> proves encryption does not change the result)
    ax[0].plot(r, [h["acc_plain"] * 100 for h in history], color=ORANGE, lw=2, marker="o", ms=8,
               markeredgecolor="white", markeredgewidth=2, label="Plain FedAvg (same updates)", zorder=2)
    ax[0].plot(r, [h["acc_he"] * 100 for h in history], color=BLUE, lw=2, marker="o", ms=5,
               markeredgecolor="white", markeredgewidth=1.5, label="Encrypted FedAvg (Paillier)", zorder=3)
    ax[0].set_title("Global accuracy per round", loc="left", color=INK, fontsize=12, fontweight="bold")
    ax[0].set_xlabel("Training round", color=MUTED); ax[0].set_ylabel("Test accuracy (%)", color=MUTED)
    ax[0].legend(frameon=False, loc="upper left", labelcolor=INK)
    ax[0].set_xticks(r)
    # panel 2: where HE time goes
    enc = [h["enc_s_per_client"] for h in history]; dec = [h["dec_s"] for h in history]; agg = [h["agg_s"] for h in history]
    ax[1].bar(r, enc, color=BLUE, width=0.7, label="Client encrypt (per client)")
    ax[1].bar(r, dec, bottom=enc, color=ORANGE, width=0.7, label="Authority decrypt", edgecolor="white", linewidth=2)
    ax[1].set_title(f"HE cost per round ({a.key_bits}-bit key)", loc="left", color=INK, fontsize=12, fontweight="bold")
    ax[1].set_xlabel("Training round", color=MUTED); ax[1].set_ylabel("Seconds", color=MUTED)
    ax[1].set_ylim(0, max(e + d for e, d in zip(enc, dec)) * 1.35)
    ax[1].legend(frameon=False, loc="upper left", labelcolor=INK, fontsize=9, ncol=1)
    ax[1].set_xticks(r)
    fig.text(0.01, 0.01, f"Server aggregation (ciphertext products) takes only {np.mean(agg):.2f}s per round. "
             f"{a.clients} clients, {'IID' if a.iid else 'Non-IID'}, digits dataset, 4,810-parameter model.",
             color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(path, dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()
