"""
Differential-privacy hook interface (placeholder for the DP sub-team)
=====================================================================

The HE pipeline calls   hook(client_id, global_weights_vec, local_weights_vec)
on every client BEFORE encryption and expects a flat vector back.

Vidula / Shazia: replace ``ClipAndNoise`` with the real adaptive per-client
epsilon implementation (budget split across rounds, privacy accountant,
gradient/update clipping). Keep this signature and nothing else in the
pipeline has to change.

OPTIONAL (powers the dashboard's privacy-budget panel): give your hook a method
    budget_report() -> {client_id: {"epsilon_total": float, "epsilon_used": float}}
The demo calls it after every round and the dashboard shows it live.

NOTE: ``ClipAndNoise`` below is only a placeholder. It does NOT compute or
claim any epsilon guarantee.
"""

import numpy as np


class NoPrivacy:
    """Identity hook (plain HE-FedAvg)."""

    def __call__(self, client_id: str, global_vec: np.ndarray, local_vec: np.ndarray) -> np.ndarray:
        return local_vec


class ClipAndNoise:
    """Clip the update (local - global) to L2 norm ``clip`` and add Gaussian noise."""

    def __init__(self, clip: float = 1.0, noise_std: float = 0.0, seed: int | None = 0):
        self.clip = clip
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

    def __call__(self, client_id: str, global_vec: np.ndarray, local_vec: np.ndarray) -> np.ndarray:
        delta = local_vec - global_vec
        norm = np.linalg.norm(delta)
        if norm > self.clip:
            delta = delta * (self.clip / norm)
        if self.noise_std > 0:
            delta = delta + self.rng.normal(0.0, self.noise_std, size=delta.shape)
        return global_vec + delta
