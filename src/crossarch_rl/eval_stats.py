"""rliable statistical utilities (IQM + stratified bootstrap 95% CIs) & paper plots."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


def interquartile_mean(values: np.ndarray) -> float:
    """Compute Interquartile Mean (IQM) per Agarwal et al. (NeurIPS 2021)."""
    flat = np.sort(np.asarray(values, dtype=np.float64).ravel())
    n = len(flat)
    if n < 4:
        return float(np.mean(flat))
    lo = int(np.floor(0.25 * n))
    hi = int(np.ceil(0.75 * n))
    return float(np.mean(flat[lo:hi]))


def stratified_bootstrap_iqm(
    scores_seed_x_task: np.ndarray,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """
    Stratified bootstrap 95% confidence interval for IQM across (n_seeds, n_tasks).
    Returns (point_iqm, ci_low, ci_high).
    """
    arr = np.asarray(scores_seed_x_task, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    n_seeds, n_tasks = arr.shape
    point = interquartile_mean(arr)

    rng = np.random.default_rng(seed)
    boot = np.empty(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        # Resample seeds independently within each task stratum
        idx = rng.integers(0, n_seeds, size=(n_seeds, n_tasks))
        sample = arr[idx, np.arange(n_tasks)[None, :]]
        boot[b] = interquartile_mean(sample)

    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(boot, alpha))
    hi = float(np.quantile(boot, 1.0 - alpha))
    return point, lo, hi


def probability_of_improvement(
    scores_a: np.ndarray, scores_b: np.ndarray
) -> float:
    """Mann-Whitney U probability that method A outperforms method B (Agarwal et al.)."""
    a = np.asarray(scores_a, dtype=np.float64).ravel()
    b = np.asarray(scores_b, dtype=np.float64).ravel()
    diff = a[:, None] - b[None, :]
    return float(np.mean((diff > 0.0).astype(np.float64) + 0.5 * (diff == 0.0).astype(np.float64)))
