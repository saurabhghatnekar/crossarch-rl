"""Strong baseline autotuners and transfer-learning methods for CrossArch-RL."""

from __future__ import annotations

import warnings
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from crossarch_rl.hardware import HardwareSpec
from crossarch_rl.schedule_mdp import (
    ScheduleConfig,
    extract_normalized_state_features,
    extract_raw_state_features,
)
from crossarch_rl.workloads import WorkloadSpec


def _summarize(
    budgets: Sequence[int],
    pct_by_k: Dict[int, List[float]],
    reg_by_k: Dict[int, List[float]],
    log_by_k: Dict[int, List[float]],
) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    for k in budgets:
        out[k] = {
            "mean_log_sol": float(np.mean(log_by_k[k])),
            "mean_regret": float(np.mean(reg_by_k[k])),
            "mean_pct_oracle": float(np.mean(pct_by_k[k])),
            "per_workload_pct_oracle": [float(v) for v in pct_by_k[k]],
            "per_workload_regret": [float(v) for v in reg_by_k[k]],
        }
    return out


class DefaultHeuristicTuner:
    """Standard fixed-heuristic schedule (e.g., library default / untuned middle tile)."""

    def adapt_and_evaluate(
        self,
        target_hw: HardwareSpec,
        workloads: Sequence[WorkloadSpec],
        candidate_pools: Dict[str, List[ScheduleConfig]],
        oracle_fn,
        budgets: Sequence[int] = (0, 1, 3, 5, 10, 15, 25),
        rng_seed: int = 0,
    ) -> Dict[int, Dict[str, float]]:
        pct_by_k = {k: [] for k in budgets}
        reg_by_k = {k: [] for k in budgets}
        log_by_k = {k: [] for k in budgets}
        for wl in workloads:
            pool = candidate_pools[wl.name]
            true_scores = np.array([oracle_fn(wl, c, target_hw, noisy=False) for c in pool], dtype=np.float64)
            best_true = float(np.max(true_scores))
            # Pick a standard mid-size default (closest to 64x64x32, 4 warps, 2 stages)
            target_vec = np.array([64, 64, 32, 4, 2, 2, 1, 0], dtype=np.float64)
            dists = [float(np.linalg.norm(np.array(c.as_tuple()) - target_vec)) for c in pool]
            idx = int(np.argmin(dists))
            val = float(true_scores[idx])
            for k in budgets:
                log_by_k[k].append(val)
                reg_by_k[k].append(best_true - val)
                pct_by_k[k].append(float(np.exp(val - best_true)))
        return _summarize(budgets, pct_by_k, reg_by_k, log_by_k)


class RandomSearchTuner:
    """Uniform random search within hardware-feasible schedule space C_h(x)."""

    def adapt_and_evaluate(
        self,
        target_hw: HardwareSpec,
        workloads: Sequence[WorkloadSpec],
        candidate_pools: Dict[str, List[ScheduleConfig]],
        oracle_fn,
        budgets: Sequence[int] = (0, 1, 3, 5, 10, 15, 25),
        rng_seed: int = 0,
    ) -> Dict[int, Dict[str, float]]:
        rng = np.random.default_rng(rng_seed)
        max_k = max(budgets)
        pct_by_k = {k: [] for k in budgets}
        reg_by_k = {k: [] for k in budgets}
        log_by_k = {k: [] for k in budgets}

        for wl in workloads:
            pool = candidate_pools[wl.name]
            n = len(pool)
            true_scores = np.array([oracle_fn(wl, c, target_hw, noisy=False) for c in pool], dtype=np.float64)
            best_true = float(np.max(true_scores))
            order = rng.permutation(n)

            if 0 in pct_by_k:
                pick = int(order[0])
                log_by_k[0].append(float(true_scores[pick]))
                reg_by_k[0].append(best_true - float(true_scores[pick]))
                pct_by_k[0].append(float(np.exp(true_scores[pick] - best_true)))

            best_obs = -1e9
            best_idx = int(order[0])
            for k in range(1, max_k + 1):
                idx = int(order[(k - 1) % n])
                obs = float(oracle_fn(wl, pool[idx], target_hw, noisy=True))
                if obs > best_obs:
                    best_obs = obs
                    best_idx = idx
                if k in pct_by_k:
                    val = float(true_scores[best_idx])
                    log_by_k[k].append(val)
                    reg_by_k[k].append(best_true - val)
                    pct_by_k[k].append(float(np.exp(val - best_true)))
        return _summarize(budgets, pct_by_k, reg_by_k, log_by_k)


class EvolutionarySearchTuner:
    """Ansor/MetaSchedule-style local mutation evolutionary search on target hardware."""

    def adapt_and_evaluate(
        self,
        target_hw: HardwareSpec,
        workloads: Sequence[WorkloadSpec],
        candidate_pools: Dict[str, List[ScheduleConfig]],
        oracle_fn,
        budgets: Sequence[int] = (0, 1, 3, 5, 10, 15, 25),
        rng_seed: int = 0,
    ) -> Dict[int, Dict[str, float]]:
        rng = np.random.default_rng(rng_seed)
        max_k = max(budgets)
        pct_by_k = {k: [] for k in budgets}
        reg_by_k = {k: [] for k in budgets}
        log_by_k = {k: [] for k in budgets}

        for wl in workloads:
            pool = candidate_pools[wl.name]
            n = len(pool)
            coords = np.array([c.as_tuple() for c in pool], dtype=np.float64)
            coords_norm = (coords - coords.mean(axis=0)) / (coords.std(axis=0) + 1e-6)
            true_scores = np.array([oracle_fn(wl, c, target_hw, noisy=False) for c in pool], dtype=np.float64)
            best_true = float(np.max(true_scores))

            queried = np.zeros(n, dtype=bool)
            pop_idx: List[int] = []
            pop_obs: List[float] = []

            init_idx = int(rng.integers(0, n))
            if 0 in pct_by_k:
                val = float(true_scores[init_idx])
                log_by_k[0].append(val)
                reg_by_k[0].append(best_true - val)
                pct_by_k[0].append(float(np.exp(val - best_true)))

            for k in range(1, max_k + 1):
                if k <= 3 or not pop_idx:
                    avail = np.where(~queried)[0]
                    pick = int(rng.choice(avail)) if len(avail) > 0 else init_idx
                else:
                    # Pick top-3 parent and mutate toward nearest unqueried neighbor
                    top_k = min(3, len(pop_idx))
                    top_order = np.argsort(pop_obs)[-top_k:]
                    parent = pop_idx[int(rng.choice(top_order))]
                    dists = np.linalg.norm(coords_norm - coords_norm[parent], axis=1)
                    dists[queried] = 1e9
                    # Sample softly from top-5 closest neighbors
                    nbrs = np.argsort(dists)[: min(5, n)]
                    pick = int(rng.choice(nbrs))

                queried[pick] = True
                obs = float(oracle_fn(wl, pool[pick], target_hw, noisy=True))
                pop_idx.append(pick)
                pop_obs.append(obs)

                if k in pct_by_k:
                    best_i = pop_idx[int(np.argmax(pop_obs))]
                    val = float(true_scores[best_i])
                    log_by_k[k].append(val)
                    reg_by_k[k].append(best_true - val)
                    pct_by_k[k].append(float(np.exp(val - best_true)))
        return _summarize(budgets, pct_by_k, reg_by_k, log_by_k)


class GPBayesianOptTuner:
    """Gaussian Process Bayesian Optimization (Matern-5/2 + Expected Improvement) from scratch."""

    def adapt_and_evaluate(
        self,
        target_hw: HardwareSpec,
        workloads: Sequence[WorkloadSpec],
        candidate_pools: Dict[str, List[ScheduleConfig]],
        oracle_fn,
        budgets: Sequence[int] = (0, 1, 3, 5, 10, 15, 25),
        rng_seed: int = 0,
    ) -> Dict[int, Dict[str, float]]:
        rng = np.random.default_rng(rng_seed)
        max_k = max(budgets)
        pct_by_k = {k: [] for k in budgets}
        reg_by_k = {k: [] for k in budgets}
        log_by_k = {k: [] for k in budgets}

        for wl in workloads:
            pool = candidate_pools[wl.name]
            n = len(pool)
            X = np.stack([extract_normalized_state_features(wl, c, target_hw) for c in pool], axis=0).astype(np.float64)
            X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)
            true_scores = np.array([oracle_fn(wl, c, target_hw, noisy=False) for c in pool], dtype=np.float64)
            best_true = float(np.max(true_scores))

            queried = np.zeros(n, dtype=bool)
            obs_idx: List[int] = []
            obs_y: List[float] = []

            init_pick = int(rng.integers(0, n))
            if 0 in pct_by_k:
                val = float(true_scores[init_pick])
                log_by_k[0].append(val)
                reg_by_k[0].append(best_true - val)
                pct_by_k[0].append(float(np.exp(val - best_true)))

            for k in range(1, max_k + 1):
                if k <= 3:
                    avail = np.where(~queried)[0]
                    pick = int(rng.choice(avail)) if len(avail) > 0 else init_pick
                else:
                    kernel = ConstantKernel(1.0) * Matern(length_scale=2.0, nu=2.5) + WhiteKernel(noise_level=1e-2)
                    gp = GaussianProcessRegressor(kernel=kernel, optimizer=None, random_state=rng_seed)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", ConvergenceWarning)
                        gp.fit(X[obs_idx], np.array(obs_y))
                    mu, std = gp.predict(X, return_std=True)
                    best_y = float(np.max(obs_y))
                    std_safe = np.maximum(std, 1e-6)
                    z = (mu - best_y - 0.01) / std_safe
                    ei = (mu - best_y - 0.01) * norm.cdf(z) + std_safe * norm.pdf(z)
                    ei[queried] = -1e9
                    pick = int(np.argmax(ei))

                queried[pick] = True
                obs = float(oracle_fn(wl, pool[pick], target_hw, noisy=True))
                obs_idx.append(pick)
                obs_y.append(obs)

                if k in pct_by_k:
                    best_i = obs_idx[int(np.argmax(obs_y))]
                    val = float(true_scores[best_i])
                    log_by_k[k].append(val)
                    reg_by_k[k].append(best_true - val)
                    pct_by_k[k].append(float(np.exp(val - best_true)))
        return _summarize(budgets, pct_by_k, reg_by_k, log_by_k)


class NeuralUCBTuner:
    """Online Neural-UCB contextual bandit trained from scratch on target hardware."""

    def adapt_and_evaluate(
        self,
        target_hw: HardwareSpec,
        workloads: Sequence[WorkloadSpec],
        candidate_pools: Dict[str, List[ScheduleConfig]],
        oracle_fn,
        budgets: Sequence[int] = (0, 1, 3, 5, 10, 15, 25),
        rng_seed: int = 0,
    ) -> Dict[int, Dict[str, float]]:
        rng = np.random.default_rng(rng_seed)
        max_k = max(budgets)
        pct_by_k = {k: [] for k in budgets}
        reg_by_k = {k: [] for k in budgets}
        log_by_k = {k: [] for k in budgets}

        for wl in workloads:
            pool = candidate_pools[wl.name]
            n = len(pool)
            X = np.stack([extract_normalized_state_features(wl, c, target_hw) for c in pool], axis=0).astype(np.float64)
            X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)
            d = X.shape[1]
            # Random Fourier / ReLU neural feature projection for fast exact Neural-UCB
            W_rf = rng.normal(0.0, 1.0 / np.sqrt(d), size=(d, 32))
            b_rf = rng.uniform(0.0, 2.0 * np.pi, size=(32,))
            Phi = np.cos(X @ W_rf + b_rf) * np.sqrt(2.0 / 32.0)

            true_scores = np.array([oracle_fn(wl, c, target_hw, noisy=False) for c in pool], dtype=np.float64)
            best_true = float(np.max(true_scores))

            A = np.eye(32) * 1.0
            b_vec = np.zeros(32)
            queried = np.zeros(n, dtype=bool)
            obs_idx: List[int] = []
            obs_y: List[float] = []

            init_pick = int(rng.integers(0, n))
            if 0 in pct_by_k:
                val = float(true_scores[init_pick])
                log_by_k[0].append(val)
                reg_by_k[0].append(best_true - val)
                pct_by_k[0].append(float(np.exp(val - best_true)))

            for k in range(1, max_k + 1):
                A_inv = np.linalg.inv(A)
                theta = A_inv @ b_vec
                mean = Phi @ theta
                unc = np.sqrt(np.maximum(np.sum((Phi @ A_inv) * Phi, axis=1), 1e-12))
                ucb = mean + 1.5 * unc + rng.normal(0.0, 1e-6, size=n)
                ucb[queried] = -1e9
                pick = int(np.argmax(ucb))

                queried[pick] = True
                obs = float(oracle_fn(wl, pool[pick], target_hw, noisy=True))
                obs_idx.append(pick)
                obs_y.append(obs)

                A += np.outer(Phi[pick], Phi[pick])
                b_vec += Phi[pick] * obs

                if k in pct_by_k:
                    best_i = obs_idx[int(np.argmax(obs_y))]
                    val = float(true_scores[best_i])
                    log_by_k[k].append(val)
                    reg_by_k[k].append(best_true - val)
                    pct_by_k[k].append(float(np.exp(val - best_true)))
        return _summarize(budgets, pct_by_k, reg_by_k, log_by_k)


class _TransferMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ZeroShotTransferMLPTuner:
    """
    End-to-end offline MLP cost model trained on source hardware(s) using raw + datasheet
    features, deployed zero-shot (or selecting top-K predicted schedules) on target hardware.
    """

    def __init__(self, use_normalized: bool = False, seed: int = 0) -> None:
        self.use_normalized = use_normalized
        self.seed = seed
        self.model: _TransferMLP | None = None
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def _feat(self, wl: WorkloadSpec, sc: ScheduleConfig, hw: HardwareSpec) -> np.ndarray:
        base = extract_normalized_state_features(wl, sc, hw) if self.use_normalized else extract_raw_state_features(wl, sc, hw)
        return np.concatenate([base, np.array(hw.datasheet_vector(), dtype=np.float32)], axis=0)

    def fit_offline(
        self, source_traces: Sequence[Tuple[HardwareSpec, WorkloadSpec, ScheduleConfig, float]]
    ) -> None:
        torch.manual_seed(self.seed)
        X = np.stack([self._feat(wl, sc, hw) for hw, wl, sc, _ in source_traces], axis=0).astype(np.float32)
        Y = np.array([y for _, _, _, y in source_traces], dtype=np.float32)
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-5
        Xn = torch.from_numpy((X - self.mean) / self.std)
        Yt = torch.from_numpy(Y)

        self.model = _TransferMLP(X.shape[1], hidden=64)
        opt = torch.optim.AdamW(self.model.parameters(), lr=3e-3, weight_decay=1e-4)
        for _ in range(140):
            perm = torch.randperm(Xn.shape[0])
            for s in range(0, Xn.shape[0], 128):
                idx = perm[s : s + 128]
                pred = self.model(Xn[idx])
                loss = ((pred - Yt[idx]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()

    def adapt_and_evaluate(
        self,
        target_hw: HardwareSpec,
        workloads: Sequence[WorkloadSpec],
        candidate_pools: Dict[str, List[ScheduleConfig]],
        oracle_fn,
        budgets: Sequence[int] = (0, 1, 3, 5, 10, 15, 25),
        rng_seed: int = 0,
    ) -> Dict[int, Dict[str, float]]:
        assert self.model is not None and self.mean is not None and self.std is not None
        max_k = max(budgets)
        pct_by_k = {k: [] for k in budgets}
        reg_by_k = {k: [] for k in budgets}
        log_by_k = {k: [] for k in budgets}

        for wl in workloads:
            pool = candidate_pools[wl.name]
            X = np.stack([self._feat(wl, c, target_hw) for c in pool], axis=0).astype(np.float32)
            Xn = torch.from_numpy((X - self.mean) / self.std)
            with torch.no_grad():
                preds = self.model(Xn).numpy()
            rank = np.argsort(preds)[::-1]
            true_scores = np.array([oracle_fn(wl, c, target_hw, noisy=False) for c in pool], dtype=np.float64)
            best_true = float(np.max(true_scores))

            if 0 in pct_by_k:
                top0 = int(rank[0])
                log_by_k[0].append(float(true_scores[top0]))
                reg_by_k[0].append(best_true - float(true_scores[top0]))
                pct_by_k[0].append(float(np.exp(true_scores[top0] - best_true)))

            best_obs = -1e9
            best_idx = int(rank[0])
            for k in range(1, max_k + 1):
                idx = int(rank[(k - 1) % len(rank)])
                obs = float(oracle_fn(wl, pool[idx], target_hw, noisy=True))
                if obs > best_obs:
                    best_obs = obs
                    best_idx = idx
                if k in pct_by_k:
                    val = float(true_scores[best_idx])
                    log_by_k[k].append(val)
                    reg_by_k[k].append(best_true - val)
                    pct_by_k[k].append(float(np.exp(val - best_true)))
        return _summarize(budgets, pct_by_k, reg_by_k, log_by_k)


class FineTunedMLPTuner(ZeroShotTransferMLPTuner):
    """Offline source MLP (with normalized features) fine-tuned online via gradient steps on target shots."""

    def __init__(self, ft_steps: int = 25, ft_lr: float = 2e-3, seed: int = 0) -> None:
        super().__init__(use_normalized=True, seed=seed)
        self.ft_steps = ft_steps
        self.ft_lr = ft_lr

    def adapt_and_evaluate(
        self,
        target_hw: HardwareSpec,
        workloads: Sequence[WorkloadSpec],
        candidate_pools: Dict[str, List[ScheduleConfig]],
        oracle_fn,
        budgets: Sequence[int] = (0, 1, 3, 5, 10, 15, 25),
        rng_seed: int = 0,
    ) -> Dict[int, Dict[str, float]]:
        import copy

        assert self.model is not None and self.mean is not None and self.std is not None
        rng = np.random.default_rng(rng_seed)
        max_k = max(budgets)
        pct_by_k = {k: [] for k in budgets}
        reg_by_k = {k: [] for k in budgets}
        log_by_k = {k: [] for k in budgets}

        for wl in workloads:
            pool = candidate_pools[wl.name]
            n = len(pool)
            X = np.stack([self._feat(wl, c, target_hw) for c in pool], axis=0).astype(np.float32)
            Xn = torch.from_numpy((X - self.mean) / self.std)
            true_scores = np.array([oracle_fn(wl, c, target_hw, noisy=False) for c in pool], dtype=np.float64)
            best_true = float(np.max(true_scores))

            model_copy = copy.deepcopy(self.model)
            queried = np.zeros(n, dtype=bool)
            obs_idx: List[int] = []
            obs_y: List[float] = []

            with torch.no_grad():
                init_preds = model_copy(Xn).numpy()
            if 0 in pct_by_k:
                top0 = int(np.argmax(init_preds))
                log_by_k[0].append(float(true_scores[top0]))
                reg_by_k[0].append(best_true - float(true_scores[top0]))
                pct_by_k[0].append(float(np.exp(true_scores[top0] - best_true)))

            for k in range(1, max_k + 1):
                with torch.no_grad():
                    preds = model_copy(Xn).numpy()
                # Epsilon-greedy exploration (20% random) to avoid greedy lock-in
                if rng.random() < 0.20:
                    avail = np.where(~queried)[0]
                    pick = int(rng.choice(avail)) if len(avail) > 0 else 0
                else:
                    preds_m = preds.copy()
                    preds_m[queried] = -1e9
                    pick = int(np.argmax(preds_m))

                queried[pick] = True
                obs = float(oracle_fn(wl, pool[pick], target_hw, noisy=True))
                obs_idx.append(pick)
                obs_y.append(obs)

                # Gradient fine-tuning on observed target measurements
                opt = torch.optim.Adam(model_copy.parameters(), lr=self.ft_lr, weight_decay=1e-3)
                xb = Xn[obs_idx]
                yb = torch.tensor(obs_y, dtype=torch.float32)
                for _ in range(self.ft_steps):
                    loss = ((model_copy(xb) - yb) ** 2).mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                if k in pct_by_k:
                    best_i = obs_idx[int(np.argmax(obs_y))]
                    val = float(true_scores[best_i])
                    log_by_k[k].append(val)
                    reg_by_k[k].append(best_true - val)
                    pct_by_k[k].append(float(np.exp(val - best_true)))
        return _summarize(budgets, pct_by_k, reg_by_k, log_by_k)
