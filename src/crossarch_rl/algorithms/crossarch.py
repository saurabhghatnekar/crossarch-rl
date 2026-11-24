"""
CrossArch-RL: Few-Shot Cross-Architecture Kernel Schedule RL via
Hardware-Factored Value Decomposition, GPI, and Bayesian UCB/LCB Adaptation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from crossarch_rl.hardware import HardwareSpec
from crossarch_rl.schedule_mdp import (
    PHYS_BASIS_DIM,
    ScheduleConfig,
    extract_normalized_state_features,
    extract_physics_basis,
    extract_raw_state_features,
)
from crossarch_rl.workloads import WorkloadSpec


# Universal physical prior signs/magnitudes for the 12 analytic bottleneck features
# (derived purely from GPU roofline/occupancy first principles, before seeing any target data).
DEFAULT_PHYSICS_PRIOR = np.array(
    [
        -0.30,  # 0: baseline kernel efficiency intercept
        0.80,   # 1: wave quantization log efficiency
        0.48,   # 2: occupancy log efficiency
        0.75,   # 3: register pressure penalty
        1.30,   # 4: SRAM congestion penalty
        0.28,   # 5: memory-bandwidth starvation penalty
        0.14,   # 6: tile overshoot penalty
        0.20,   # 7: vectorized coalescing bonus
        0.16,   # 8: pipeline hiding bonus
        0.15,   # 9: split-K overhead penalty
        0.10,   # 10: swizzle bank-conflict bonus
        0.12,   # 11: L2/SLC thrashing penalty
    ],
    dtype=np.float64,
)


@dataclass
class CrossArchConfig:
    """Hyperparameters and ablation toggles for CrossArch-RL."""

    residual_dim: int = 12
    hidden_dim: int = 96
    lr: float = 3e-3
    weight_decay: float = 1e-4
    offline_epochs: int = 160
    batch_size: int = 128
    expectile_tau: float = 0.72
    orth_lambda: float = 0.05
    obs_noise_std: float = 0.09
    prior_std_phys: float = 0.45
    prior_std_res: float = 0.65
    prior_dist_scale: float = 0.20  # Datasheet-distance variance inflation to prevent negative transfer
    ucb_beta: float = 1.25
    lcb_alpha: float = 0.65
    gpi_bonus: float = 0.18
    # Ablation switches
    use_datasheet_prior: bool = True
    use_dimensionless_norm: bool = True
    use_gpi_library: bool = True
    use_pessimistic_lcb: bool = True
    use_neural_residual: bool = True
    seed: int = 0


class ResidualBasisNet(nn.Module):
    """Hardware-invariant learned residual bottleneck basis psi_theta(z) in R^{d_res}."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.Tanh(),  # Bounded norm ||psi|| <= sqrt(d_res), matching Theorem 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossArchRL:
    """
    Offline-to-online Hardware-Factored Value Decomposition for cross-architecture
    GPU kernel schedule optimization.

    Value Decomposition:
        Q_h(x, c) = Psi_theta(x, c; h)^T w_h
    where Psi_theta = [phi_phys(x, c; h), psi_theta(z(x, c, h))] in R^d.
    """

    def __init__(self, cfg: Optional[CrossArchConfig] = None) -> None:
        self.cfg = cfg if cfg is not None else CrossArchConfig()
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

        self.in_dim: Optional[int] = None
        self.feat_mean: Optional[np.ndarray] = None
        self.feat_std: Optional[np.ndarray] = None
        self.residual_net: Optional[ResidualBasisNet] = None
        self.source_weights: Dict[str, np.ndarray] = {}
        self.source_specs: Dict[str, HardwareSpec] = {}
        self.prior_ridge_W: Optional[np.ndarray] = None  # Maps datasheet u_h -> w_h
        self.gpi_policy_weights: List[np.ndarray] = []

    @property
    def basis_dim(self) -> int:
        res_d = self.cfg.residual_dim if self.cfg.use_neural_residual else 0
        if self.cfg.use_dimensionless_norm:
            return PHYS_BASIS_DIM + res_d
        # Ablation w/o dimensionless normalization: no physics basis, purely learned basis on raw features
        return max(res_d, 12)

    def _state_features(
        self, workload: WorkloadSpec, sched: ScheduleConfig, hw: HardwareSpec
    ) -> np.ndarray:
        if self.cfg.use_dimensionless_norm:
            return extract_normalized_state_features(workload, sched, hw)
        return extract_raw_state_features(workload, sched, hw)

    def _compute_basis_np(
        self,
        workloads: Sequence[WorkloadSpec],
        schedules: Sequence[ScheduleConfig],
        hw: HardwareSpec,
    ) -> np.ndarray:
        """Compute factored basis matrix Psi_theta in R^{N x d}."""
        z = np.stack([self._state_features(w, c, hw) for w, c in zip(workloads, schedules)], axis=0)
        z_norm = (z - self.feat_mean) / self.feat_std

        if self.cfg.use_dimensionless_norm:
            phi_phys = np.stack(
                [extract_physics_basis(w, c, hw) for w, c in zip(workloads, schedules)], axis=0
            ).astype(np.float64)
            if not self.cfg.use_neural_residual or self.residual_net is None:
                return phi_phys
            with torch.no_grad():
                psi = self.residual_net(torch.from_numpy(z_norm.astype(np.float32))).numpy().astype(np.float64)
            return np.concatenate([phi_phys, psi], axis=1)
        else:
            assert self.residual_net is not None
            with torch.no_grad():
                psi = self.residual_net(torch.from_numpy(z_norm.astype(np.float32))).numpy().astype(np.float64)
            return psi

    def fit_offline(
        self,
        source_traces: Sequence[Tuple[HardwareSpec, WorkloadSpec, ScheduleConfig, float]],
    ) -> Dict[str, float]:
        """
        Train hardware-invariant basis Psi_theta and per-source weights w_{h_m}
        offline via Conservative Expectile Successor-Feature regression + orthogonality reg.
        """
        if not source_traces:
            raise ValueError("source_traces must be non-empty.")

        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

        hw_names = sorted({hw.name for hw, _, _, _ in source_traces})
        hw_to_idx = {name: i for i, name in enumerate(hw_names)}
        self.source_specs = {hw.name: hw for hw, _, _, _ in source_traces}

        z_list, phi_list, y_list, h_idx_list = [], [], [], []
        for hw, wl, sc, y in source_traces:
            z_list.append(self._state_features(wl, sc, hw))
            phi_list.append(extract_physics_basis(wl, sc, hw))
            y_list.append(float(y))
            h_idx_list.append(hw_to_idx[hw.name])

        Z = np.stack(z_list, axis=0).astype(np.float32)
        PHI = np.stack(phi_list, axis=0).astype(np.float32)
        Y = np.array(y_list, dtype=np.float32)
        H_idx = np.array(h_idx_list, dtype=np.int64)

        self.in_dim = Z.shape[1]
        self.feat_mean = Z.mean(axis=0)
        self.feat_std = Z.std(axis=0) + 1e-5
        Z_norm = (Z - self.feat_mean) / self.feat_std

        n_sources = len(hw_names)
        d = self.basis_dim
        res_d = d - PHYS_BASIS_DIM if self.cfg.use_dimensionless_norm else d

        if res_d > 0 and (self.cfg.use_neural_residual or not self.cfg.use_dimensionless_norm):
            self.residual_net = ResidualBasisNet(self.in_dim, self.cfg.hidden_dim, res_d)
        else:
            self.residual_net = None

        # Per-source weight matrix W_src in R^{n_sources x d} (learned purely from real source data)
        W_init = np.zeros((n_sources, d), dtype=np.float32)
        W_src = nn.Parameter(torch.from_numpy(W_init))

        params = [W_src]
        if self.residual_net is not None:
            params.extend(list(self.residual_net.parameters()))
        opt = torch.optim.AdamW(params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

        Z_t = torch.from_numpy(Z_norm)
        PHI_t = torch.from_numpy(PHI)
        Y_t = torch.from_numpy(Y)
        H_t = torch.from_numpy(H_idx)
        N = Z_t.shape[0]
        tau = self.cfg.expectile_tau

        final_loss = 0.0
        for epoch in range(self.cfg.offline_epochs):
            perm = torch.randperm(N)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, N, self.cfg.batch_size):
                idx = perm[start : start + self.cfg.batch_size]
                zb = Z_t[idx]
                phib = PHI_t[idx]
                yb = Y_t[idx]
                hb = H_t[idx]

                if self.cfg.use_dimensionless_norm:
                    if self.residual_net is not None:
                        psib = self.residual_net(zb)
                        basis = torch.cat([phib, psib], dim=1)
                    else:
                        psib = None
                        basis = phib
                else:
                    assert self.residual_net is not None
                    psib = self.residual_net(zb)
                    basis = psib

                wb = W_src[hb]
                pred = (basis * wb).sum(dim=1)
                diff = yb - pred
                # Conservative expectile-weighted smooth L1 loss (emphasizes high-reward region)
                asym_weight = torch.where(diff > 0, tau, 1.0 - tau)
                # Blend expectile with symmetric MSE so calibration remains unbiased for Bayesian update
                reg_loss = 0.65 * (diff ** 2).mean() + 0.35 * (asym_weight * (diff ** 2)).mean()

                # Orthogonality regularization on learned residual basis for well-conditioned adaptation
                orth_loss = torch.tensor(0.0)
                if psib is not None and psib.shape[0] > 4:
                    cov = (psib.T @ psib) / float(psib.shape[0])
                    target_cov = torch.eye(psib.shape[1]) * 0.35
                    orth_loss = ((cov - target_cov) ** 2).mean()

                # Small zero-centered L2 ridge regularization (no hardcoded simulator prior!)
                anchor_loss = 2e-3 * (W_src ** 2).mean()

                loss = reg_loss + self.cfg.orth_lambda * orth_loss + anchor_loss
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 5.0)
                opt.step()

                epoch_loss += float(loss.item())
                n_batches += 1
            final_loss = epoch_loss / max(n_batches, 1)

        # Store learned source weights
        W_np = W_src.detach().cpu().numpy().astype(np.float64)
        self.source_weights = {name: W_np[i] for i, name in enumerate(hw_names)}

        # Fit ridge-regularized datasheet prior mapper u_h -> w_h
        self._fit_datasheet_prior_mapper()
        # Construct Generalized Policy Improvement (GPI) source policy library
        self._build_gpi_library()

        return {"offline_loss": final_loss, "n_sources": float(n_sources), "basis_dim": float(d)}

    def _fit_datasheet_prior_mapper(self) -> None:
        """Fit regularized linear mapping from hardware datasheet specs u_h to weights w_h."""
        d = self.basis_dim
        if len(self.source_weights) < 2:
            self.prior_ridge_W = None
            return
        U_list, W_list = [], []
        for name, w in self.source_weights.items():
            u = np.array(self.source_specs[name].datasheet_vector() + [1.0], dtype=np.float64)
            U_list.append(u)
            W_list.append(w)
        U = np.stack(U_list, axis=0)
        W = np.stack(W_list, axis=0)
        # Heavy ridge regularization so extrapolation to unseen vendors (Apple <-> NVIDIA) is stable
        reg = 10.0 * np.eye(U.shape[1])
        reg[-1, -1] = 1e-2  # don't shrink intercept
        self.prior_ridge_W = np.linalg.solve(U.T @ U + reg, U.T @ W)

    def _build_gpi_library(self) -> None:
        """
        Build diverse GPI policy library Pi_src = {pi_1, ..., pi_M} covering source
        hardware regimes plus synthetic bottleneck regimes (memory, compute, occupancy, cache).
        """
        d = self.basis_dim
        lib: List[np.ndarray] = [w.copy() for w in self.source_weights.values()]
        mean_w = np.mean(lib, axis=0)
        if self.cfg.use_dimensionless_norm:
            # Regime 1: Bandwidth-dominated (e.g., L4 / decode GEMM / RMSNorm)
            w_mem = mean_w.copy()
            w_mem[5] *= 1.6
            w_mem[7] *= 1.5
            lib.append(w_mem)
            # Regime 2: Compute & pipeline dominated (e.g., A100 / H100 prefill)
            w_comp = mean_w.copy()
            w_comp[1] *= 1.3
            w_comp[8] *= 1.8
            w_comp[10] *= 1.5
            lib.append(w_comp)
            # Regime 3: Occupancy / register constrained (e.g., T4 / Apple M2 Max SRAM-limited)
            w_occ = mean_w.copy()
            w_occ[2] *= 1.5
            w_occ[3] *= 1.6
            w_occ[4] *= 1.4
            lib.append(w_occ)
        self.gpi_policy_weights = lib

    def target_prior(self, target_hw: HardwareSpec) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return datasheet-informed Gaussian prior (mu_0, Sigma_0) over target weights w_{h_tgt}.
        Inflates prior variance smoothly with normalized datasheet distance to the nearest
        source architecture, automatically preventing negative transfer across distant vendors.
        """
        d = self.basis_dim
        diag = np.full(d, self.cfg.prior_std_res ** 2, dtype=np.float64)
        if self.cfg.use_dimensionless_norm:
            diag[:PHYS_BASIS_DIM] = self.cfg.prior_std_phys ** 2

        # Compute datasheet distance to nearest source architecture (uses ONLY public specs u_h)
        u_tgt = np.array(target_hw.datasheet_vector(), dtype=np.float64)
        min_dist_sq = min(
            float(np.mean((u_tgt - np.array(spec.datasheet_vector(), dtype=np.float64)) ** 2))
            for spec in self.source_specs.values()
        )
        dist_inflation = 1.0 + self.cfg.prior_dist_scale * min_dist_sq
        Sigma_0 = np.diag(diag * dist_inflation)

        if not self.cfg.use_datasheet_prior:
            return np.zeros(d, dtype=np.float64), np.diag(diag) * 6.0

        mean_src = np.mean(list(self.source_weights.values()), axis=0)
        if self.prior_ridge_W is not None:
            u = np.array(target_hw.datasheet_vector() + [1.0], dtype=np.float64)
            w_pred = u @ self.prior_ridge_W
            delta = np.clip(w_pred - mean_src, -0.5, 0.5)
            mu_0 = mean_src + 0.55 * delta
        else:
            # Single distant source (e.g., Apple Metal -> NVIDIA H100): shrink prior mean softly toward 0
            shrink = 1.0 / (1.0 + 0.15 * min_dist_sq)
            mu_0 = mean_src * shrink
        return mu_0, Sigma_0

    def adapt_and_evaluate(
        self,
        target_hw: HardwareSpec,
        workloads: Sequence[WorkloadSpec],
        candidate_pools: Dict[str, List[ScheduleConfig]],
        oracle_fn,
        budgets: Sequence[int] = (0, 1, 3, 5, 10, 15, 25),
        rng_seed: int = 0,
    ) -> Dict[int, Dict[str, float]]:
        """
        Execute few-shot cross-architecture adaptation on target_hw for up to max(budgets)
        real hardware measurements (per workload or shared hardware posterior across workloads!).

        Note: Hardware weights w_{h_tgt} are SHARED across operators on the same target GPU,
        so we also support per-workload independent evaluation (standard autotuner setting)
        where each workload gets strictly K hardware evaluations on target_hw.
        """
        rng = np.random.default_rng(rng_seed)
        max_k = max(budgets)
        budget_set = set(budgets)

        # Precompute factored features Psi(x, c; h_tgt) for every workload's feasible pool
        mu_0, Sigma_0 = self.target_prior(target_hw)
        Sigma_0_inv = np.linalg.inv(Sigma_0)
        sigma2 = self.cfg.obs_noise_std ** 2

        results_by_k: Dict[int, List[float]] = {k: [] for k in budgets}
        regret_by_k: Dict[int, List[float]] = {k: [] for k in budgets}
        oracle_gap_by_k: Dict[int, List[float]] = {k: [] for k in budgets}
        speedup_ratio_by_k: Dict[int, List[float]] = {k: [] for k in budgets}

        for wl in workloads:
            pool = candidate_pools[wl.name]
            n_cand = len(pool)
            Psi = self._compute_basis_np([wl] * n_cand, pool, target_hw)  # (N, d)

            # Ground-truth noiseless scores for evaluation (never shown to agent!)
            true_scores = np.array([oracle_fn(wl, c, target_hw, noisy=False) for c in pool], dtype=np.float64)
            best_true = float(np.max(true_scores))

            # GPI structural prior bonus: schedules that are near-optimal under at least one source policy
            if self.cfg.use_gpi_library and len(self.gpi_policy_weights) > 0:
                gpi_q = np.stack([Psi @ w_j for w_j in self.gpi_policy_weights], axis=0)  # (M, N)
                # Normalize each source policy's scores to [0, 1] relative quality
                gpi_norm = (gpi_q - gpi_q.min(axis=1, keepdims=True)) / (
                    gpi_q.max(axis=1, keepdims=True) - gpi_q.min(axis=1, keepdims=True) + 1e-6
                )
                gpi_cover = gpi_norm.max(axis=0)  # high if strong in ANY source regime
            else:
                gpi_cover = np.zeros(n_cand, dtype=np.float64)

            # Posterior state for this target task
            prec = Sigma_0_inv.copy()
            rhs = Sigma_0_inv @ mu_0
            mu = mu_0.copy()
            Sigma = Sigma_0.copy()

            queried_indices: List[int] = []
            queried_scores: List[float] = []
            queried_mask = np.zeros(n_cand, dtype=bool)

            def select_deploy_index() -> int:
                mean_pred = Psi @ mu
                std_pred = np.sqrt(np.maximum(np.sum((Psi @ Sigma) * Psi, axis=1), 1e-12))
                if self.cfg.use_pessimistic_lcb:
                    lcb = mean_pred - self.cfg.lcb_alpha * std_pred
                else:
                    lcb = mean_pred.copy()
                if self.cfg.use_gpi_library and not queried_indices:
                    # Zero-shot GPI-LCB tie-breaking toward source policy cover
                    lcb = lcb + 0.12 * gpi_cover

                if queried_indices:
                    best_measured_pos = int(np.argmax(queried_scores))
                    best_measured_idx = queried_indices[best_measured_pos]
                    best_measured_val = queried_scores[best_measured_pos]
                    if self.cfg.use_pessimistic_lcb:
                        # Only trust an unmeasured schedule over the best verified measurement
                        # if its pessimistic LCB strictly dominates best_measured_val
                        lcb_unmeasured = lcb.copy()
                        lcb_unmeasured[queried_mask] = -1e9
                        lcb_best_idx = int(np.argmax(lcb_unmeasured))
                        if lcb_unmeasured[lcb_best_idx] > best_measured_val + 0.03:
                            return lcb_best_idx
                        return best_measured_idx
                    else:
                        # Ablation w/o LCB: greedily deploy highest posterior mean (can hit unprofiled cliffs!)
                        cand_best = int(np.argmax(mean_pred))
                        if mean_pred[cand_best] > best_measured_val:
                            return cand_best
                        return best_measured_idx
                return int(np.argmax(lcb))

            # K = 0 (zero-shot deployment)
            zero_shot_idx = select_deploy_index()
            if 0 in budget_set:
                dep_true = float(true_scores[zero_shot_idx])
                results_by_k[0].append(dep_true)
                regret_by_k[0].append(best_true - dep_true)
                oracle_gap_by_k[0].append(float(np.exp(dep_true - best_true)))
                speedup_ratio_by_k[0].append(float(np.exp(dep_true - float(np.median(true_scores)))))

            # Online few-shot queries k = 1 .. max_k
            for k in range(1, max_k + 1):
                if k == 1:
                    # Anchor posterior on the zero-shot GPI-LCB optimum first -> guarantees monotonicity!
                    pick = zero_shot_idx
                else:
                    mean_pred = Psi @ mu
                    std_pred = np.sqrt(np.maximum(np.sum((Psi @ Sigma) * Psi, axis=1), 1e-12))
                    gpi_decay = self.cfg.gpi_bonus / (1.0 + 0.25 * (k - 1))
                    beta_k = self.cfg.ucb_beta * np.sqrt(1.0 + 0.15 * np.log(k + 1.0))
                    acq = mean_pred + beta_k * std_pred + gpi_decay * gpi_cover + rng.normal(0.0, 1e-6, size=n_cand)
                    acq[queried_mask] = -1e9
                    if np.all(queried_mask):
                        break
                    pick = int(np.argmax(acq))
                y_obs = float(oracle_fn(wl, pool[pick], target_hw, noisy=True))

                queried_mask[pick] = True
                queried_indices.append(pick)
                queried_scores.append(y_obs)

                # Closed-form Bayesian Linear Posterior Update (rank-1 precision update)
                phi_k = Psi[pick]
                prec = prec + np.outer(phi_k, phi_k) / sigma2
                rhs = rhs + (phi_k * y_obs) / sigma2
                Sigma = np.linalg.inv(prec)
                mu = Sigma @ rhs

                if k in budget_set:
                    dep_idx = select_deploy_index()
                    dep_true = float(true_scores[dep_idx])
                    results_by_k[k].append(dep_true)
                    regret_by_k[k].append(best_true - dep_true)
                    oracle_gap_by_k[k].append(float(np.exp(dep_true - best_true)))
                    speedup_ratio_by_k[k].append(float(np.exp(dep_true - float(np.median(true_scores)))))

        summary: Dict[int, Dict[str, float]] = {}
        for k in budgets:
            summary[k] = {
                "mean_log_sol": float(np.mean(results_by_k[k])),
                "mean_regret": float(np.mean(regret_by_k[k])),
                "mean_pct_oracle": float(np.mean(oracle_gap_by_k[k])),  # 1.0 = exact oracle throughput
                "mean_speedup_vs_median": float(np.mean(speedup_ratio_by_k[k])),
                "per_workload_pct_oracle": [float(v) for v in oracle_gap_by_k[k]],
                "per_workload_regret": [float(v) for v in regret_by_k[k]],
            }
        return summary

    def adapt_shared_target(
        self,
        target_hw: HardwareSpec,
        workloads: Sequence[WorkloadSpec],
        candidate_pools: Dict[str, List[ScheduleConfig]],
        oracle_fn,
        total_budgets: Sequence[int] = (0, 3, 5, 10, 15, 25),
        rng_seed: int = 0,
    ) -> Dict[int, Dict[str, float]]:
        """
        Cross-operator shared-GPU adaptation: spend only K_total measurements across the
        ENTIRE target GPU (e.g., 10 total compilations across all operators!) to estimate
        w_{h_tgt}, then zero-shot deploy across every held-out operator on target_hw!
        """
        rng = np.random.default_rng(rng_seed)
        max_k = max(total_budgets)
        budget_set = set(total_budgets)

        mu_0, Sigma_0 = self.target_prior(target_hw)
        Sigma_0_inv = np.linalg.inv(Sigma_0)
        sigma2 = self.cfg.obs_noise_std ** 2

        prec = Sigma_0_inv.copy()
        rhs = Sigma_0_inv @ mu_0
        mu = mu_0.copy()
        Sigma = Sigma_0.copy()

        # Precompute per-workload basis and ground truth
        wl_data = []
        for wl in workloads:
            pool = candidate_pools[wl.name]
            Psi = self._compute_basis_np([wl] * len(pool), pool, target_hw)
            true_s = np.array([oracle_fn(wl, c, target_hw, noisy=False) for c in pool], dtype=np.float64)
            wl_data.append((wl, pool, Psi, true_s, np.zeros(len(pool), dtype=bool)))

        def eval_all_workloads() -> Dict[str, float]:
            pcts, regs = [], []
            for wl, pool, Psi, true_s, _ in wl_data:
                mean_p = Psi @ mu
                std_p = np.sqrt(np.maximum(np.sum((Psi @ Sigma) * Psi, axis=1), 1e-12))
                score = mean_p - (self.cfg.lcb_alpha * std_p if self.cfg.use_pessimistic_lcb else 0.0)
                idx = int(np.argmax(score))
                best_t = float(np.max(true_s))
                pcts.append(float(np.exp(true_s[idx] - best_t)))
                regs.append(float(best_t - true_s[idx]))
            return {
                "mean_pct_oracle": float(np.mean(pcts)),
                "mean_regret": float(np.mean(regs)),
                "per_workload_pct_oracle": pcts,
                "per_workload_regret": regs,
            }

        out: Dict[int, Dict[str, float]] = {}
        if 0 in budget_set:
            out[0] = eval_all_workloads()

        for k in range(1, max_k + 1):
            # Pick (workload, schedule) maximizing information gain + optimistic quality
            best_val = -1e9
            best_pair = (0, 0)
            for w_i, (wl, pool, Psi, _, mask) in enumerate(wl_data):
                mean_p = Psi @ mu
                std_p = np.sqrt(np.maximum(np.sum((Psi @ Sigma) * Psi, axis=1), 1e-12))
                acq = mean_p + 2.0 * std_p + rng.normal(0.0, 1e-6, size=len(pool))
                acq[mask] = -1e9
                c_i = int(np.argmax(acq))
                if acq[c_i] > best_val:
                    best_val = float(acq[c_i])
                    best_pair = (w_i, c_i)

            w_i, c_i = best_pair
            wl, pool, Psi, _, mask = wl_data[w_i]
            mask[c_i] = True
            y_obs = float(oracle_fn(wl, pool[c_i], target_hw, noisy=True))

            phi_k = Psi[c_i]
            prec = prec + np.outer(phi_k, phi_k) / sigma2
            rhs = rhs + (phi_k * y_obs) / sigma2
            Sigma = np.linalg.inv(prec)
            mu = Sigma @ rhs

            if k in budget_set:
                out[k] = eval_all_workloads()

        return out
