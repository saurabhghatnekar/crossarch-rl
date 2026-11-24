"""
True Multi-Step Sequential RL Algorithms on Live-Silicon ScheduleGraphMDP:

  1. SequentialCrossArchRL (TD-SF + GPI):
     - Trains hardware-invariant step cumulant features phi_theta(s, a; h) = Psi(s') - Psi(s)
     - Trains multi-step Successor Feature networks SF_omega^{pi_j}(s, a; h) via true
       Bellman temporal-difference backups with Polyak target networks across source policies
     - On unseen target GPU: runs multi-step GPI rollouts with closed-form Bayesian posterior
       updates over target hardware weights w_{h_tgt} from observed step rewards!

  2. OfflineIQL (Kostrikov et al., ICLR 2022):
     - Expectile value V_psi(s) + TD Q-network Q_phi(s, a) + online target fine-tuning.

  3. OfflineCQL (Kumar et al., NeurIPS 2020):
     - Conservative Q-Learning with log-sum-exp OOD action penalty + TD Bellman backup
       + online target fine-tuning.

  4. OnlinePPO (Schulman et al., 2017):
     - Clipped policy gradient + GAE trained online from scratch on target-GPU rollouts.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from crossarch_rl.hardware import HardwareSpec
from crossarch_rl.schedule_mdp import (
    PHYS_BASIS_DIM,
    extract_normalized_state_features,
    extract_physics_basis,
)
from crossarch_rl.seq_mdp import ScheduleGraphMDP


def precompute_mdp_features(mdp: ScheduleGraphMDP) -> Tuple[np.ndarray, np.ndarray]:
    """Return (Z_states [N, 27], PHI_states [N, 12]) for all verified schedules in mdp."""
    Z = np.stack([extract_normalized_state_features(mdp.workload, c, mdp.hw) for c in mdp.pool], axis=0).astype(np.float32)
    PHI = np.stack([extract_physics_basis(mdp.workload, c, mdp.hw) for c in mdp.pool], axis=0).astype(np.float32)
    return Z, PHI


class CumulantBasisNet(nn.Module):
    """Learns bounded hardware-invariant state potential residual psi_theta(z) in [-1, 1]^{d_res}."""

    def __init__(self, in_dim: int, hidden: int = 96, res_dim: int = 12) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, res_dim),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class SuccessorFeatureNet(nn.Module):
    """
    Multi-step TD Successor Feature Network: maps (z_s, z_s_next) -> Psi^{pi_j}(s, a) in R^d
    representing E_{pi_j}[ sum_{t=0}^{H} gamma^t phi(s_t, a_t) | s_0=s, a_0=a ].
    """

    def __init__(self, state_dim: int, basis_dim: int, hidden: int = 96) -> None:
        super().__init__()
        # Input: [z_s, z_s_next - z_s] (transition representation)
        self.net = nn.Sequential(
            nn.Linear(state_dim * 2, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, basis_dim),
        )

    def forward(self, z_s: torch.Tensor, z_next: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_s, z_next - z_s], dim=-1))


class SequentialCrossArchRL:
    """
    True Multi-Step TD Successor Features + Generalized Policy Improvement (GPI) +
    Datasheet-Prior Bayesian Linear Adaptation on ScheduleGraphMDP.
    """

    def __init__(
        self,
        res_dim: int = 12,
        gamma: float = 0.90,
        obs_noise_std: float = 0.09,
        prior_dist_scale: float = 0.20,
        seed: int = 0,
    ) -> None:
        self.res_dim = res_dim
        self.basis_dim = PHYS_BASIS_DIM + res_dim
        self.gamma = gamma
        self.obs_noise_std = obs_noise_std
        self.prior_dist_scale = prior_dist_scale
        self.seed = seed

        self.feat_mean: np.ndarray | None = None
        self.feat_std: np.ndarray | None = None
        self.cumulant_net: CumulantBasisNet | None = None
        self.sf_heads: List[SuccessorFeatureNet] = []  # One TD-SF head per source policy pi_j
        self.source_weights: Dict[str, np.ndarray] = {}
        self.source_specs: Dict[str, HardwareSpec] = {}
        self.prior_ridge_W: np.ndarray | None = None

    def _norm_z(self, Z: np.ndarray) -> np.ndarray:
        return (Z - self.feat_mean) / self.feat_std

    def _state_psi_np(self, Z: np.ndarray, PHI: np.ndarray) -> np.ndarray:
        assert self.cumulant_net is not None
        Zn = torch.from_numpy(self._norm_z(Z).astype(np.float32))
        with torch.no_grad():
            res = self.cumulant_net(Zn).numpy().astype(np.float64)
        return np.concatenate([PHI.astype(np.float64), res], axis=1)

    def fit_offline_td(
        self,
        transitions: List[Dict[str, object]],
        cumulant_epochs: int = 90,
        td_epochs: int = 70,
        batch_size: int = 256,
    ) -> Dict[str, float]:
        """
        Two-stage offline RL training on source-GPU sequential trajectories:
          Stage 1: Learn step cumulants phi_theta(s, a) = Psi(s') - Psi(s) & source weights w_{h_m}
          Stage 2: Train multi-step Successor Feature networks SF^{pi_j}(s, a) via Bellman TD backups!
        """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        hw_names = sorted({str(tr["hw_name"]) for tr in transitions})
        hw_to_idx = {n: i for i, n in enumerate(hw_names)}
        self.source_specs = {str(tr["hw_name"]): tr["hw"] for tr in transitions}  # type: ignore

        # Precompute per-MDP feature tables once
        mdp_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for tr in transitions:
            mdp = tr["mdp"]
            if id(mdp) not in mdp_cache:
                mdp_cache[id(mdp)] = precompute_mdp_features(mdp)  # type: ignore

        z_s_l, z_n_l, phi_s_l, phi_n_l, r_l, d_l, h_l = [], [], [], [], [], [], []
        # Also store all neighbor next-state features for exact Bellman max/policy bootstrap
        nbr_z_l = []
        for tr in transitions:
            mdp: ScheduleGraphMDP = tr["mdp"]  # type: ignore
            Z, PHI = mdp_cache[id(mdp)]
            si, sni = int(tr["s_idx"]), int(tr["s_next_idx"])
            z_s_l.append(Z[si])
            z_n_l.append(Z[sni])
            phi_s_l.append(PHI[si])
            phi_n_l.append(PHI[sni])
            r_l.append(float(tr["r"]))
            d_l.append(float(tr["done"]))
            h_l.append(hw_to_idx[str(tr["hw_name"])])
            # All next states reachable from s_next (including STOP = s_next)
            next_actions = [mdp.next_state_idx(sni, a) for a in range(mdp.num_actions)]
            nbr_z_l.append(Z[next_actions])

        Zs = np.stack(z_s_l, axis=0)
        Zn = np.stack(z_n_l, axis=0)
        PHIs = np.stack(phi_s_l, axis=0)
        PHIn = np.stack(phi_n_l, axis=0)
        R = np.array(r_l, dtype=np.float32)
        D = np.array(d_l, dtype=np.float32)
        H_idx = np.array(h_l, dtype=np.int64)
        Z_nbr = np.stack(nbr_z_l, axis=0)  # (N, num_actions, state_dim)

        self.feat_mean = Zs.mean(axis=0)
        self.feat_std = Zs.std(axis=0) + 1e-5
        Zs_n = torch.from_numpy(self._norm_z(Zs).astype(np.float32))
        Zn_n = torch.from_numpy(self._norm_z(Zn).astype(np.float32))
        Z_nbr_n = torch.from_numpy(self._norm_z(Z_nbr).astype(np.float32))
        dPHI = torch.from_numpy((PHIn - PHIs).astype(np.float32))
        R_t = torch.from_numpy(R)
        D_t = torch.from_numpy(D)
        H_t = torch.from_numpy(H_idx)

        state_dim = Zs.shape[1]
        n_sources = len(hw_names)
        N = Zs.shape[0]

        # ---- Stage 1: Step-Cumulant Representation Learning ----
        self.cumulant_net = CumulantBasisNet(state_dim, hidden=96, res_dim=self.res_dim)
        W_src = nn.Parameter(torch.zeros((n_sources, self.basis_dim), dtype=torch.float32))
        opt1 = torch.optim.AdamW(list(self.cumulant_net.parameters()) + [W_src], lr=3e-3, weight_decay=1e-4)

        cum_loss_val = 0.0
        for _ in range(cumulant_epochs):
            perm = torch.randperm(N)
            for st in range(0, N, batch_size):
                idx = perm[st : st + batch_size]
                res_s = self.cumulant_net(Zs_n[idx])
                res_n = self.cumulant_net(Zn_n[idx])
                phi_step = torch.cat([dPHI[idx], res_n - res_s], dim=1)  # cumulant phi(s, a)
                pred_r = (phi_step * W_src[H_t[idx]]).sum(dim=1)
                loss = F.mse_loss(pred_r, R_t[idx]) + 2e-3 * (W_src ** 2).mean()
                opt1.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self.cumulant_net.parameters()) + [W_src], 5.0)
                opt1.step()
                cum_loss_val = float(loss.item())

        # Freeze cumulant network and compute exact step cumulants phi(s, a) for TD learning
        with torch.no_grad():
            all_phi_step = torch.cat(
                [dPHI, self.cumulant_net(Zn_n) - self.cumulant_net(Zs_n)], dim=1
            )  # (N, d)
            W_np = W_src.detach().numpy().astype(np.float64)
            self.source_weights = {name: W_np[i] for i, name in enumerate(hw_names)}

        # Fit datasheet prior mapper u_h -> w_h
        if n_sources >= 2:
            U = np.stack([np.array(self.source_specs[n].datasheet_vector() + [1.0]) for n in hw_names], axis=0)
            W_mat = np.stack([self.source_weights[n] for n in hw_names], axis=0)
            reg = 10.0 * np.eye(U.shape[1])
            reg[-1, -1] = 1e-2
            self.prior_ridge_W = np.linalg.solve(U.T @ U + reg, U.T @ W_mat)
        else:
            self.prior_ridge_W = None

        # ---- Stage 2: Multi-Step Bellman TD Backup for Each Source Policy pi_j ----
        self.sf_heads = []
        td_loss_val = 0.0
        n_act = Z_nbr_n.shape[1]

        for j in range(n_sources):
            sf_net = SuccessorFeatureNet(state_dim, self.basis_dim, hidden=96)
            sf_target = copy.deepcopy(sf_net)
            w_j = torch.from_numpy(W_np[j].astype(np.float32))  # (d,)
            opt_td = torch.optim.AdamW(sf_net.parameters(), lr=3e-3, weight_decay=1e-4)

            for _ in range(td_epochs):
                perm = torch.randperm(N)
                for st in range(0, N, batch_size):
                    idx = perm[st : st + batch_size]
                    bs = len(idx)
                    sf_pred = sf_net(Zs_n[idx], Zn_n[idx])  # (B, d)

                    with torch.no_grad():
                        # Evaluate all valid next actions from s_next under target SF network
                        zn_rep = Zn_n[idx].unsqueeze(1).expand(bs, n_act, state_dim).reshape(bs * n_act, state_dim)
                        znbr_flat = Z_nbr_n[idx].reshape(bs * n_act, state_dim)
                        sf_next_all = sf_target(zn_rep, znbr_flat).reshape(bs, n_act, self.basis_dim)
                        # Greedy action of source policy pi_j: a* = argmax_a SF^{pi_j}(s', a)^T w_j
                        q_next_j = torch.einsum("bad,d->ba", sf_next_all, w_j)
                        best_a = q_next_j.argmax(dim=1)  # (B,)
                        sf_next_star = sf_next_all[torch.arange(bs), best_a]  # (B, d)
                        td_target = all_phi_step[idx] + self.gamma * (1.0 - D_t[idx].unsqueeze(1)) * sf_next_star

                    td_loss = F.mse_loss(sf_pred, td_target)
                    opt_td.zero_grad()
                    td_loss.backward()
                    torch.nn.utils.clip_grad_norm_(sf_net.parameters(), 5.0)
                    opt_td.step()
                    td_loss_val = float(td_loss.item())

                    # Polyak target update
                    for p, pt in zip(sf_net.parameters(), sf_target.parameters()):
                        pt.data.mul_(0.95).add_(p.data, alpha=0.05)

            self.sf_heads.append(sf_net)

        return {"cumulant_loss": cum_loss_val, "td_loss": td_loss_val, "n_sf_heads": float(len(self.sf_heads))}

    def target_prior(self, target_hw: HardwareSpec) -> Tuple[np.ndarray, np.ndarray]:
        d = self.basis_dim
        diag = np.full(d, 0.60 ** 2, dtype=np.float64)
        diag[:PHYS_BASIS_DIM] = 0.45 ** 2
        u_tgt = np.array(target_hw.datasheet_vector(), dtype=np.float64)
        min_dist_sq = min(
            float(np.mean((u_tgt - np.array(spec.datasheet_vector(), dtype=np.float64)) ** 2))
            for spec in self.source_specs.values()
        )
        Sigma_0 = np.diag(diag * (1.0 + self.prior_dist_scale * min_dist_sq))
        mean_src = np.mean(list(self.source_weights.values()), axis=0)
        if self.prior_ridge_W is not None:
            u = np.array(target_hw.datasheet_vector() + [1.0], dtype=np.float64)
            delta = np.clip((u @ self.prior_ridge_W) - mean_src, -0.5, 0.5)
            mu_0 = mean_src + 0.55 * delta
        else:
            mu_0 = mean_src / (1.0 + 0.15 * min_dist_sq)
        return mu_0, Sigma_0

    def evaluate_target_mdps(
        self,
        target_hw: HardwareSpec,
        mdps: List[ScheduleGraphMDP],
        episode_budgets: Sequence[int] = (0, 1, 2, 3, 5),
        rng_seed: int = 0,
    ) -> Dict[int, List[float]]:
        """
        Run multi-step GPI rollouts on unseen target GPU MDPs!
        - Episode 0 (zero-shot): pure GPI rollout using datasheet prior mu_0.
        - Episodes 1..max_ep: execute multi-step GPI-UCB rollouts on target GPU, updating the
          Bayesian linear posterior (mu, Sigma) after every observed step reward (s_t, a_t, r_t),
          then evaluate the adapted greedy GPI rollout from the default starting schedule!
        """
        rng = np.random.default_rng(rng_seed)
        mu_0, Sigma_0 = self.target_prior(target_hw)
        Sigma_0_inv = np.linalg.inv(Sigma_0)
        sigma2 = self.obs_noise_std ** 2
        max_ep = max(episode_budgets)

        out: Dict[int, List[float]] = {e: [] for e in episode_budgets}

        for mdp in mdps:
            Z, PHI = precompute_mdp_features(mdp)
            Zn_t = torch.from_numpy(self._norm_z(Z).astype(np.float32))
            Psi_states = self._state_psi_np(Z, PHI)  # (N, d)

            # Precompute TD successor-feature tensors SF^{pi_j}(s, a) for local transitions in this MDP
            # (computed on-the-fly per visited state in real deployment; cached here only for speed)
            def gpi_q_values(s_idx: int, mu: np.ndarray, Sigma: np.ndarray, beta_ucb: float = 0.0) -> np.ndarray:
                nbrs = [mdp.next_state_idx(s_idx, a) for a in range(mdp.num_actions)]
                zs_rep = Zn_t[s_idx].unsqueeze(0).expand(len(nbrs), -1)
                zn_batch = Zn_t[nbrs]
                q_heads = []
                with torch.no_grad():
                    for head in self.sf_heads:
                        sf = head(zs_rep, zn_batch).numpy().astype(np.float64)  # (A, d)
                        mean_q = sf @ mu
                        if beta_ucb != 0.0:
                            std_q = np.sqrt(np.maximum(np.sum((sf @ Sigma) * sf, axis=1), 1e-12))
                            q_heads.append(mean_q + beta_ucb * std_q)
                        else:
                            q_heads.append(mean_q)
                # Generalized Policy Improvement: max_j Q^{pi_j}(s, a)
                return np.max(np.stack(q_heads, axis=0), axis=0)

            def eval_greedy_rollout(mu: np.ndarray, Sigma: np.ndarray) -> float:
                s = mdp.reset()
                best_seen = mdp.true_scores[s]
                visited = {s}
                for _ in range(mdp.horizon):
                    q = gpi_q_values(s, mu, Sigma, beta_ucb=0.0)
                    # Prevent trivial 2-cycles during evaluation
                    for a in range(mdp.k_nbr):
                        if mdp.next_state_idx(s, a) in visited:
                            q[a] -= 1e3
                    a = int(np.argmax(q))
                    s, _, done, info = mdp.step(a, noisy=False)
                    visited.add(s)
                    best_seen = max(best_seen, info["true_score"])
                    if done:
                        break
                return float(np.exp(best_seen - mdp.best_true))

            prec = Sigma_0_inv.copy()
            rhs = Sigma_0_inv @ mu_0
            mu = mu_0.copy()
            Sigma = Sigma_0.copy()

            if 0 in out:
                out[0].append(eval_greedy_rollout(mu, Sigma))

            best_ep_score = mdp.true_scores[mdp.default_start_idx]
            for ep in range(1, max_ep + 1):
                start_s = mdp.default_start_idx if ep == 1 else int(rng.integers(0, mdp.n_states))
                s = mdp.reset(start_idx=start_s)
                visited = {s}
                for _ in range(mdp.horizon):
                    q_ucb = gpi_q_values(s, mu, Sigma, beta_ucb=0.8) + rng.normal(0.0, 1e-5, size=mdp.num_actions)
                    for a in range(mdp.k_nbr):
                        if mdp.next_state_idx(s, a) in visited:
                            q_ucb[a] -= 1e3
                    a = int(np.argmax(q_ucb))
                    s_next, r_obs, done, info = mdp.step(a, noisy=True)
                    best_ep_score = max(best_ep_score, info["true_score"])
                    visited.add(s_next)
                    if s_next != s:
                        # Bayesian linear posterior update from observed step cumulant phi(s, a) = Psi(s') - Psi(s)
                        phi_sa = Psi_states[s_next] - Psi_states[s]
                        prec = prec + np.outer(phi_sa, phi_sa) / sigma2
                        rhs = rhs + (phi_sa * r_obs) / sigma2
                        Sigma = np.linalg.inv(prec)
                        mu = Sigma @ rhs
                    s = s_next
                    if done:
                        break

                if ep in out:
                    # Deploy better of adapted greedy GPI rollout or best schedule discovered during rollouts
                    gpi_pct = eval_greedy_rollout(mu, Sigma)
                    disc_pct = float(np.exp(best_ep_score - mdp.best_true))
                    out[ep].append(max(gpi_pct, disc_pct))
        return out


class _TransitionQNet(nn.Module):
    """Q(s, a) taking normalized [z_s, z_s_next - z_s, u_h] -> scalar Q-value."""

    def __init__(self, in_dim: int, hidden: int = 96) -> None:
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


class _StateVNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 96) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class OfflineIQLAgent:
    """Offline Implicit Q-Learning (Kostrikov et al., ICLR 2022) + online target fine-tuning."""

    def __init__(self, gamma: float = 0.90, expectile: float = 0.75, seed: int = 0) -> None:
        self.gamma = gamma
        self.expectile = expectile
        self.seed = seed
        self.q_net: _TransitionQNet | None = None
        self.v_net: _StateVNet | None = None
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def _state_vec(self, Z: np.ndarray, hw: HardwareSpec) -> np.ndarray:
        u = np.broadcast_to(np.array(hw.datasheet_vector(), dtype=np.float32), (Z.shape[0], 10))
        return np.concatenate([Z, u], axis=1)

    def fit_offline(self, transitions: List[Dict[str, object]], epochs: int = 75, batch_size: int = 256) -> None:
        torch.manual_seed(self.seed)
        mdp_cache: Dict[int, np.ndarray] = {}
        for tr in transitions:
            mdp = tr["mdp"]
            if id(mdp) not in mdp_cache:
                Z, _ = precompute_mdp_features(mdp)  # type: ignore
                mdp_cache[id(mdp)] = self._state_vec(Z, mdp.hw)  # type: ignore

        xs, xn, rs, ds = [], [], [], []
        for tr in transitions:
            S = mdp_cache[id(tr["mdp"])]
            xs.append(S[int(tr["s_idx"])])
            xn.append(S[int(tr["s_next_idx"])])
            rs.append(float(tr["r"]))
            ds.append(float(tr["done"]))

        Xs = np.stack(xs, axis=0)
        Xn = np.stack(xn, axis=0)
        self.mean = Xs.mean(axis=0)
        self.std = Xs.std(axis=0) + 1e-5
        Xs_t = torch.from_numpy(((Xs - self.mean) / self.std).astype(np.float32))
        Xn_t = torch.from_numpy(((Xn - self.mean) / self.std).astype(np.float32))
        R_t = torch.from_numpy(np.array(rs, dtype=np.float32))
        D_t = torch.from_numpy(np.array(ds, dtype=np.float32))

        sdim = Xs.shape[1]
        self.q_net = _TransitionQNet(sdim * 2, hidden=96)
        q_targ = copy.deepcopy(self.q_net)
        self.v_net = _StateVNet(sdim, hidden=96)
        opt_q = torch.optim.AdamW(self.q_net.parameters(), lr=3e-3, weight_decay=1e-4)
        opt_v = torch.optim.AdamW(self.v_net.parameters(), lr=3e-3, weight_decay=1e-4)

        N = Xs.shape[0]
        tau = self.expectile
        for _ in range(epochs):
            perm = torch.randperm(N)
            for st in range(0, N, batch_size):
                idx = perm[st : st + batch_size]
                qa_in = torch.cat([Xs_t[idx], Xn_t[idx] - Xs_t[idx]], dim=1)
                with torch.no_grad():
                    q_val = q_targ(qa_in)
                    v_next = self.v_net(Xn_t[idx])
                # IQL Expectile V-loss
                v_pred = self.v_net(Xs_t[idx])
                diff = q_val - v_pred
                w = torch.where(diff > 0, tau, 1.0 - tau)
                loss_v = (w * (diff ** 2)).mean()
                opt_v.zero_grad()
                loss_v.backward()
                opt_v.step()

                # IQL Bellman Q-loss using V(s')
                q_pred = self.q_net(qa_in)
                q_target = R_t[idx] + self.gamma * (1.0 - D_t[idx]) * v_next
                loss_q = F.mse_loss(q_pred, q_target)
                opt_q.zero_grad()
                loss_q.backward()
                opt_q.step()

                for p, pt in zip(self.q_net.parameters(), q_targ.parameters()):
                    pt.data.mul_(0.95).add_(p.data, alpha=0.05)

    def evaluate_target_mdps(
        self,
        target_hw: HardwareSpec,
        mdps: List[ScheduleGraphMDP],
        episode_budgets: Sequence[int] = (0, 1, 2, 3, 5),
        rng_seed: int = 0,
    ) -> Dict[int, List[float]]:
        assert self.q_net is not None and self.v_net is not None
        rng = np.random.default_rng(rng_seed)
        max_ep = max(episode_budgets)
        out: Dict[int, List[float]] = {e: [] for e in episode_budgets}

        for mdp in mdps:
            Z, _ = precompute_mdp_features(mdp)
            S = self._state_vec(Z, target_hw)
            Sn_t = torch.from_numpy(((S - self.mean) / self.std).astype(np.float32))
            q_loc = copy.deepcopy(self.q_net)
            v_loc = copy.deepcopy(self.v_net)
            opt_q = torch.optim.Adam(q_loc.parameters(), lr=2e-3)
            opt_v = torch.optim.Adam(v_loc.parameters(), lr=2e-3)

            def rollout(eps_explore: float = 0.0) -> Tuple[float, List[Tuple[int, int, float, float]]]:
                s = mdp.reset()
                best_s = mdp.true_scores[s]
                visited = {s}
                buf = []
                for _ in range(mdp.horizon):
                    nbrs = [mdp.next_state_idx(s, a) for a in range(mdp.num_actions)]
                    if eps_explore > 0 and rng.random() < eps_explore:
                        a = int(rng.integers(0, mdp.num_actions))
                    else:
                        with torch.no_grad():
                            zs = Sn_t[s].unsqueeze(0).expand(len(nbrs), -1)
                            zn = Sn_t[nbrs]
                            qv = q_loc(torch.cat([zs, zn - zs], dim=1)).numpy()
                        for ai in range(mdp.k_nbr):
                            if nbrs[ai] in visited:
                                qv[ai] -= 1e3
                        a = int(np.argmax(qv))
                    sn, r, done, info = mdp.step(a, noisy=(eps_explore > 0))
                    buf.append((s, sn, r, float(done)))
                    visited.add(sn)
                    best_s = max(best_s, info["true_score"])
                    s = sn
                    if done:
                        break
                return float(np.exp(best_s - mdp.best_true)), buf

            if 0 in out:
                pct, _ = rollout(eps_explore=0.0)
                out[0].append(pct)

            replay: List[Tuple[int, int, float, float]] = []
            best_disc = mdp.true_scores[mdp.default_start_idx]
            for ep in range(1, max_ep + 1):
                pct_ep, buf = rollout(eps_explore=0.25)
                best_disc = max(best_disc, mdp.best_true + np.log(max(pct_ep, 1e-6)))
                replay.extend(buf)
                # Online IQL TD fine-tuning on collected target transitions
                si = torch.tensor([b[0] for b in replay], dtype=torch.long)
                sni = torch.tensor([b[1] for b in replay], dtype=torch.long)
                rb = torch.tensor([b[2] for b in replay], dtype=torch.float32)
                db = torch.tensor([b[3] for b in replay], dtype=torch.float32)
                qa_in = torch.cat([Sn_t[si], Sn_t[sni] - Sn_t[si]], dim=1)
                for _ in range(15):
                    with torch.no_grad():
                        qv = q_loc(qa_in)
                        vn = v_loc(Sn_t[sni])
                    vp = v_loc(Sn_t[si])
                    diff = qv - vp
                    lv = (torch.where(diff > 0, self.expectile, 1.0 - self.expectile) * diff ** 2).mean()
                    opt_v.zero_grad(); lv.backward(); opt_v.step()
                    lq = F.mse_loss(q_loc(qa_in), rb + self.gamma * (1.0 - db) * vn)
                    opt_q.zero_grad(); lq.backward(); opt_q.step()
                if ep in out:
                    p_eval, _ = rollout(eps_explore=0.0)
                    out[ep].append(max(p_eval, float(np.exp(best_disc - mdp.best_true))))
        return out


class OfflineCQLAgent(OfflineIQLAgent):
    """Offline Conservative Q-Learning (Kumar et al., NeurIPS 2020) + online fine-tuning."""

    def __init__(self, gamma: float = 0.90, cql_alpha: float = 0.25, seed: int = 0) -> None:
        super().__init__(gamma=gamma, seed=seed)
        self.cql_alpha = cql_alpha

    def fit_offline(self, transitions: List[Dict[str, object]], epochs: int = 75, batch_size: int = 256) -> None:
        torch.manual_seed(self.seed)
        mdp_cache: Dict[int, np.ndarray] = {}
        for tr in transitions:
            mdp = tr["mdp"]
            if id(mdp) not in mdp_cache:
                Z, _ = precompute_mdp_features(mdp)  # type: ignore
                mdp_cache[id(mdp)] = self._state_vec(Z, mdp.hw)  # type: ignore

        xs, xn, rs, ds, xnbr_s, xnbr_n = [], [], [], [], [], []
        for tr in transitions:
            mdp: ScheduleGraphMDP = tr["mdp"]  # type: ignore
            S = mdp_cache[id(mdp)]
            si, sni = int(tr["s_idx"]), int(tr["s_next_idx"])
            xs.append(S[si]); xn.append(S[sni])
            rs.append(float(tr["r"])); ds.append(float(tr["done"]))
            xnbr_s.append(S[[mdp.next_state_idx(si, a) for a in range(mdp.num_actions)]])
            xnbr_n.append(S[[mdp.next_state_idx(sni, a) for a in range(mdp.num_actions)]])

        Xs = np.stack(xs, axis=0)
        Xn = np.stack(xn, axis=0)
        self.mean = Xs.mean(axis=0)
        self.std = Xs.std(axis=0) + 1e-5
        Xs_t = torch.from_numpy(((Xs - self.mean) / self.std).astype(np.float32))
        Xn_t = torch.from_numpy(((Xn - self.mean) / self.std).astype(np.float32))
        Xns_t = torch.from_numpy(((np.stack(xnbr_s) - self.mean) / self.std).astype(np.float32))
        Xnn_t = torch.from_numpy(((np.stack(xnbr_n) - self.mean) / self.std).astype(np.float32))
        R_t = torch.from_numpy(np.array(rs, dtype=np.float32))
        D_t = torch.from_numpy(np.array(ds, dtype=np.float32))

        sdim = Xs.shape[1]
        n_act = Xns_t.shape[1]
        self.q_net = _TransitionQNet(sdim * 2, hidden=96)
        q_targ = copy.deepcopy(self.q_net)
        self.v_net = _StateVNet(sdim, hidden=96)  # unused in CQL; kept for shared eval interface
        opt = torch.optim.AdamW(self.q_net.parameters(), lr=3e-3, weight_decay=1e-4)

        N = Xs.shape[0]
        for _ in range(epochs):
            perm = torch.randperm(N)
            for st in range(0, N, batch_size):
                idx = perm[st : st + batch_size]
                bs = len(idx)
                q_data = self.q_net(torch.cat([Xs_t[idx], Xn_t[idx] - Xs_t[idx]], dim=1))
                # CQL log-sum-exp over all local actions at s
                xs_rep = Xs_t[idx].unsqueeze(1).expand(bs, n_act, sdim)
                q_all_s = self.q_net(torch.cat([xs_rep, Xns_t[idx] - xs_rep], dim=-1).reshape(bs * n_act, sdim * 2)).reshape(bs, n_act)
                cql_pen = (torch.logsumexp(q_all_s, dim=1) - q_data).mean()

                with torch.no_grad():
                    xn_rep = Xn_t[idx].unsqueeze(1).expand(bs, n_act, sdim)
                    q_all_n = q_targ(torch.cat([xn_rep, Xnn_t[idx] - xn_rep], dim=-1).reshape(bs * n_act, sdim * 2)).reshape(bs, n_act)
                    q_next_max = q_all_n.max(dim=1).values
                    td_targ = R_t[idx] + self.gamma * (1.0 - D_t[idx]) * q_next_max

                loss = F.mse_loss(q_data, td_targ) + self.cql_alpha * cql_pen
                opt.zero_grad(); loss.backward(); opt.step()
                for p, pt in zip(self.q_net.parameters(), q_targ.parameters()):
                    pt.data.mul_(0.95).add_(p.data, alpha=0.05)


class OnlinePPOAgent:
    """Online PPO (Schulman et al., 2017) with GAE trained from scratch on target-GPU rollouts."""

    def __init__(self, gamma: float = 0.90, lam: float = 0.95, clip_eps: float = 0.2) -> None:
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps

    def evaluate_target_mdps(
        self,
        target_hw: HardwareSpec,
        mdps: List[ScheduleGraphMDP],
        episode_budgets: Sequence[int] = (0, 1, 2, 3, 5),
        rng_seed: int = 0,
    ) -> Dict[int, List[float]]:
        torch.manual_seed(rng_seed)
        np.random.seed(rng_seed)
        max_ep = max(episode_budgets)
        out: Dict[int, List[float]] = {e: [] for e in episode_budgets}

        for mdp in mdps:
            Z, _ = precompute_mdp_features(mdp)
            Zn = (Z - Z.mean(axis=0)) / (Z.std(axis=0) + 1e-5)
            Zn_t = torch.from_numpy(Zn.astype(np.float32))
            sdim = Zn.shape[1]

            actor = nn.Sequential(nn.Linear(sdim, 64), nn.Tanh(), nn.Linear(64, mdp.num_actions))
            critic = nn.Sequential(nn.Linear(sdim, 64), nn.Tanh(), nn.Linear(64, 1))
            opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=5e-3)

            def eval_greedy() -> float:
                s = mdp.reset()
                best_s = mdp.true_scores[s]
                visited = {s}
                for _ in range(mdp.horizon):
                    with torch.no_grad():
                        logits = actor(Zn_t[s]).numpy()
                    for ai in range(mdp.k_nbr):
                        if mdp.next_state_idx(s, ai) in visited:
                            logits[ai] -= 1e3
                    a = int(np.argmax(logits))
                    s, _, done, info = mdp.step(a, noisy=False)
                    visited.add(s)
                    best_s = max(best_s, info["true_score"])
                    if done:
                        break
                return float(np.exp(best_s - mdp.best_true))

            if 0 in out:
                out[0].append(eval_greedy())

            best_disc = mdp.true_scores[mdp.default_start_idx]
            for ep in range(1, max_ep + 1):
                s = mdp.reset()
                states, actions, logps, rewards, vals = [], [], [], [], []
                for _ in range(mdp.horizon):
                    dist = torch.distributions.Categorical(logits=actor(Zn_t[s]))
                    a = dist.sample()
                    lp = dist.log_prob(a)
                    v = critic(Zn_t[s]).squeeze(-1)
                    sn, r, done, info = mdp.step(int(a.item()), noisy=True)
                    best_disc = max(best_disc, info["true_score"])
                    states.append(s); actions.append(a); logps.append(lp.detach()); rewards.append(r); vals.append(v.detach())
                    s = sn
                    if done:
                        break
                # Compute GAE returns
                returns, advs = [], []
                gae = 0.0
                next_v = 0.0
                for t in reversed(range(len(rewards))):
                    delta = rewards[t] + self.gamma * next_v - float(vals[t].item())
                    gae = delta + self.gamma * self.lam * gae
                    advs.insert(0, gae)
                    returns.insert(0, gae + float(vals[t].item()))
                    next_v = float(vals[t].item())
                st_t = Zn_t[states]
                ac_t = torch.stack(actions)
                old_lp = torch.stack(logps)
                adv_t = torch.tensor(advs, dtype=torch.float32)
                if adv_t.numel() > 1:
                    adv_t = (adv_t - adv_t.mean()) / (adv_t.std(unbiased=False) + 1e-6)
                ret_t = torch.tensor(returns, dtype=torch.float32)

                for _ in range(8):
                    dist = torch.distributions.Categorical(logits=actor(st_t))
                    ratio = torch.exp(dist.log_prob(ac_t) - old_lp)
                    surr = torch.min(ratio * adv_t, torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_t)
                    v_loss = F.mse_loss(critic(st_t).squeeze(-1), ret_t)
                    loss = -surr.mean() + 0.5 * v_loss - 0.01 * dist.entropy().mean()
                    opt.zero_grad(); loss.backward(); opt.step()

                if ep in out:
                    out[ep].append(max(eval_greedy(), float(np.exp(best_disc - mdp.best_true))))
        return out


class RandomWalkMDPBaseline:
    """Random walk over the local ScheduleGraphMDP neighborhood."""

    def evaluate_target_mdps(
        self,
        target_hw: HardwareSpec,
        mdps: List[ScheduleGraphMDP],
        episode_budgets: Sequence[int] = (0, 1, 2, 3, 5),
        rng_seed: int = 0,
    ) -> Dict[int, List[float]]:
        rng = np.random.default_rng(rng_seed)
        max_ep = max(episode_budgets)
        out: Dict[int, List[float]] = {e: [] for e in episode_budgets}
        for mdp in mdps:
            if 0 in out:
                out[0].append(float(np.exp(mdp.true_scores[mdp.default_start_idx] - mdp.best_true)))
            best_s = mdp.true_scores[mdp.default_start_idx]
            for ep in range(1, max_ep + 1):
                s = mdp.reset()
                for _ in range(mdp.horizon):
                    a = int(rng.integers(0, mdp.k_nbr))
                    s, _, done, info = mdp.step(a, noisy=True)
                    best_s = max(best_s, info["true_score"])
                    if done:
                        break
                if ep in out:
                    out[ep].append(float(np.exp(best_s - mdp.best_true)))
        return out
