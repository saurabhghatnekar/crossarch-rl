"""
Live-Silicon Multi-Step Sequential Schedule Transformation MDP (ScheduleGraphMDP).

Constructs a true finite-horizon MDP M_{x,h} = (S, A, P_h, r_h, gamma) directly over
verified real physical hardware measurements (NVIDIA T4/L4/A100/H100 Triton + Apple Metal).
At each step t, the agent observes ONLY its current schedule state s_t and a local
neighborhood of k_nbr single-axis/local schedule mutations (plus STOP), never the
full schedule table. Reaching the optimal schedule requires multi-step planning!
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from crossarch_rl.hardware import HardwareSpec
from crossarch_rl.schedule_mdp import ScheduleConfig
from crossarch_rl.workloads import WorkloadSpec


class ScheduleGraphMDP:
    """
    Real-hardware sequential schedule-editing MDP over verified measured schedules.

    Action space at state i (current schedule index in pool):
      0 .. k_nbr - 1 : transition to the m-th local schedule neighbor (sorted by L1 edit distance)
      k_nbr          : STOP (remain at current schedule, episode terminates)
    """

    def __init__(
        self,
        workload: WorkloadSpec,
        hw: HardwareSpec,
        pool: List[ScheduleConfig],
        true_scores: np.ndarray,
        k_nbr: int = 6,
        horizon: int = 8,
        gamma: float = 0.90,
        noise_std: float = 0.008,
        seed: int = 0,
    ) -> None:
        self.workload = workload
        self.hw = hw
        self.pool = pool
        self.n_states = len(pool)
        self.true_scores = np.asarray(true_scores, dtype=np.float64)
        self.best_true = float(np.max(self.true_scores))
        self.k_nbr = min(k_nbr, max(1, self.n_states - 1))
        self.stop_action = self.k_nbr
        self.num_actions = self.k_nbr + 1
        self.horizon = horizon
        self.gamma = gamma
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

        # Build local edit-distance adjacency graph over verified schedule tuples
        coords = np.array([c.as_tuple() for c in pool], dtype=np.float64)
        # Log-scale tile/warp axes so doubling/halving is unit step
        log_coords = coords.copy()
        for col in (0, 1, 2, 3, 5, 6):
            log_coords[:, col] = np.log2(np.maximum(coords[:, col], 1.0))
        self.neighbors: np.ndarray = np.zeros((self.n_states, self.k_nbr), dtype=np.int64)
        for i in range(self.n_states):
            dists = np.sum(np.abs(log_coords - log_coords[i]), axis=1)
            dists[i] = 1e9
            # Pick k_nbr closest single/local-axis mutations
            order = np.argsort(dists)[: self.k_nbr]
            self.neighbors[i] = order

        # Default poor/mid starting state (bottom-quartile or default heuristic config)
        # so multi-step planning is genuinely needed to climb to the global optimum!
        order_scores = np.argsort(self.true_scores)
        self.default_start_idx = int(order_scores[max(0, int(0.25 * self.n_states))])

        self.state_idx: int = self.default_start_idx
        self.steps: int = 0

    def next_state_idx(self, state_idx: int, action: int) -> int:
        if action == self.stop_action:
            return state_idx
        return int(self.neighbors[state_idx, action])

    def reset(self, start_idx: Optional[int] = None) -> int:
        self.state_idx = self.default_start_idx if start_idx is None else int(start_idx)
        self.steps = 0
        return self.state_idx

    def step(self, action: int, noisy: bool = True) -> Tuple[int, float, bool, Dict[str, float]]:
        self.steps += 1
        prev_idx = self.state_idx
        if action == self.stop_action:
            return (
                prev_idx,
                0.0,
                True,
                {
                    "true_score": float(self.true_scores[prev_idx]),
                    "pct_oracle": float(np.exp(self.true_scores[prev_idx] - self.best_true)),
                },
            )
        nxt_idx = int(self.neighbors[prev_idx, action])
        r_true = float(self.true_scores[nxt_idx] - self.true_scores[prev_idx])
        r_obs = r_true + (float(self.rng.normal(0.0, self.noise_std)) if noisy and self.noise_std > 0 else 0.0)
        self.state_idx = nxt_idx
        done = self.steps >= self.horizon
        return (
            nxt_idx,
            r_obs,
            done,
            {
                "true_score": float(self.true_scores[nxt_idx]),
                "pct_oracle": float(np.exp(self.true_scores[nxt_idx] - self.best_true)),
            },
        )


def collect_offline_trajectories(
    mdps: List[Tuple[str, ScheduleGraphMDP]],
    episodes_per_mdp: int = 18,
    seed: int = 0,
) -> List[Dict[str, object]]:
    """
    Generate realistic mixed-quality offline RL transitions (s, a, r, s', done) on source GPUs
    mixing exploratory random walks (50%) and noisy hill-climbing demonstrations (50%).
    """
    rng = np.random.default_rng(seed)
    transitions: List[Dict[str, object]] = []
    for hw_name, mdp in mdps:
        for ep in range(episodes_per_mdp):
            start_idx = int(rng.integers(0, mdp.n_states))
            s = mdp.reset(start_idx=start_idx)
            greedy_prob = 0.65 if (ep % 2 == 0) else 0.15
            for _ in range(mdp.horizon):
                if rng.random() < greedy_prob:
                    # Pick improving local mutation (noisy behavior policy)
                    cand_scores = [mdp.true_scores[mdp.next_state_idx(s, a)] for a in range(mdp.num_actions)]
                    a = int(np.argmax(cand_scores))
                else:
                    a = int(rng.integers(0, mdp.num_actions))
                s_next, r, done, _ = mdp.step(a, noisy=True)
                transitions.append(
                    {
                        "hw_name": hw_name,
                        "hw": mdp.hw,
                        "wl": mdp.workload,
                        "mdp": mdp,
                        "s_idx": s,
                        "a": a,
                        "r": r,
                        "s_next_idx": s_next,
                        "done": float(done),
                    }
                )
                s = s_next
                if done:
                    break
    return transitions
