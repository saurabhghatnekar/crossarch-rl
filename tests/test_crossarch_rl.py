"""Unit tests for CrossArch-RL MDP, hardware specs, Bayesian updates, and baselines."""

from __future__ import annotations

import numpy as np

from crossarch_rl.algorithms.baselines import (
    DefaultHeuristicTuner,
    EvolutionarySearchTuner,
    GPBayesianOptTuner,
    NeuralUCBTuner,
    RandomSearchTuner,
    ZeroShotTransferMLPTuner,
)
from crossarch_rl.algorithms.crossarch import CrossArchConfig, CrossArchRL
from crossarch_rl.eval_stats import interquartile_mean, probability_of_improvement, stratified_bootstrap_iqm
from crossarch_rl.hardware import HARDWARE_CATALOG, get_hardware
from crossarch_rl.schedule_mdp import (
    PHYS_BASIS_DIM,
    ScheduleMDP,
    check_feasibility,
    enumerate_feasible_schedules,
    extract_normalized_state_features,
    extract_physics_basis,
)
from crossarch_rl.simulator import CalibratedHardwareSimulator
from crossarch_rl.workloads import default_llm_workloads


def test_hardware_support_shift() -> None:
    """Verify genuine feasible-schedule support shift across Apple M2 Max (32KB) and A100 (164KB)."""
    m2 = get_hardware("apple_m2_max")
    a100 = get_hardware("nvidia_a100")
    wl = default_llm_workloads(dtype_bytes=4)[2]  # 1024x1024x1024 GEMM

    c_m2 = set(enumerate_feasible_schedules(wl, m2))
    c_a100 = set(enumerate_feasible_schedules(wl, a100))
    assert len(c_m2) > 50
    assert len(c_a100) > len(c_m2), "A100 (164KB SRAM) must support strictly larger feasible schedule space than M2 Max (32KB)"
    assert len(c_a100 - c_m2) > 100


def test_schedule_mdp_transitions() -> None:
    hw = get_hardware("nvidia_a100")
    wl = default_llm_workloads(dtype_bytes=2)[1]
    sim = CalibratedHardwareSimulator(noise_std=0.0, seed=0)
    mdp = ScheduleMDP(wl, hw, lambda w, c, h: sim.true_log_sol_efficiency(w, c, h), horizon=8)
    s0 = mdp.reset()
    ok, _ = check_feasibility(wl, s0, hw)
    assert ok
    valid = mdp.valid_actions(s0)
    assert ScheduleMDP.STOP_ACTION in valid
    assert len(valid) >= 2
    s1, r, done, info = mdp.step(valid[1])
    assert isinstance(r, float)
    ok1, _ = check_feasibility(wl, s1, hw)
    assert ok1


def test_crossarch_rl_beats_random_on_cross_vendor_transfer() -> None:
    """Smoke test: offline training on Apple M2 Max -> few-shot transfer to NVIDIA A100."""
    sim = CalibratedHardwareSimulator(noise_std=0.015, seed=7)
    src_hw = get_hardware("apple_m2_max")
    tgt_hw = get_hardware("nvidia_a100")
    workloads = default_llm_workloads(dtype_bytes=2)[:6]

    # Collect offline source traces on Apple M2 Max
    traces = []
    for wl in workloads:
        pool = enumerate_feasible_schedules(wl, src_hw, max_configs=60, seed=0)
        for c in pool:
            traces.append((src_hw, wl, c, sim.evaluate(wl, c, src_hw, noisy=True)))

    tgt_pools = {wl.name: enumerate_feasible_schedules(wl, tgt_hw, max_configs=100, seed=1) for wl in workloads}

    agent = CrossArchRL(CrossArchConfig(offline_epochs=80, seed=0))
    agent.fit_offline(traces)
    ca_res = agent.adapt_and_evaluate(tgt_hw, workloads, tgt_pools, sim.evaluate, budgets=(0, 3, 5, 10), rng_seed=0)
    rand_res = RandomSearchTuner().adapt_and_evaluate(tgt_hw, workloads, tgt_pools, sim.evaluate, budgets=(0, 3, 5, 10), rng_seed=0)

    assert ca_res[0]["mean_pct_oracle"] > rand_res[0]["mean_pct_oracle"] + 0.10
    assert ca_res[5]["mean_pct_oracle"] > rand_res[5]["mean_pct_oracle"] + 0.05
    assert ca_res[10]["mean_pct_oracle"] > 0.85


def test_rliable_stats() -> None:
    a = np.array([[0.90, 0.92], [0.91, 0.93], [0.89, 0.94], [0.92, 0.91]])
    b = np.array([[0.70, 0.72], [0.71, 0.73], [0.69, 0.74], [0.72, 0.71]])
    pt, lo, hi = stratified_bootstrap_iqm(a, n_resamples=200, seed=0)
    assert lo <= pt <= hi
    assert probability_of_improvement(a, b) == 1.0


if __name__ == "__main__":
    test_hardware_support_shift()
    test_schedule_mdp_transitions()
    test_crossarch_rl_beats_random_on_cross_vendor_transfer()
    test_rliable_stats()
    print("ALL UNIT TESTS PASSED!")
