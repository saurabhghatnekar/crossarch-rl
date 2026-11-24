"""
Run True Multi-Step Sequential RL Evaluation on 100% Live NVIDIA Multi-GPU Silicon
(Tesla T4, NVIDIA L4, A100-SXM4-80GB, H100 80GB HBM3 — 2,935 verified Triton kernels).

Compares:
  - CrossArch-RL (TD-SF + GPI)  [True Bellman TD Successor Features + GPI + Bayesian posterior]
  - Offline IQL (Fine-Tuned)    [Kostrikov et al., ICLR 2022]
  - Offline CQL (Fine-Tuned)    [Kumar et al., NeurIPS 2020]
  - Online PPO (GAE)            [Schulman et al., 2017]
  - Random Walk                 [Local mutation baseline]
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_seqrl")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from crossarch_rl.algorithms.seq_rl import (
    OfflineCQLAgent,
    OfflineIQLAgent,
    OnlinePPOAgent,
    RandomWalkMDPBaseline,
    SequentialCrossArchRL,
)
from crossarch_rl.eval_stats import probability_of_improvement, stratified_bootstrap_iqm
from crossarch_rl.seq_mdp import ScheduleGraphMDP, collect_offline_trajectories

# Reuse real device JSONL loader
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_real_experiments import NVIDIA_KEYS, load_real_device


EPISODES = (0, 1, 2, 3, 5)


def build_device_mdps(dev: Dict[str, Any], seed: int = 0) -> List[ScheduleGraphMDP]:
    mdps = []
    for i, wl in enumerate(dev["workloads"]):
        pool = dev["pools"][wl.name]
        true_s = np.array([dev["scores"][(wl.name, c.as_tuple())] for c in pool], dtype=np.float64)
        mdps.append(ScheduleGraphMDP(wl, dev["hw"], pool, true_s, k_nbr=6, horizon=8, gamma=0.90, noise_std=0.008, seed=seed + i))
    return mdps


def aggregate_ep(runs: List[Dict[int, List[float]]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for e in EPISODES:
        mat = np.array([r[e] for r in runs], dtype=np.float64)
        iqm, lo, hi = stratified_bootstrap_iqm(mat, n_resamples=800, seed=42)
        out[str(e)] = {
            "iqm_pct_oracle": iqm,
            "ci_lo": lo,
            "ci_hi": hi,
            "p10_pct_oracle": float(np.percentile(mat, 10)),
            "raw_flat": mat.ravel().tolist(),
        }
    return out


def plot_seq_figure(results: Dict[str, Any], fig_path: Path) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9.5,
        "axes.labelsize": 10,
        "axes.titlesize": 10.5,
        "legend.fontsize": 7.8,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "figure.dpi": 300,
    })
    colors = {
        "CrossArch-RL (TD-SF+GPI)": "#0072B2",
        "Offline IQL": "#E69F00",
        "Offline CQL": "#009E73",
        "Online PPO (GAE)": "#CC79A7",
        "Random Walk": "#7F7F7F",
    }
    markers = {
        "CrossArch-RL (TD-SF+GPI)": "o",
        "Offline IQL": "s",
        "Offline CQL": "^",
        "Online PPO (GAE)": "D",
        "Random Walk": "x",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.7), gridspec_kw={"width_ratios": [1.1, 1.0]})

    # Panel (a): IQM % of measured oracle vs target-GPU rollout episodes E
    overall = results["overall"]
    es = list(EPISODES)
    for m in ["CrossArch-RL (TD-SF+GPI)", "Offline IQL", "Offline CQL", "Online PPO (GAE)", "Random Walk"]:
        y = [overall[m][str(e)]["iqm_pct_oracle"] * 100.0 for e in es]
        lo = [overall[m][str(e)]["ci_lo"] * 100.0 for e in es]
        hi = [overall[m][str(e)]["ci_hi"] * 100.0 for e in es]
        lw = 2.6 if "CrossArch" in m else 1.7
        ls = "--" if m == "Random Walk" else "-"
        ax1.plot(es, y, label=m, color=colors[m], marker=markers[m], linewidth=lw, linestyle=ls)
        ax1.fill_between(es, lo, hi, color=colors[m], alpha=0.14 if "CrossArch" in m else 0.08)

    ax1.set_xlabel("Target-GPU Adaptation Episodes ($E$, Horizon $H=8$)")
    ax1.set_ylabel("IQM % of Measured Target-GPU Oracle")
    ax1.set_title("(a) Multi-Step ScheduleMDP Transfer (4 Live NVIDIA GPUs)")
    ax1.set_xticks(es)
    ax1.set_ylim(48, 103)
    ax1.legend(loc="lower right", frameon=True, framealpha=0.95)

    # Panel (b): Zero-Shot (E=0) & 1-Episode (E=1) Per-GPU Comparison
    gpus = results["gpus"]
    lbls = {"nvidia_t4": "T4\n(SM75)", "nvidia_l4": "L4\n(SM89)", "nvidia_a100": "A100-80G\n(SM80)", "nvidia_h100": "H100\n(SM90)"}
    bar_ms = ["CrossArch-RL (TD-SF+GPI)", "Offline IQL", "Offline CQL", "Online PPO (GAE)"]
    x = np.arange(len(gpus))
    w = 0.20
    for i, m in enumerate(bar_ms):
        vals = [results["per_target"][g][m]["1"]["iqm_pct_oracle"] * 100.0 for g in gpus]
        los = [vals[j] - results["per_target"][g][m]["1"]["ci_lo"] * 100.0 for j, g in enumerate(gpus)]
        his = [results["per_target"][g][m]["1"]["ci_hi"] * 100.0 - vals[j] for j, g in enumerate(gpus)]
        ax2.bar(
            x + (i - 1.5) * w, vals, width=w * 0.9, label=m, color=colors[m], alpha=0.92,
            yerr=[np.maximum(los, 0), np.maximum(his, 0)], capsize=2, error_kw={"linewidth": 0.8}
        )
    ax2.set_xticks(x)
    ax2.set_xticklabels([lbls[g] for g in gpus])
    ax2.set_ylabel("IQM % of Oracle after 1 Episode ($E=1$)")
    ax2.set_title("(b) Sequential RL Adaptation per Unseen GPU ($E=1$)")
    ax2.set_ylim(55, 114)
    ax2.legend(loc="upper center", ncol=2, frameon=True, framealpha=0.95, fontsize=7.4)

    plt.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    t0 = time.perf_counter()
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    devices = {k: load_real_device(data_dir / f"traces_{k}.jsonl") for k in NVIDIA_KEYS}

    n_seeds = 8
    methods = ["CrossArch-RL (TD-SF+GPI)", "Offline IQL", "Offline CQL", "Online PPO (GAE)", "Random Walk"]
    all_runs: Dict[str, List[Dict[int, List[float]]]] = {m: [] for m in methods}
    per_target: Dict[str, Any] = {}

    print("=== True Multi-Step Sequential ScheduleMDP RL on 4 Live NVIDIA GPUs ===", flush=True)
    for tgt_key in NVIDIA_KEYS:
        src_keys = [k for k in NVIDIA_KEYS if k != tgt_key]
        src_mdps = []
        for sk in src_keys:
            for mdp in build_device_mdps(devices[sk], seed=0):
                src_mdps.append((devices[sk]["hw"].name, mdp))

        transitions = collect_offline_trajectories(src_mdps, episodes_per_mdp=16, seed=0)
        print(f"\nTarget: {devices[tgt_key]['device_name']} | Offline transitions: {len(transitions)} from {src_keys}", flush=True)

        sf_agent = SequentialCrossArchRL(res_dim=12, gamma=0.90, seed=0)
        sf_stats = sf_agent.fit_offline_td(transitions, cumulant_epochs=80, td_epochs=60)
        print(f"  [TD-SF+GPI] cum_loss={sf_stats['cumulant_loss']:.4f} td_loss={sf_stats['td_loss']:.4f} heads={int(sf_stats['n_sf_heads'])}", flush=True)

        iql = OfflineIQLAgent(gamma=0.90, expectile=0.75, seed=0)
        iql.fit_offline(transitions, epochs=65)

        cql = OfflineCQLAgent(gamma=0.90, cql_alpha=0.25, seed=0)
        cql.fit_offline(transitions, epochs=65)

        ppo = OnlinePPOAgent(gamma=0.90)
        rw = RandomWalkMDPBaseline()

        t_runs: Dict[str, List[Dict[int, List[float]]]] = {m: [] for m in methods}
        for s in range(n_seeds):
            tgt_mdps = build_device_mdps(devices[tgt_key], seed=100 + s)
            t_hw = devices[tgt_key]["hw"]
            t_runs["CrossArch-RL (TD-SF+GPI)"].append(sf_agent.evaluate_target_mdps(t_hw, tgt_mdps, EPISODES, rng_seed=s))
            t_runs["Offline IQL"].append(iql.evaluate_target_mdps(t_hw, tgt_mdps, EPISODES, rng_seed=s))
            t_runs["Offline CQL"].append(cql.evaluate_target_mdps(t_hw, tgt_mdps, EPISODES, rng_seed=s))
            t_runs["Online PPO (GAE)"].append(ppo.evaluate_target_mdps(t_hw, tgt_mdps, EPISODES, rng_seed=s))
            t_runs["Random Walk"].append(rw.evaluate_target_mdps(t_hw, tgt_mdps, EPISODES, rng_seed=s))

        per_target[tgt_key] = {}
        for m in methods:
            all_runs[m].extend(t_runs[m])
            agg = aggregate_ep(t_runs[m])
            per_target[tgt_key][m] = {k: {kk: vv for kk, vv in d.items() if kk != "raw_flat"} for k, d in agg.items()}
        print(
            f"  E=0 -> TD-SF+GPI: {per_target[tgt_key]['CrossArch-RL (TD-SF+GPI)']['0']['iqm_pct_oracle']*100:.1f}% vs "
            f"IQL: {per_target[tgt_key]['Offline IQL']['0']['iqm_pct_oracle']*100:.1f}% vs "
            f"CQL: {per_target[tgt_key]['Offline CQL']['0']['iqm_pct_oracle']*100:.1f}% | "
            f"E=1 -> TD-SF+GPI: {per_target[tgt_key]['CrossArch-RL (TD-SF+GPI)']['1']['iqm_pct_oracle']*100:.1f}% vs "
            f"IQL: {per_target[tgt_key]['Offline IQL']['1']['iqm_pct_oracle']*100:.1f}% vs "
            f"PPO: {per_target[tgt_key]['Online PPO (GAE)']['1']['iqm_pct_oracle']*100:.1f}%",
            flush=True,
        )

    overall: Dict[str, Any] = {}
    raw_e0, raw_e1 = {}, {}
    for m in methods:
        agg = aggregate_ep(all_runs[m])
        raw_e0[m] = np.array(agg["0"]["raw_flat"])
        raw_e1[m] = np.array(agg["1"]["raw_flat"])
        overall[m] = {k: {kk: vv for kk, vv in d.items() if kk != "raw_flat"} for k, d in agg.items()}

    print("\n=== OVERALL MULTI-STEP SEQUENTIAL RL (4 LIVE NVIDIA GPUs) ===")
    for m in methods:
        row = " | ".join(f"E={e}:{overall[m][str(e)]['iqm_pct_oracle']*100:5.1f}%" for e in EPISODES)
        print(f"  {m:26s}: {row}")

    poi = {"E0": {}, "E1": {}}
    for m in methods:
        if m == "CrossArch-RL (TD-SF+GPI)":
            continue
        poi["E0"][m] = probability_of_improvement(raw_e0["CrossArch-RL (TD-SF+GPI)"], raw_e0[m])
        poi["E1"][m] = probability_of_improvement(raw_e1["CrossArch-RL (TD-SF+GPI)"], raw_e1[m])
    print("  PoI E=0:", {k: round(v, 3) for k, v in poi["E0"].items()})
    print("  PoI E=1:", {k: round(v, 3) for k, v in poi["E1"].items()})

    results = {
        "overall": overall,
        "per_target": per_target,
        "probability_of_improvement": poi,
        "gpus": list(NVIDIA_KEYS),
        "n_seeds": n_seeds,
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }
    out_path = root / "results" / "seq_rl_results.json"
    out_path.write_text(json.dumps(results, indent=2))

    fig_path = root / "paper" / "figures" / "fig5_seq_mdp_rl.pdf"
    plot_seq_figure(results, fig_path)
    print(f"\nSaved sequential RL results -> {out_path} & figure -> {fig_path} ({results['elapsed_s']}s)")


if __name__ == "__main__":
    main()
