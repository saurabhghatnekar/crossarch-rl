"""Generate publication-quality vector PDF figures for the CrossArch-RL ICLR Workshop paper."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_crossarch")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Okabe-Ito colorblind-safe palette
COLORS = {
    "CrossArch-RL": "#0072B2",
    "CrossArch-RL (Full)": "#0072B2",
    "Fine-Tuned MLP": "#E69F00",
    "GP-BO (Matern-5/2)": "#009E73",
    "Neural-UCB": "#CC79A7",
    "Evolutionary (Ansor-style)": "#D55E00",
    "Zero-Shot MLP": "#56B4E9",
    "Random Search": "#7F7F7F",
    "Default Heuristic": "#222222",
    # Ablations
    "w/o Datasheet Prior": "#D55E00",
    "w/o Dimensionless Norm": "#CC79A7",
    "w/o GPI Library": "#009E73",
    "w/o Pessimistic LCB": "#E69F00",
    "Linear Physics Only": "#56B4E9",
}

MARKERS = {
    "CrossArch-RL": "o",
    "CrossArch-RL (Full)": "o",
    "Fine-Tuned MLP": "s",
    "GP-BO (Matern-5/2)": "^",
    "Neural-UCB": "D",
    "Evolutionary (Ansor-style)": "v",
    "Zero-Shot MLP": "P",
    "Random Search": "x",
    "Default Heuristic": "_",
    "w/o Datasheet Prior": "s",
    "w/o Dimensionless Norm": "^",
    "w/o GPI Library": "D",
    "w/o Pessimistic LCB": "v",
    "Linear Physics Only": "P",
}


def _set_pub_style() -> None:
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
        "lines.linewidth": 2.0,
        "lines.markersize": 5.0,
        "figure.dpi": 300,
    })


def plot_fig1_multi_gpu(data: dict, fig_dir: Path) -> None:
    exp1 = data["exp1_multi_gpu"]
    overall = exp1["overall"]
    per_target = exp1["per_target"]
    ks = [0, 1, 3, 5, 10, 15, 25]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.1), gridspec_kw={"width_ratios": [1.15, 1.0]})

    # Panel (a): IQM % of Oracle vs Compile Budget K
    plot_order = [
        "CrossArch-RL",
        "Fine-Tuned MLP",
        "GP-BO (Matern-5/2)",
        "Neural-UCB",
        "Evolutionary (Ansor-style)",
        "Zero-Shot MLP",
        "Random Search",
        "Default Heuristic",
    ]
    for m in plot_order:
        y = [overall[m][str(k)]["iqm_pct_oracle"] * 100.0 for k in ks]
        lo = [overall[m][str(k)]["ci_lo_pct_oracle"] * 100.0 for k in ks]
        hi = [overall[m][str(k)]["ci_hi_pct_oracle"] * 100.0 for k in ks]
        ls = "--" if m in ("Default Heuristic", "Random Search", "Zero-Shot MLP") else "-"
        lw = 2.6 if m == "CrossArch-RL" else 1.65
        z = 10 if m == "CrossArch-RL" else 3
        ax1.plot(ks, y, label=m, color=COLORS[m], marker=MARKERS[m], linestyle=ls, linewidth=lw, zorder=z)
        ax1.fill_between(ks, lo, hi, color=COLORS[m], alpha=0.14 if m == "CrossArch-RL" else 0.08, zorder=z - 1)

    ax1.set_xlabel("Target-GPU Kernel Compilations ($K$)")
    ax1.set_ylabel("IQM % of Target-GPU Oracle Throughput")
    ax1.set_title("(a) 6-GPU Cross-Architecture Transfer (6 Unseen Ops)")
    ax1.set_xticks(ks)
    ax1.set_ylim(16, 104)
    ax1.legend(loc="lower right", frameon=True, framealpha=0.95, ncol=2, fontsize=7.3)

    # Panel (b): Per-Target-GPU IQM at K=5 compilations
    gpu_keys = ["apple_m2_max", "nvidia_t4", "nvidia_l4", "nvidia_a10g", "nvidia_a100", "nvidia_h100"]
    gpu_labels = ["M2 Max", "T4", "L4", "A10G", "A100", "H100"]
    bar_methods = ["CrossArch-RL", "Fine-Tuned MLP", "GP-BO (Matern-5/2)", "Evolutionary (Ansor-style)"]
    x = np.arange(len(gpu_keys))
    width = 0.20

    for i, m in enumerate(bar_methods):
        vals = [per_target[gk][m]["5"]["iqm_pct_oracle"] * 100.0 for gk in gpu_keys]
        errs_lo = [vals[j] - per_target[gk][m]["5"]["ci_lo_pct_oracle"] * 100.0 for j, gk in enumerate(gpu_keys)]
        errs_hi = [per_target[gk][m]["5"]["ci_hi_pct_oracle"] * 100.0 - vals[j] for j, gk in enumerate(gpu_keys)]
        offset = (i - 1.5) * width
        ax2.bar(
            x + offset, vals, width=width * 0.90, label=m, color=COLORS[m], alpha=0.92,
            yerr=[np.maximum(errs_lo, 0), np.maximum(errs_hi, 0)], capsize=2, error_kw={"linewidth": 0.8}
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels(gpu_labels)
    ax2.set_ylabel("IQM % of Oracle at $K=5$ Shots")
    ax2.set_title("(b) Few-Shot Transfer per Unseen Target GPU ($K=5$)")
    ax2.set_ylim(25, 118)
    ax2.legend(loc="upper center", frameon=True, framealpha=0.95, ncol=2, fontsize=7.4)

    plt.tight_layout()
    fig.savefig(fig_dir / "fig1_transfer_curves.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_fig2_real_hardware(data: dict, fig_dir: Path) -> None:
    if "exp2_real_hardware" not in data:
        return
    exp2 = data["exp2_real_hardware"]
    ks = [0, 1, 3, 5, 10, 15]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12.0, 3.7), gridspec_kw={"width_ratios": [1.0, 1.0, 1.18]})

    plot_order = [
        "CrossArch-RL",
        "Fine-Tuned MLP",
        "GP-BO (Matern-5/2)",
        "Neural-UCB",
        "Evolutionary (Ansor-style)",
        "Random Search",
        "Default Heuristic",
    ]

    # Panel (a): Real ARM64 CPU -> Real Apple M2 Max Metal GPU
    c2m = exp2["directions"]["cpu_to_metal"]["summary"]
    for m in plot_order:
        y = [c2m[m][str(k)]["iqm_pct_oracle"] * 100.0 for k in ks]
        lo = [c2m[m][str(k)]["ci_lo_pct_oracle"] * 100.0 for k in ks]
        hi = [c2m[m][str(k)]["ci_hi_pct_oracle"] * 100.0 for k in ks]
        ls = "--" if m in ("Default Heuristic", "Random Search") else "-"
        lw = 2.5 if m == "CrossArch-RL" else 1.5
        z = 10 if m == "CrossArch-RL" else 3
        ax1.plot(ks, y, label=m, color=COLORS[m], marker=MARKERS[m], linestyle=ls, linewidth=lw, zorder=z)
        ax1.fill_between(ks, lo, hi, color=COLORS[m], alpha=0.14 if m == "CrossArch-RL" else 0.07, zorder=z - 1)
    ax1.set_xlabel("Real Metal Compilations ($K$)")
    ax1.set_ylabel("IQM % of Measured Oracle")
    ax1.set_title("(a) Real CPU $\\to$ Metal GPU")
    ax1.set_xticks(ks)
    ax1.set_ylim(75, 101.5)

    # Panel (b): Real Apple M2 Max Metal GPU -> Real ARM64 CPU
    m2c = exp2["directions"]["metal_to_cpu"]["summary"]
    for m in plot_order:
        y = [m2c[m][str(k)]["iqm_pct_oracle"] * 100.0 for k in ks]
        lo = [m2c[m][str(k)]["ci_lo_pct_oracle"] * 100.0 for k in ks]
        hi = [m2c[m][str(k)]["ci_hi_pct_oracle"] * 100.0 for k in ks]
        ls = "--" if m in ("Default Heuristic", "Random Search") else "-"
        lw = 2.5 if m == "CrossArch-RL" else 1.5
        z = 10 if m == "CrossArch-RL" else 3
        ax2.plot(ks, y, label=m, color=COLORS[m], marker=MARKERS[m], linestyle=ls, linewidth=lw, zorder=z)
        ax2.fill_between(ks, lo, hi, color=COLORS[m], alpha=0.14 if m == "CrossArch-RL" else 0.07, zorder=z - 1)
    ax2.set_xlabel("Real ARM64 CPU Compilations ($K$)")
    ax2.set_ylabel("IQM % of Measured Oracle")
    ax2.set_title("(b) Real Metal GPU $\\to$ CPU")
    ax2.set_xticks(ks)
    ax2.set_ylim(22, 103)

    # Shared legend cleanly placed below panels (a)-(b)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.34, -0.07), ncol=4, frameon=True, fontsize=7.4)

    # Panel (c): Per-operator real Metal GPU % of measured oracle at K=5
    per_wl = exp2["directions"]["cpu_to_metal"]["per_workload_k5"]
    wl_names = list(per_wl.keys())
    short_names = [
        w.replace("gemm_decode_64x1024x1024", "GEMM\nDec")
         .replace("gemm_prefill_512x512x512", "GEMM\nPre")
         .replace("gemm_mlp_512x1024x512", "GEMM\nMLP")
         .replace("rmsnorm_1024x2048", "RMS\n2K")
         .replace("rmsnorm_2048x4096", "RMS\n4K")
         .replace("softmax_1024x2048", "Soft\n2K")
         .replace("softmax_1024x4096", "Soft\n4K")
         .replace("swiglu_1024x4096", "Swi\n1K")
         .replace("swiglu_2048x4096", "Swi\n2K")
        for w in wl_names
    ]
    bar_ms = ["CrossArch-RL", "Fine-Tuned MLP", "GP-BO (Matern-5/2)", "Default Heuristic"]
    x = np.arange(len(wl_names))
    w_bar = 0.20
    for i, m in enumerate(bar_ms):
        vals = [per_wl[wn][m]["iqm_pct_oracle"] * 100.0 for wn in wl_names]
        ax3.bar(x + (i - 1.5) * w_bar, vals, width=w_bar * 0.90, label=m, color=COLORS[m], alpha=0.92)

    ax3.set_xticks(x)
    ax3.set_xticklabels(short_names, fontsize=7.0)
    ax3.set_ylabel("IQM % of Measured Metal Oracle ($K=5$)")
    ax3.set_title("(c) Real Metal GPU per Operator ($K=5$)")
    ax3.set_ylim(30, 115)
    ax3.legend(loc="upper center", frameon=True, framealpha=0.95, ncol=2, fontsize=7.0)

    plt.tight_layout(rect=[0.0, 0.06, 1.0, 1.0])
    fig.savefig(fig_dir / "fig2_real_hardware.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_fig3_ablations_and_theory(data: dict, fig_dir: Path) -> None:
    exp3 = data["exp3_ablations"]
    abls = exp3["ablations"]
    ks = [0, 1, 3, 5, 10, 15, 25]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11.8, 3.6))

    # Panel (a): Ablation curves (IQM % of Oracle)
    abl_order = [
        "CrossArch-RL (Full)",
        "w/o GPI Library",
        "w/o Pessimistic LCB",
        "Linear Physics Only",
        "w/o Datasheet Prior",
        "w/o Dimensionless Norm",
    ]
    for m in abl_order:
        y = [abls[m][str(k)]["iqm_pct_oracle"] * 100.0 for k in ks]
        lo = [abls[m][str(k)]["ci_lo_pct_oracle"] * 100.0 for k in ks]
        hi = [abls[m][str(k)]["ci_hi_pct_oracle"] * 100.0 for k in ks]
        lw = 2.5 if m == "CrossArch-RL (Full)" else 1.55
        ax1.plot(ks, y, label=m, color=COLORS[m], marker=MARKERS[m], linewidth=lw)
        ax1.fill_between(ks, lo, hi, color=COLORS[m], alpha=0.10)

    ax1.set_xlabel("Target-GPU Compilations ($K$)")
    ax1.set_ylabel("IQM % of Oracle (A100 & H100)")
    ax1.set_title("(a) Surgical Component Ablations")
    ax1.set_xticks(ks)
    ax1.set_ylim(22, 104)
    ax1.legend(loc="lower right", fontsize=6.9, frameon=True, framealpha=0.95)

    # Panel (b): 10th-Percentile Cliff Safety at K=0 and K=5
    short_abl = ["Full", "w/o\nGPI", "w/o\nLCB", "Linear\nPhys", "w/o\nPrior", "w/o\nNorm"]
    p10_k0 = [abls[m]["0"]["p10_pct_oracle"] * 100.0 for m in abl_order]
    p10_k5 = [abls[m]["5"]["p10_pct_oracle"] * 100.0 for m in abl_order]
    x = np.arange(len(abl_order))
    ax2.bar(x - 0.18, p10_k0, width=0.34, label="$K=0$ (Zero-Shot)", color="#56B4E9", alpha=0.92)
    ax2.bar(x + 0.18, p10_k5, width=0.34, label="$K=5$ (Few-Shot)", color="#0072B2", alpha=0.92)
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_abl, fontsize=7.5)
    ax2.set_ylabel("10th-Percentile % of Oracle")
    ax2.set_title("(b) Tail Cliff-Avoidance Safety (p10)")
    ax2.set_ylim(0, 122)
    ax2.legend(loc="upper right", fontsize=7.4, frameon=True, framealpha=0.95)

    # Panel (c): Theorem 1 Empirical Regret Decay vs O(1/sqrt(K)) Envelope
    k_nz = np.array([1, 3, 5, 10, 15, 25], dtype=np.float64)
    reg_full = np.array([abls["CrossArch-RL (Full)"][str(int(k))]["iqm_regret"] for k in k_nz])
    reg_noprior = np.array([abls["w/o Datasheet Prior"][str(int(k))]["iqm_regret"] for k in k_nz])
    reg_nonorm = np.array([abls["w/o Dimensionless Norm"][str(int(k))]["iqm_regret"] for k in k_nz])
    # Reference O(1/sqrt(K)) theoretical slope anchored at w/o Datasheet Prior K=1
    theory_curve = reg_noprior[0] / np.sqrt(k_nz)

    ax3.plot(k_nz, reg_full, label="CrossArch-RL (Full)", color=COLORS["CrossArch-RL (Full)"], marker="o", linewidth=2.4)
    ax3.plot(k_nz, reg_noprior, label="w/o Datasheet Prior", color=COLORS["w/o Datasheet Prior"], marker="s", linewidth=1.8)
    ax3.plot(k_nz, reg_nonorm, label="w/o Dimensionless Norm", color=COLORS["w/o Dimensionless Norm"], marker="^", linewidth=1.8)
    ax3.plot(k_nz, theory_curve, label="$O(1/\\sqrt{K})$ Envelope", color="#333333", linestyle=":", linewidth=1.6)
    ax3.set_xlabel("Target-GPU Compilations ($K$)")
    ax3.set_ylabel("IQM Log-SOL Suboptimality (Regret)")
    ax3.set_title("(c) Theorem 1 Regret Scaling")
    ax3.set_xticks(k_nz)
    ax3.set_ylim(-0.04, max(reg_noprior[0], reg_nonorm[0]) * 1.12)
    ax3.legend(loc="upper right", fontsize=7.0, frameon=True, framealpha=0.95)

    plt.tight_layout()
    fig.savefig(fig_dir / "fig3_ablations_and_theory.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_fig4_prob_improvement(data: dict, fig_dir: Path) -> None:
    poi = data["exp1_multi_gpu"]["probability_of_improvement"]
    baselines = [
        "Fine-Tuned MLP",
        "GP-BO (Matern-5/2)",
        "Neural-UCB",
        "Evolutionary (Ansor-style)",
        "Zero-Shot MLP",
        "Random Search",
        "Default Heuristic",
    ]
    fig, ax = plt.subplots(figsize=(6.8, 3.1))
    y = np.arange(len(baselines))

    for offset, key, color, lbl in (
        (-0.16, "K5", "#0072B2", "Budget $K=5$"),
        (0.16, "K10", "#009E73", "Budget $K=10$"),
    ):
        pts = [poi[key][b]["poi"] for b in baselines]
        los = [pts[i] - poi[key][b]["ci_lo"] for i, b in enumerate(baselines)]
        his = [poi[key][b]["ci_hi"] - pts[i] for i, b in enumerate(baselines)]
        ax.errorbar(
            pts, y + offset, xerr=[np.maximum(los, 0), np.maximum(his, 0)],
            fmt="o", color=color, capsize=3, label=lbl, markersize=5.5, linewidth=1.6
        )

    ax.axvline(0.5, color="#666666", linestyle="--", linewidth=1.1, label="Parity (0.5)")
    ax.set_yticks(y)
    ax.set_yticklabels([f"CrossArch-RL > {b}" for b in baselines])
    ax.set_xlabel("Probability of Improvement (with 95% Bootstrap CI)")
    ax.set_title("rliable Probability of Improvement (6 Target GPUs $\\times$ 6 Unseen LLM Ops)")
    ax.set_xlim(0.46, 1.03)
    ax.invert_yaxis()
    ax.legend(loc="lower left", frameon=True, framealpha=0.95, fontsize=8.0)

    plt.tight_layout()
    fig.savefig(fig_dir / "fig4_prob_improvement.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    _set_pub_style()
    root = Path(__file__).resolve().parents[1]
    res_path = root / "results" / "experiment_results.json"
    fig_dir = root / "paper" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(res_path.read_text())
    plot_fig1_multi_gpu(data, fig_dir)
    plot_fig2_real_hardware(data, fig_dir)
    plot_fig3_ablations_and_theory(data, fig_dir)
    plot_fig4_prob_improvement(data, fig_dir)
    print(f"Generated publication vector PDF figures in {fig_dir}")


if __name__ == "__main__":
    main()
