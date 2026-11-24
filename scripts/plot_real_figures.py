"""Generate publication-quality vector PDF figures from 100% REAL multi-GPU silicon traces."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_crossarch_real")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


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
        "legend.fontsize": 7.6,
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


def plot_fig1_real_nvidia_4gpu(data: dict, fig_dir: Path) -> None:
    exp1 = data["exp1_real_nvidia_4gpu"]
    overall = exp1["overall"]
    per_target = exp1["per_target"]
    gpus = exp1["gpus"]
    ks = [0, 1, 3, 5, 10, 15, 25]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.1), gridspec_kw={"width_ratios": [1.15, 1.0]})

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
    min_y = 100.0
    for m in plot_order:
        y = [overall[m][str(k)]["iqm_pct_oracle"] * 100.0 for k in ks]
        lo = [overall[m][str(k)]["ci_lo_pct_oracle"] * 100.0 for k in ks]
        hi = [overall[m][str(k)]["ci_hi_pct_oracle"] * 100.0 for k in ks]
        min_y = min(min_y, min(lo))
        ls = "--" if m in ("Default Heuristic", "Random Search", "Zero-Shot MLP") else "-"
        lw = 2.6 if m == "CrossArch-RL" else 1.65
        z = 10 if m == "CrossArch-RL" else 3
        ax1.plot(ks, y, label=m, color=COLORS[m], marker=MARKERS[m], linestyle=ls, linewidth=lw, zorder=z)
        ax1.fill_between(ks, lo, hi, color=COLORS[m], alpha=0.14 if m == "CrossArch-RL" else 0.08, zorder=z - 1)

    ax1.set_xlabel("Real Target-GPU Triton Compilations ($K$)")
    ax1.set_ylabel("IQM % of Measured Target-GPU Oracle")
    ax1.set_title("(a) Live 4-GPU NVIDIA Triton Leave-One-GPU-Out (12 Ops)")
    ax1.set_xticks(ks)
    ax1.set_ylim(max(0, min_y - 8), 104)
    ax1.legend(loc="lower right", frameon=True, framealpha=0.95, ncol=2, fontsize=7.2)

    lbl_map = {"nvidia_t4": "T4\n(SM75)", "nvidia_l4": "L4\n(SM89)", "nvidia_a100": "A100-80G\n(SM80)", "nvidia_h100": "H100\n(SM90)"}
    bar_methods = ["CrossArch-RL", "Fine-Tuned MLP", "GP-BO (Matern-5/2)", "Evolutionary (Ansor-style)"]
    x = np.arange(len(gpus))
    width = 0.20
    bar_min = 100.0
    for i, m in enumerate(bar_methods):
        vals = [per_target[gk][m]["5"]["iqm_pct_oracle"] * 100.0 for gk in gpus]
        los = [vals[j] - per_target[gk][m]["5"]["ci_lo_pct_oracle"] * 100.0 for j, gk in enumerate(gpus)]
        his = [per_target[gk][m]["5"]["ci_hi_pct_oracle"] * 100.0 - vals[j] for j, gk in enumerate(gpus)]
        bar_min = min(bar_min, min(vals) - 5)
        ax2.bar(
            x + (i - 1.5) * width, vals, width=width * 0.90, label=m, color=COLORS[m], alpha=0.92,
            yerr=[np.maximum(los, 0), np.maximum(his, 0)], capsize=2, error_kw={"linewidth": 0.8}
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels([lbl_map.get(g, g) for g in gpus])
    ax2.set_ylabel("IQM % of Measured Oracle at $K=5$")
    ax2.set_title("(b) Live Few-Shot Transfer per Unseen NVIDIA GPU ($K=5$)")
    ax2.set_ylim(max(0, bar_min - 8), 116)
    ax2.legend(loc="upper center", frameon=True, framealpha=0.95, ncol=2, fontsize=7.4)

    plt.tight_layout()
    fig.savefig(fig_dir / "fig1_transfer_curves.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_fig2_cross_vendor_real(data: dict, fig_dir: Path) -> None:
    exp2 = data["exp2_real_cross_vendor"]
    ks = [0, 1, 3, 5, 10, 15]
    panels = [
        ("metal_to_h100", "(a) Apple Metal $\\to$ NVIDIA H100"),
        ("metal_to_a100", "(b) Apple Metal $\\to$ NVIDIA A100"),
        ("cpu_to_metal", "(c) ARM64 CPU $\\to$ Apple Metal"),
    ]
    panels = [(k, t) for k, t in panels if k in exp2]
    if "metal_to_cpu" in exp2 and len(panels) < 3:
        panels.append(("metal_to_cpu", "(c) Apple Metal $\\to$ ARM64 CPU"))

    fig, axes = plt.subplots(1, len(panels), figsize=(11.8, 3.7))
    if len(panels) == 1:
        axes = [axes]

    plot_order = [
        "CrossArch-RL",
        "Fine-Tuned MLP",
        "GP-BO (Matern-5/2)",
        "Neural-UCB",
        "Evolutionary (Ansor-style)",
        "Random Search",
        "Default Heuristic",
    ]
    for ax, (pkey, title) in zip(axes, panels):
        sub = exp2[pkey]
        min_y = 100.0
        for m in plot_order:
            y = [sub[m][str(k)]["iqm_pct_oracle"] * 100.0 for k in ks]
            lo = [sub[m][str(k)]["ci_lo_pct_oracle"] * 100.0 for k in ks]
            hi = [sub[m][str(k)]["ci_hi_pct_oracle"] * 100.0 for k in ks]
            min_y = min(min_y, min(lo))
            ls = "--" if m in ("Default Heuristic", "Random Search") else "-"
            lw = 2.5 if m == "CrossArch-RL" else 1.5
            z = 10 if m == "CrossArch-RL" else 3
            ax.plot(ks, y, label=m, color=COLORS[m], marker=MARKERS[m], linestyle=ls, linewidth=lw, zorder=z)
            ax.fill_between(ks, lo, hi, color=COLORS[m], alpha=0.14 if m == "CrossArch-RL" else 0.07, zorder=z - 1)
        ax.set_xlabel("Real Target Compilations ($K$)")
        ax.set_ylabel("IQM % of Measured Oracle")
        ax.set_title(title)
        ax.set_xticks(ks)
        ax.set_ylim(max(0, min_y - 6), 103)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.07), ncol=4, frameon=True, fontsize=7.6)
    plt.tight_layout(rect=[0.0, 0.06, 1.0, 1.0])
    fig.savefig(fig_dir / "fig2_real_hardware.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_fig3_real_ablations(data: dict, fig_dir: Path) -> None:
    exp3 = data["exp3_real_ablations"]
    abls = exp3["ablations"]
    ks = [0, 1, 3, 5, 10, 15, 25]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11.8, 3.6))

    abl_order = [
        "CrossArch-RL (Full)",
        "w/o GPI Library",
        "w/o Pessimistic LCB",
        "Linear Physics Only",
        "w/o Datasheet Prior",
        "w/o Dimensionless Norm",
    ]
    min_y = 100.0
    for m in abl_order:
        y = [abls[m][str(k)]["iqm_pct_oracle"] * 100.0 for k in ks]
        lo = [abls[m][str(k)]["ci_lo_pct_oracle"] * 100.0 for k in ks]
        hi = [abls[m][str(k)]["ci_hi_pct_oracle"] * 100.0 for k in ks]
        min_y = min(min_y, min(lo))
        lw = 2.5 if m == "CrossArch-RL (Full)" else 1.55
        ax1.plot(ks, y, label=m, color=COLORS[m], marker=MARKERS[m], linewidth=lw)
        ax1.fill_between(ks, lo, hi, color=COLORS[m], alpha=0.10)

    ax1.set_xlabel("Real Target-GPU Compilations ($K$)")
    ax1.set_ylabel("IQM % of Measured Oracle (A100 & H100)")
    ax1.set_title("(a) Live Multi-GPU Ablations")
    ax1.set_xticks(ks)
    ax1.set_ylim(max(0, min_y - 6), 104)
    ax1.legend(loc="lower right", fontsize=6.9, frameon=True, framealpha=0.95)

    short_abl = ["Full", "w/o\nGPI", "w/o\nLCB", "Linear\nPhys", "w/o\nPrior", "w/o\nNorm"]
    p10_k0 = [abls[m]["0"]["p10_pct_oracle"] * 100.0 for m in abl_order]
    p10_k5 = [abls[m]["5"]["p10_pct_oracle"] * 100.0 for m in abl_order]
    x = np.arange(len(abl_order))
    ax2.bar(x - 0.18, p10_k0, width=0.34, label="$K=0$ (Zero-Shot)", color="#56B4E9", alpha=0.92)
    ax2.bar(x + 0.18, p10_k5, width=0.34, label="$K=5$ (Few-Shot)", color="#0072B2", alpha=0.92)
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_abl, fontsize=7.5)
    ax2.set_ylabel("10th-Percentile % of Measured Oracle")
    ax2.set_title("(b) Live Cliff-Avoidance Safety (p10)")
    ax2.set_ylim(0, 122)
    ax2.legend(loc="upper right", fontsize=7.4, frameon=True, framealpha=0.95)

    k_nz = np.array([1, 3, 5, 10, 15, 25], dtype=np.float64)
    reg_full = np.array([abls["CrossArch-RL (Full)"][str(int(k))]["iqm_regret"] for k in k_nz])
    reg_noprior = np.array([abls["w/o Datasheet Prior"][str(int(k))]["iqm_regret"] for k in k_nz])
    reg_nonorm = np.array([abls["w/o Dimensionless Norm"][str(int(k))]["iqm_regret"] for k in k_nz])
    anchor = max(reg_noprior[0], reg_nonorm[0], 0.1)
    theory_curve = anchor / np.sqrt(k_nz)

    ax3.plot(k_nz, reg_full, label="CrossArch-RL (Full)", color=COLORS["CrossArch-RL (Full)"], marker="o", linewidth=2.4)
    ax3.plot(k_nz, reg_noprior, label="w/o Datasheet Prior", color=COLORS["w/o Datasheet Prior"], marker="s", linewidth=1.8)
    ax3.plot(k_nz, reg_nonorm, label="w/o Dimensionless Norm", color=COLORS["w/o Dimensionless Norm"], marker="^", linewidth=1.8)
    ax3.plot(k_nz, theory_curve, label="$O(1/\\sqrt{K})$ Envelope", color="#333333", linestyle=":", linewidth=1.6)
    ax3.set_xlabel("Real Target-GPU Compilations ($K$)")
    ax3.set_ylabel("IQM Log-SOL Suboptimality (Regret)")
    ax3.set_title("(c) Theorem 1 Scaling on Live Silicon")
    ax3.set_xticks(k_nz)
    ax3.set_ylim(-0.02, anchor * 1.15)
    ax3.legend(loc="upper right", fontsize=7.0, frameon=True, framealpha=0.95)

    plt.tight_layout()
    fig.savefig(fig_dir / "fig3_ablations_and_theory.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_fig4_real_poi(data: dict, fig_dir: Path) -> None:
    poi = data["exp1_real_nvidia_4gpu"]["probability_of_improvement"]
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
    ax.set_title("rliable Probability of Improvement Across 4 Live NVIDIA GPUs $\\times$ 12 Triton Ops")
    ax.set_xlim(0.35, 1.03)
    ax.invert_yaxis()
    ax.legend(loc="lower left", frameon=True, framealpha=0.95, fontsize=8.0)

    plt.tight_layout()
    fig.savefig(fig_dir / "fig4_prob_improvement.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    _set_pub_style()
    root = Path(__file__).resolve().parents[1]
    res_path = root / "results" / "real_hardware_results.json"
    fig_dir = root / "paper" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(res_path.read_text())
    plot_fig1_real_nvidia_4gpu(data, fig_dir)
    plot_fig2_cross_vendor_real(data, fig_dir)
    plot_fig3_real_ablations(data, fig_dir)
    plot_fig4_real_poi(data, fig_dir)
    print(f"Generated 100% REAL-SILICON vector PDF figures in {fig_dir}")


if __name__ == "__main__":
    main()
