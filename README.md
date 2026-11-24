# CrossArch-RL

**Few-Shot Cross-Architecture GPU Kernel Schedule RL via Hardware-Factored Value Decomposition**

GPU kernel autotuners (Triton, TVM/MetaSchedule, Ansor) treat every new GPU generation as a cold-start black-box search problem, recompiling and benchmarking dozens to hundreds of candidate tile/warp/pipeline schedules per operator shape. **CrossArch-RL** transfers kernel scheduling policies across unseen hardware architectures—achieving **93.1% of oracle speed zero-shot** (99.0% on H100) and **97.6% within $K=5$ hardware measurements** (**12.2×–12.3× faster** than full Triton autotuning sweeps), verified on **3,847 live physical hardware executions** across 6 real compute backends.

---

## Core Idea

Instead of retraining an end-to-end neural cost model or running black-box search from scratch on each new GPU, CrossArch-RL factors the kernel schedule value function into a **shared cross-architecture representation** and a **hardware-specific linear weight vector**:

$$Q(s, a, h) = \psi(s, a, h)^\top w_h, \qquad \psi(s, a, h) = \big[\,\phi_{\text{phys}}(s, a, h);\; \psi_{\theta}(s, a, h)\,\big] \in \mathbb{R}^{24}$$

1. **Analytic Bottleneck Basis ($\phi_{\text{phys}} \in \mathbb{R}^{12}$):** Dimensionless hardware-relative ratios capturing roofline compute/memory bounds, shared-memory footprint, SM wave quantization, occupancy, and arithmetic intensity.
2. **Learned Residual Representation ($\psi_{\theta} \in \mathbb{R}^{12}$):** A lightweight MLP trained offline across source GPUs (using conservative expectile regression + orthogonality regularization against $\phi_{\text{phys}}$) to capture nonlinear compiler and microarchitectural effects (e.g., Tensor Core `mma` packing and `cp.async` multi-stage software pipelining) that analytic roofline models miss.
3. **Datasheet-Scaled Zero-Shot Prior & Closed-Form Few-Shot Adaptation:** On a brand-new unseen target GPU $h_{\text{tgt}}$, CrossArch-RL initializes $w_{h_{\text{tgt}}}$ from source posteriors with prior variance scaled by hardware datasheet distance $\Sigma_0(h_{\text{tgt}})$, selects informative schedules via **GPI-UCB** (anchored at the zero-shot LCB-best schedule for $k=1$), and updates $w_{h_{\text{tgt}}}$ via **closed-form Bayesian ridge regression**—requiring **zero gradient steps** on the target GPU.
4. **Multi-Step Sequential `ScheduleMDP` Extension:** Extends the same decomposition to iterative schedule refinement over measured hardware graphs ($H=8, \gamma=0.90$) using **Bellman TD Successor Features + Generalized Policy Improvement (GPI)** with Polyak target networks.

---

## Live Physical Hardware Dataset (3,847 Verified Kernels)

All main evaluations run on real, numerically verified (`allclose` against FP32 reference) kernel measurements stored in [`data/`](data/):

| Hardware Backend | Compiler / Runtime | Verified Kernels | Operators Covered | Latency Range (Worst / Best) |
|---|---|---:|---|---|
| **NVIDIA Tesla T4** (Turing) | Triton 3.9 (Modal) | 730 | GEMM, RMSNorm, Softmax, SwiGLU, FlashAttn-2 | up to **89.6×** |
| **NVIDIA L4** (Ada Lovelace) | Triton 3.9 (Modal) | 733 | GEMM, RMSNorm, Softmax, SwiGLU, FlashAttn-2 | up to **62.3×** |
| **NVIDIA A100-SXM4-80GB** (Ampere) | Triton 3.9 (Modal) | 736 | GEMM, RMSNorm, Softmax, SwiGLU, FlashAttn-2 | up to **124.7×** |
| **NVIDIA H100 80GB HBM3** (Hopper) | Triton 3.9 (Modal) | 736 | GEMM, RMSNorm, Softmax, SwiGLU, FlashAttn-2 | up to **178.9×** (GEMM) / **56.6×** (FlashAttn-2) |
| **Apple M2 Max GPU** | Custom MSL (`torch.mps.compile_shader`) | 456 | GEMM, RMSNorm, Softmax, SwiGLU, FlashAttn-2 | up to **5.96×** |
| **Apple M2 Max CPU** | Native Clang `-O3` + pthreads (ARM64) | 456 | GEMM, RMSNorm, Softmax, SwiGLU, FlashAttn-2 | up to **23.03×** |
| **Total** | **6 Physical Backends** | **3,847** | **5 LLM Operators** | — |

---

## Key Results (100% Live Silicon)

All scores report `rliable` Interquartile Mean (IQM) **% of Oracle Speed** across workloads and seeds (exact JSON logs in [`results/`](results/)).

### 1. 4-GPU Live NVIDIA Triton Leave-One-GPU-Out Transfer

Train offline on 3 NVIDIA GPUs, evaluate zero-shot ($K=0$) and few-shot ($K \in \{3, 5, 10\}$) on the held-out 4th GPU (`results/real_hardware_results.json`):

| Method | $K=0$ (Zero-Shot) | $K=3$ | $K=5$ | $K=10$ |
|---|---:|---:|---:|---:|
| **CrossArch-RL (Ours)** | **93.1%** | **96.4%** | **97.6%** | 99.5% |
| Fine-Tuned MLP | 88.0% | 94.8% | 97.3% | **99.6%** |
| Zero-Shot MLP | 81.6% | 93.1% | 95.9% | 98.2% |
| GP-BO (Matérn-5/2) | 71.2% | 84.4% | 93.6% | 98.1% |
| Neural-UCB | 66.2% | 86.7% | 93.2% | 97.7% |
| Evolutionary (Ansor-style) | 66.8% | 86.1% | 89.8% | 95.3% |
| Random Search | 67.9% | 85.0% | 90.9% | 95.7% |
| Default Heuristic | 64.8% | 64.8% | 64.8% | 64.8% |

- **Per-GPU highlights (`CrossArch-RL`):** Zero-shot reaches **99.0%** on H100 and **96.2%** on L4; on the hardest out-of-distribution legacy target (Turing **T4**), `CrossArch-RL` reaches **95.9%** at $K=5$ (**+11.6 pts** over Fine-Tuned MLP at 84.3%).
- **Measured wall-clock tuning speedup ($K=5$ vs. full Triton sweep):** **12.2×–12.3×** faster across all 4 GPUs (e.g., T4: **7.7 s** vs. 94.0 s; L4: **2.6 s** vs. 32.3 s; A100: **1.4 s** vs. 17.4 s; H100: **1.8 s** vs. 21.5 s).
- **Cross-vendor transfer (`Apple Metal` $\leftrightarrow$ `Apple CPU` & `NVIDIA`):** On `Metal -> CPU`, `CrossArch-RL` reaches **96.1%** at $K=5$ vs. 74.5% for Fine-Tuned MLP (+21.6 pts); on `Metal -> H100`, it reaches **92.6%** at $K=5$ vs. 85.1% for Fine-Tuned MLP (+7.5 pts).

### 2. Multi-Step Sequential `ScheduleMDP` RL

Evaluating multi-step iterative schedule optimization ($H=8, \gamma=0.90$) across the 4 live NVIDIA GPUs (`results/seq_rl_results.json`):

| Sequential RL Agent | $E=0$ (Zero-Shot) | $E=1$ | $E=2$ | $E=3$ | $E=5$ | T4 OOD ($E=0 \to E=5$) |
|---|---:|---:|---:|---:|---:|---:|
| **CrossArch-RL (TD-SF + GPI)** | 89.3% | 93.4% | **97.0%** | **97.9%** | **99.0%** | **80.1%** $\to$ **98.2%** |
| Offline CQL | **91.8%** | **94.8%** | 96.7% | 97.7% | 98.6% | 64.4% $\to$ 94.4% |
| Offline IQL | 89.7% | 94.0% | 95.4% | 96.6% | 98.0% | 61.3% $\to$ 86.8% |
| Online PPO (GAE) | 80.1% | 84.7% | 88.5% | 90.3% | 93.1% | 78.3% $\to$ 94.0% |
| Random Walk | 52.7% | 83.6% | 89.8% | 93.4% | 96.4% | 44.6% $\to$ 98.6% |

On out-of-distribution Turing **T4**, `CrossArch-RL (TD-SF + GPI)` outperforms Offline IQL by **+18.8 pts** and Offline CQL by **+15.7 pts** zero-shot, and outperforms Online PPO by **+5.9 pts** overall at $E=5$.

### 3. Why the Small Learned Residual MLP Matters: Live H100 PTX / CUBIN Verification

In ablations (`results/real_hardware_results.json`), removing the learned residual (`Linear Physics Only`) collapses zero-shot IQM from **85.9% to 30.6%**. Direct PTX/CUBIN inspection on live **NVIDIA H100 80GB HBM3** (`results/h100_compiler_profile.json`) explains why:

| H100 Workload | Variant | Tile / Stages | Shared Mem | SM Occ. | PTX `cp.async` | Spills | Achieved Perf |
|---|---|---|---:|---:|---:|---:|---:|
| **GEMM MLP** (`1024×4096×1024`) | Oracle | `256×128×32`, `ns=4` | 98.3 KB | 12.5% | 34 | 0 | **413.7 TFLOPS** |
| | **CrossArch-RL (+MLP)** | `64×128×32`, `ns=4` | 49.2 KB | 25.0% | 34 | 0 | **368.9 TFLOPS** |
| | Linear Physics Only | `16×32×32`, `ns=1` | 3.1 KB | 100.0% | **0** | 0 | **39.9 TFLOPS (9.2× slower)** |
| **FlashAttn-2** (`s=1024, d=64`) | Oracle | `64×128`, `ns=3` | 106.5 KB | 12.5% | 32 | 0 | **105.0 TFLOPS** |
| | **CrossArch-RL (+MLP)** | `64×64`, `ns=3` | 57.3 KB | 25.0% | 20 | 0 | **99.5 TFLOPS** |
| | Linear Physics Only | `16×16`, `ns=1` | 4.6 KB | 56.2% | **0** | 0 | **26.1 TFLOPS (3.8× slower)** |
| **RMSNorm** (`1024×8192`) | Oracle | `bn=2048`, `ns=3` | 16 B | 75.0% | 0 | 0 | **1596.4 GB/s** |
| | **CrossArch-RL (+MLP)** | `bn=512`, `ns=3` | 32 B | 100.0% | 0 | 0 | **1379.1 GB/s** |
| | Linear Physics Only | `bn=64`, `ns=1` | 8 B | 100.0% | 0 | 0 | **504.1 GB/s (2.7× slower)** |

All 18 profiled H100 kernels had **`n_spills = 0`**. Without the nonlinear residual, the linear analytic model chases high SM occupancy (56–100%) using tiny 1–4.6 KB single-stage tiles (`ns=1`), which drops PTX `cp.async` software pipelining to **0** and starves Tensor Cores. The 12-dim learned MLP residual captures this hardware-compiler interaction directly from offline traces.

---

## Quickstart & Reproducibility

### 1. Install
```bash
uv sync
# or with pip:
pip install -e .
```

### 2. Run Unit Tests
```bash
python3 tests/test_crossarch_rl.py
```

### 3. Reproduce Live-Silicon Experiments
All scripts run directly on the verified hardware traces in [`data/`](data/):

```bash
# 4-GPU NVIDIA transfer, cross-vendor transfer, and ablations -> results/real_hardware_results.json
python3 scripts/run_real_experiments.py

# Multi-step sequential ScheduleMDP RL (TD-SF+GPI vs. IQL, CQL, PPO) -> results/seq_rl_results.json
python3 scripts/run_seq_rl_experiments.py
```

### 4. Collect Fresh Hardware Traces / Live H100 PTX Profiles (Optional)
```bash
# Local Apple Silicon: custom Metal GPU shaders + native Clang -O3 ARM64 CPU kernels
python3 scripts/collect_real_traces.py --backend cpu
python3 scripts/collect_real_traces.py --backend metal

# Cloud NVIDIA GPUs (T4, L4, A100, H100) via Modal + Triton 3.9
uv run --with modal modal run scripts/modal_triton_bench.py

# Live H100 PTX/CUBIN compiler & occupancy telemetry
uv run --with modal modal run scripts/modal_h100_profile.py
```

---

## Repository Layout

```text
crossarch-rl/
├── pyproject.toml
├── uv.lock
├── src/crossarch_rl/
│   ├── hardware.py            # Architecture specs (M2 Max CPU/Metal, T4, L4, A100, H100)
│   ├── workloads.py           # 5 LLM operators & analytic Speed-of-Light (SOL) bounds
│   ├── schedule_mdp.py        # Feasibility checker, 12-dim bottleneck basis, 27-dim dimensionless features
│   ├── seq_mdp.py             # Multi-step sequential ScheduleGraphMDP over measured hardware traces
│   ├── metal_bench.py         # Real Apple Silicon custom Metal shader compiler & verifier
│   ├── cpu_bench.py           # Real native Clang -O3 / pthreads ARM64 CPU benchmarker & verifier
│   ├── simulator.py           # Legacy analytical roofline simulator (not used in live-silicon tables)
│   ├── eval_stats.py          # rliable IQM, stratified bootstrap 95% CIs, Probability of Improvement
│   └── algorithms/
│       ├── crossarch.py       # CrossArch-RL (conservative expectile SF + Bayesian ridge + GPI-UCB + LCB)
│       ├── baselines.py       # Heuristic, Random, Evolutionary, GP-BO, Neural-UCB, Zero-Shot & FT MLP
│       └── seq_rl.py          # SequentialCrossArchRL (TD-SF+GPI), Offline IQL, Offline CQL, Online PPO
├── scripts/
│   ├── collect_real_traces.py     # Collect verified local Apple Silicon Metal & CPU traces
│   ├── modal_triton_bench.py      # Collect verified live Triton 3.9 traces on Modal (T4/L4/A100/H100)
│   ├── modal_h100_profile.py      # Live H100 PTX/CUBIN register, shared-mem, occupancy & cp.async profiler
│   ├── run_real_experiments.py    # Live-silicon leave-one-GPU-out, cross-vendor & ablation runner
│   ├── run_seq_rl_experiments.py  # Multi-step sequential ScheduleMDP RL benchmark runner
│   └── plot_real_figures.py       # Publication-ready figure generator from results/*.json
├── tests/
│   └── test_crossarch_rl.py       # Unit tests for MDP features, Bayesian updates, and verification
├── data/                          # 3,847 verified live physical hardware traces (6 JSONL files)
└── results/                       # Auditable JSON results (live transfer, sequential RL, H100 PTX telemetry)
```
