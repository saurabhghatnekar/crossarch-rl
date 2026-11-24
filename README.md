# CrossArch-RL

**Few-Shot Cross-Architecture GPU Kernel Schedule RL via Hardware-Factored Value Decomposition**

Target Venue: **ICLR 2027 Workshop** (~Feb 1, 2027) -> **NeurIPS 2027 / RLJ** archival expansion.

## Repository Layout
```text
rl-paper/
├── pyproject.toml
├── src/crossarch_rl/
│   ├── hardware.py        # GPU hardware descriptors (M2 Max, T4, L4, A100, H100) & feasibility
│   ├── workloads.py       # LLM operator suite (GEMM, FlashAttention, RMSNorm, SwiGLU, Softmax)
│   ├── schedule_mdp.py    # Contextual Kernel Schedule MDP & dimensionless physics features
│   ├── simulator.py       # Calibrated analytical + noise roofline simulator for fast RL prototyping
│   ├── metal_bench.py     # Real Apple Silicon custom Metal kernel compiler & anti-hack profiler
│   ├── algorithms/
│   │   ├── crossarch.py   # CrossArch-RL: Offline IQL-SF + Physics-Prior Bayesian GPI + LCB
│   │   └── baselines.py   # Random, Evolutionary/CD, GP-BO, Neural-UCB, Zero-Shot/FT MLP, Ablations
│   └── eval_stats.py      # rliable IQM, stratified bootstrap 95% CIs, paper figure generation
├── scripts/
│   ├── collect_metal_traces.py   # Collect real custom Metal kernel measurements on local Apple Silicon
│   ├── modal_triton_bench.py     # Serverless multi-GPU (T4/L4/A100/H100) Triton benchmark runner
│   └── run_experiments.py        # End-to-end training, cross-arch transfer eval & figure generation
├── tests/                 # Unit tests for MDP, Metal verification, Bayesian updates, & theorem bounds
├── data/                  # Measured hardware traces (JSONL/Parquet)
├── results/               # Experiment logs, tables, and publication PDF figures
└── paper/                 # ICLR 2027 LaTeX manuscript
    ├── main.tex
    └── references.bib
```
