"""
Collect verified real-hardware kernel schedule traces on local Apple M2 Max
(both native ARM64 Clang/pthreads CPU and custom Metal Shading Language GPU).
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from crossarch_rl.cpu_bench import Arm64CpuBenchmarker
from crossarch_rl.hardware import get_hardware
from crossarch_rl.metal_bench import MetalKernelBenchmarker
from crossarch_rl.schedule_mdp import ScheduleConfig, check_feasibility
from crossarch_rl.workloads import WorkloadSpec


def real_hardware_workloads() -> List[WorkloadSpec]:
    """9 core LLM workloads across GEMM, RMSNorm+Res, Softmax, and SwiGLU (FP32)."""
    return [
        WorkloadSpec("gemm_decode_64x1024x1024", "gemm", 64, 1024, 1024, 4),
        WorkloadSpec("gemm_prefill_512x512x512", "gemm", 512, 512, 512, 4),
        WorkloadSpec("gemm_mlp_512x1024x512", "gemm", 512, 1024, 512, 4),
        WorkloadSpec("rmsnorm_1024x2048", "rmsnorm", 1024, 2048, 1, 4),
        WorkloadSpec("rmsnorm_2048x4096", "rmsnorm", 2048, 4096, 1, 4),
        WorkloadSpec("softmax_1024x2048", "softmax", 1024, 2048, 1, 4),
        WorkloadSpec("softmax_1024x4096", "softmax", 1024, 4096, 1, 4),
        WorkloadSpec("swiglu_1024x4096", "swiglu", 1024, 4096, 1, 4),
        WorkloadSpec("swiglu_2048x4096", "swiglu", 2048, 4096, 1, 4),
    ]


def candidate_grid_for(wl: WorkloadSpec, backend: str) -> List[ScheduleConfig]:
    hw = get_hardware("apple_m2_max" if backend == "metal" else "apple_m2_max_cpu")
    configs: List[ScheduleConfig] = []

    if wl.op_type == "gemm":
        bms = (16, 32, 64) if backend == "metal" else (16, 32, 64, 128)
        bns = (16, 32, 64) if backend == "metal" else (16, 32, 64, 128)
        bks = (8, 16, 32)
        warps = (2, 4, 8) if backend == "metal" else (1, 2, 4, 8, 12)
        vecs = (1, 2, 4)
        swizzles = (0, 1) if backend == "metal" else (0,)
        for bm, bn, bk, nw, vw, sw in itertools.product(bms, bns, bks, warps, vecs, swizzles):
            # Keep threadgroup SRAM strictly inside 32KB on Metal
            if backend == "metal" and 4 * (bm * (bk + sw) + bk * (bn + sw)) > 32 * 1024:
                continue
            # Ensure thread layout (tx*ty == nw*32) fits inside bm*bn for Metal
            if backend == "metal" and (bm * bn) < (nw * 32):
                continue
            cfg = ScheduleConfig(bm, bn, bk, nw, 1, vw, 1, sw)
            if backend == "cpu":
                if bm <= wl.m and bn <= wl.n and bk <= wl.k:
                    configs.append(cfg)
            else:
                ok, _ = check_feasibility(wl, cfg, hw)
                if ok:
                    configs.append(cfg)
        # Subsample evenly to ~56 diverse schedules per GEMM shape for fast collection
        if len(configs) > 56:
            step = len(configs) / 56.0
            configs = [configs[int(i * step)] for i in range(56)]

    elif wl.op_type in ("rmsnorm", "softmax"):
        bms = (1, 2, 4) if backend == "metal" else (1, 4, 16, 64)
        bns = (128, 256, 512)
        warps = (1, 2, 4, 8) if backend == "metal" else (1, 2, 4, 8, 12)
        vecs = (1, 2, 4)
        for bm, bn, nw, vw in itertools.product(bms, bns, warps, vecs):
            cfg = ScheduleConfig(bm, bn, 8, nw, 1, vw, 1, 0)
            configs.append(cfg)
        if len(configs) > 48:
            step = len(configs) / 48.0
            configs = [configs[int(i * step)] for i in range(48)]

    elif wl.op_type == "swiglu":
        bms = (1, 2, 4, 8)
        bns = (64, 128, 256)
        warps = (1, 2, 4, 8) if backend == "metal" else (1, 2, 4, 8, 12)
        vecs = (1, 2, 4)
        for bm, bn, nw, vw in itertools.product(bms, bns, warps, vecs):
            cfg = ScheduleConfig(bm, bn, 8, nw, 1, vw, 1, 0)
            configs.append(cfg)
        if len(configs) > 48:
            step = len(configs) / 48.0
            configs = [configs[int(i * step)] for i in range(48)]

    return configs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["cpu", "metal"], required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    bench = (
        Arm64CpuBenchmarker(warmup=args.warmup, iters=args.iters)
        if args.backend == "cpu"
        else MetalKernelBenchmarker(warmup=args.warmup, iters=args.iters)
    )
    workloads = real_hardware_workloads()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_ok = 0
    total_fail = 0
    with out_path.open("w") as f:
        for wl in workloads:
            grid = candidate_grid_for(wl, args.backend)
            ok_wl = 0
            best_ms = float("inf")
            worst_ms = 0.0
            for cfg in grid:
                m = bench.benchmark(wl, cfg)
                rec = {
                    "hardware": m.hardware,
                    "workload": m.workload,
                    "op_type": m.op_type,
                    "shape": [wl.m, wl.n, wl.k],
                    "dtype_bytes": wl.dtype_bytes,
                    "schedule": m.schedule,
                    "verified": m.verified,
                    "latency_ms": m.latency_ms,
                    "iqr_ms": m.iqr_ms,
                    "sol_latency_ms": m.sol_latency_ms,
                    "log_sol_eff": m.log_sol_eff,
                    "max_abs_err": m.max_abs_err,
                }
                if m.verified:
                    f.write(json.dumps(rec) + "\n")
                    ok_wl += 1
                    total_ok += 1
                    best_ms = min(best_ms, m.latency_ms)
                    worst_ms = max(worst_ms, m.latency_ms)
                else:
                    total_fail += 1
            f.flush()
            ratio = (worst_ms / best_ms) if best_ms < float("inf") and best_ms > 0 else 1.0
            print(
                f"[{args.backend}] {wl.name:28s}: {ok_wl:2d}/{len(grid):2d} verified | "
                f"best={best_ms:.3f}ms worst={worst_ms:.3f}ms (range={ratio:.2f}x)",
                flush=True,
            )

    print(f"DONE [{args.backend}]: {total_ok} verified traces saved to {out_path} ({total_fail} skipped).")


if __name__ == "__main__":
    main()
