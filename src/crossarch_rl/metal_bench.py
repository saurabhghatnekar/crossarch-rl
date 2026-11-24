"""Real Apple Silicon Metal (MSL) custom kernel compiler & anti-reward-hacking profiler."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from crossarch_rl.hardware import get_hardware
from crossarch_rl.schedule_mdp import ScheduleConfig
from crossarch_rl.workloads import WorkloadSpec


@dataclass
class KernelMeasurement:
    """Verified real-hardware kernel execution measurement."""

    hardware: str
    workload: str
    op_type: str
    schedule: Dict[str, int]
    verified: bool
    latency_ms: float
    iqr_ms: float
    sol_latency_ms: float
    log_sol_eff: float
    max_abs_err: float
    error_msg: str = ""


def _generate_gemm_msl(cfg: ScheduleConfig) -> Tuple[str, Tuple[int, int, int], Tuple[int, int, int]]:
    """
    Generate parameterized tiled Metal GEMM kernel C[M, N] = A[M, K] @ B[K, N].

    Parameters directly control hardware schedule:
      - BM, BN, BK: threadgroup tile dimensions
      - num_warps: threadgroup threads = num_warps * 32, tiled as (tx, ty)
      - swizzle: +1 bank-conflict padding stride on shared memory tiles
      - vec_width: inner-loop unroll factor (1, 2, 4)
    """
    bm, bn, bk = cfg.block_m, cfg.block_n, cfg.block_k
    total_threads = cfg.num_warps * 32
    # Choose 2D threadgroup layout (tx, ty) such that tx * ty == total_threads
    # and bm % ty == 0, bn % tx == 0
    tx = min(bn, 32)
    while tx > 1 and (total_threads % tx != 0 or bn % tx != 0):
        tx //= 2
    ty = total_threads // tx
    if bm % ty != 0 or ty > bm:
        ty = min(bm, total_threads)
        while ty > 1 and (total_threads % ty != 0 or bm % ty != 0):
            ty //= 2
        tx = total_threads // ty
    if bm % ty != 0 or bn % tx != 0 or tx * ty != total_threads:
        raise ValueError(f"Incompatible Metal GEMM thread layout for BM={bm}, BN={bn}, warps={cfg.num_warps}")

    tm = bm // ty
    tn = bn // tx
    pad = 1 if cfg.swizzle else 0
    bk_stride = bk + pad
    bn_stride = bn + pad
    unroll_pragma = f"#pragma unroll({min(cfg.vec_width, bk)})" if cfg.vec_width > 1 else ""

    src = f"""
    #include <metal_stdlib>
    using namespace metal;

    constant uint BM = {bm};
    constant uint BN = {bn};
    constant uint BK = {bk};
    constant uint TM = {tm};
    constant uint TN = {tn};
    constant uint TX = {tx};
    constant uint TY = {ty};

    kernel void gemm_tiled(
        device const float* A [[buffer(0)]],
        device const float* B [[buffer(1)]],
        device float* C       [[buffer(2)]],
        device const uint* dims [[buffer(3)]],
        uint2 tg_pos [[threadgroup_position_in_grid]],
        uint2 tid    [[thread_position_in_threadgroup]],
        uint  tidx   [[thread_index_in_threadgroup]]
    ) {{
        const uint M = dims[0];
        const uint N = dims[1];
        const uint K = dims[2];

        threadgroup float As[{bm} * {bk_stride}];
        threadgroup float Bs[{bk} * {bn_stride}];

        float acc[TM][TN];
        for (uint i = 0; i < TM; ++i) {{
            for (uint j = 0; j < TN; ++j) {{
                acc[i][j] = 0.0f;
            }}
        }}

        const uint row_base = tg_pos.y * BM;
        const uint col_base = tg_pos.x * BN;
        const uint n_threads = TX * TY;

        for (uint k0 = 0; k0 < K; k0 += BK) {{
            // Cooperative load of A tile (BM x BK)
            for (uint idx = tidx; idx < BM * BK; idx += n_threads) {{
                uint r = idx / BK;
                uint c = idx % BK;
                uint gr = row_base + r;
                uint gc = k0 + c;
                As[r * {bk_stride} + c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
            }}
            // Cooperative load of B tile (BK x BN)
            for (uint idx = tidx; idx < BK * BN; idx += n_threads) {{
                uint r = idx / BN;
                uint c = idx % BN;
                uint gr = k0 + r;
                uint gc = col_base + c;
                Bs[r * {bn_stride} + c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);

            {unroll_pragma}
            for (uint kk = 0; kk < BK; ++kk) {{
                float a_reg[TM];
                float b_reg[TN];
                for (uint i = 0; i < TM; ++i) {{
                    a_reg[i] = As[(tid.y * TM + i) * {bk_stride} + kk];
                }}
                for (uint j = 0; j < TN; ++j) {{
                    b_reg[j] = Bs[kk * {bn_stride} + (tid.x * TN + j)];
                }}
                for (uint i = 0; i < TM; ++i) {{
                    for (uint j = 0; j < TN; ++j) {{
                        acc[i][j] = fma(a_reg[i], b_reg[j], acc[i][j]);
                    }}
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        for (uint i = 0; i < TM; ++i) {{
            uint gr = row_base + tid.y * TM + i;
            if (gr < M) {{
                for (uint j = 0; j < TN; ++j) {{
                    uint gc = col_base + tid.x * TN + j;
                    if (gc < N) {{
                        C[gr * N + gc] = acc[i][j];
                    }}
                }}
            }}
        }}
    }}
    """
    return src, (tx, ty, 1), (bm, bn, 1)


def _generate_rmsnorm_msl(cfg: ScheduleConfig) -> Tuple[str, int]:
    """Parameterized fused RMSNorm + Residual Add kernel in Metal Shading Language."""
    threads = cfg.num_warps * 32
    vw = cfg.vec_width
    rows_per_tg = max(1, min(cfg.block_m, 4))
    src = f"""
    #include <metal_stdlib>
    using namespace metal;

    constant uint N_THREADS = {threads};
    constant uint N_SIMDS = {cfg.num_warps};
    constant uint ROWS_PER_TG = {rows_per_tg};
    constant uint VEC = {vw};

    kernel void rmsnorm_res_fused(
        device const float* X   [[buffer(0)]],
        device const float* Res [[buffer(1)]],
        device const float* W   [[buffer(2)]],
        device float* Y         [[buffer(3)]],
        device const uint* dims [[buffer(4)]],
        uint tg_id   [[threadgroup_position_in_grid]],
        uint tid     [[thread_index_in_threadgroup]],
        uint simd_id [[simdgroup_index_in_threadgroup]],
        uint lane_id [[thread_index_in_simdgroup]]
    ) {{
        const uint M = dims[0];
        const uint N = dims[1];
        threadgroup float shared_sum[N_SIMDS];

        for (uint r_off = 0; r_off < ROWS_PER_TG; ++r_off) {{
            uint row = tg_id * ROWS_PER_TG + r_off;
            if (row >= M) break;
            const uint base = row * N;

            float local_ss = 0.0f;
            if (VEC == 4 && (N % 4 == 0)) {{
                device const float4* X4 = (device const float4*)(X + base);
                device const float4* R4 = (device const float4*)(Res + base);
                uint n4 = N / 4;
                for (uint i = tid; i < n4; i += N_THREADS) {{
                    float4 v = X4[i] + R4[i];
                    local_ss += dot(v, v);
                }}
            }} else if (VEC == 2 && (N % 2 == 0)) {{
                device const float2* X2 = (device const float2*)(X + base);
                device const float2* R2 = (device const float2*)(Res + base);
                uint n2 = N / 2;
                for (uint i = tid; i < n2; i += N_THREADS) {{
                    float2 v = X2[i] + R2[i];
                    local_ss += dot(v, v);
                }}
            }} else {{
                for (uint i = tid; i < N; i += N_THREADS) {{
                    float v = X[base + i] + Res[base + i];
                    local_ss = fma(v, v, local_ss);
                }}
            }}

            float sg_sum = simd_sum(local_ss);
            if (lane_id == 0) {{
                shared_sum[simd_id] = sg_sum;
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float total_ss = 0.0f;
            if (simd_id == 0) {{
                float val = (lane_id < N_SIMDS) ? shared_sum[lane_id] : 0.0f;
                total_ss = simd_sum(val);
                if (lane_id == 0) {{
                    shared_sum[0] = rsqrt(total_ss / float(N) + 1e-5f);
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float inv_rms = shared_sum[0];

            if (VEC == 4 && (N % 4 == 0)) {{
                device const float4* X4 = (device const float4*)(X + base);
                device const float4* R4 = (device const float4*)(Res + base);
                device const float4* W4 = (device const float4*)(W);
                device float4* Y4 = (device float4*)(Y + base);
                uint n4 = N / 4;
                for (uint i = tid; i < n4; i += N_THREADS) {{
                    Y4[i] = (X4[i] + R4[i]) * inv_rms * W4[i];
                }}
            }} else if (VEC == 2 && (N % 2 == 0)) {{
                device const float2* X2 = (device const float2*)(X + base);
                device const float2* R2 = (device const float2*)(Res + base);
                device const float2* W2 = (device const float2*)(W);
                device float2* Y2 = (device float2*)(Y + base);
                uint n2 = N / 2;
                for (uint i = tid; i < n2; i += N_THREADS) {{
                    Y2[i] = (X2[i] + R2[i]) * inv_rms * W2[i];
                }}
            }} else {{
                for (uint i = tid; i < N; i += N_THREADS) {{
                    Y[base + i] = (X[base + i] + Res[base + i]) * inv_rms * W[i];
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
    }}
    """
    return src, rows_per_tg


def _generate_softmax_msl(cfg: ScheduleConfig) -> Tuple[str, int]:
    """Parameterized fused row-wise Softmax kernel in Metal Shading Language."""
    threads = cfg.num_warps * 32
    vw = cfg.vec_width
    rows_per_tg = max(1, min(cfg.block_m, 4))
    src = f"""
    #include <metal_stdlib>
    using namespace metal;

    constant uint N_THREADS = {threads};
    constant uint N_SIMDS = {cfg.num_warps};
    constant uint ROWS_PER_TG = {rows_per_tg};
    constant uint VEC = {vw};

    kernel void softmax_fused(
        device const float* X   [[buffer(0)]],
        device float* Y         [[buffer(1)]],
        device const uint* dims [[buffer(2)]],
        uint tg_id   [[threadgroup_position_in_grid]],
        uint tid     [[thread_index_in_threadgroup]],
        uint simd_id [[simdgroup_index_in_threadgroup]],
        uint lane_id [[thread_index_in_simdgroup]]
    ) {{
        const uint M = dims[0];
        const uint N = dims[1];
        threadgroup float shared_buf[N_SIMDS];

        for (uint r_off = 0; r_off < ROWS_PER_TG; ++r_off) {{
            uint row = tg_id * ROWS_PER_TG + r_off;
            if (row >= M) break;
            const uint base = row * N;

            float local_max = -INFINITY;
            if (VEC == 4 && (N % 4 == 0)) {{
                device const float4* X4 = (device const float4*)(X + base);
                uint n4 = N / 4;
                for (uint i = tid; i < n4; i += N_THREADS) {{
                    float4 v = X4[i];
                    local_max = max(local_max, max(max(v.x, v.y), max(v.z, v.w)));
                }}
            }} else {{
                for (uint i = tid; i < N; i += N_THREADS) {{
                    local_max = max(local_max, X[base + i]);
                }}
            }}

            float sg_max = simd_max(local_max);
            if (lane_id == 0) shared_buf[simd_id] = sg_max;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (simd_id == 0) {{
                float val = (lane_id < N_SIMDS) ? shared_buf[lane_id] : -INFINITY;
                float rmax = simd_max(val);
                if (lane_id == 0) shared_buf[0] = rmax;
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float row_max = shared_buf[0];

            float local_sum = 0.0f;
            if (VEC == 4 && (N % 4 == 0)) {{
                device const float4* X4 = (device const float4*)(X + base);
                uint n4 = N / 4;
                for (uint i = tid; i < n4; i += N_THREADS) {{
                    float4 e = exp(X4[i] - row_max);
                    local_sum += (e.x + e.y + e.z + e.w);
                }}
            }} else {{
                for (uint i = tid; i < N; i += N_THREADS) {{
                    local_sum += exp(X[base + i] - row_max);
                }}
            }}

            float sg_sum = simd_sum(local_sum);
            if (lane_id == 0) shared_buf[simd_id] = sg_sum;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (simd_id == 0) {{
                float val = (lane_id < N_SIMDS) ? shared_buf[lane_id] : 0.0f;
                float rsum = simd_sum(val);
                if (lane_id == 0) shared_buf[0] = 1.0f / rsum;
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float inv_sum = shared_buf[0];

            if (VEC == 4 && (N % 4 == 0)) {{
                device const float4* X4 = (device const float4*)(X + base);
                device float4* Y4 = (device float4*)(Y + base);
                uint n4 = N / 4;
                for (uint i = tid; i < n4; i += N_THREADS) {{
                    Y4[i] = exp(X4[i] - row_max) * inv_sum;
                }}
            }} else {{
                for (uint i = tid; i < N; i += N_THREADS) {{
                    Y[base + i] = exp(X[base + i] - row_max) * inv_sum;
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
    }}
    """
    return src, rows_per_tg


def _generate_swiglu_msl(cfg: ScheduleConfig) -> Tuple[str, int]:
    """Parameterized fused SwiGLU (silu(gate) * up) kernel in Metal Shading Language."""
    vw = cfg.vec_width
    items_per_thread = max(1, min(cfg.block_m, 8))
    src = f"""
    #include <metal_stdlib>
    using namespace metal;

    constant uint VEC = {vw};
    constant uint IPT = {items_per_thread};

    kernel void swiglu_fused(
        device const float* Gate [[buffer(0)]],
        device const float* Up   [[buffer(1)]],
        device float* Out        [[buffer(2)]],
        device const uint* dims  [[buffer(3)]],
        uint gid [[thread_position_in_grid]],
        uint gsize [[threads_per_grid]]
    ) {{
        const uint total = dims[0];
        if (VEC == 4 && (total % 4 == 0)) {{
            device const float4* G4 = (device const float4*)Gate;
            device const float4* U4 = (device const float4*)Up;
            device float4* O4 = (device float4*)Out;
            uint n4 = total / 4;
            for (uint k = 0; k < IPT; ++k) {{
                uint idx = gid + k * gsize;
                if (idx < n4) {{
                    float4 g = G4[idx];
                    float4 s = g / (1.0f + exp(-g));
                    O4[idx] = s * U4[idx];
                }}
            }}
        }} else if (VEC == 2 && (total % 2 == 0)) {{
            device const float2* G2 = (device const float2*)Gate;
            device const float2* U2 = (device const float2*)Up;
            device float2* O2 = (device float2*)Out;
            uint n2 = total / 2;
            for (uint k = 0; k < IPT; ++k) {{
                uint idx = gid + k * gsize;
                if (idx < n2) {{
                    float2 g = G2[idx];
                    float2 s = g / (1.0f + exp(-g));
                    O2[idx] = s * U2[idx];
                }}
            }}
        }} else {{
            for (uint k = 0; k < IPT; ++k) {{
                uint idx = gid + k * gsize;
                if (idx < total) {{
                    float g = Gate[idx];
                    Out[idx] = (g / (1.0f + exp(-g))) * Up[idx];
                }}
            }}
        }}
    }}
    """
    return src, items_per_thread


class MetalKernelBenchmarker:
    """
    Real Apple Silicon Metal GPU benchmarker with anti-reward-hacking controls:
      1. Numerical verification against PyTorch FP32 reference before timing
      2. Explicit 64 MB SLC cache eviction sweep between timed iterations
      3. Full torch.mps.synchronize() barrier around every timed execution
      4. Median + IQR latency reporting across timed iterations
    """

    def __init__(self, warmup: int = 5, iters: int = 15) -> None:
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "PyTorch MPS Metal GPU is not available in this process. "
                "Run on Apple Silicon outside GPU-restricted sandboxes."
            )
        self.hw = get_hardware("apple_m2_max")
        self.warmup = warmup
        self.iters = iters
        # 64 MB SLC eviction buffer (> 48 MB M2 Max SLC)
        self._flush_buf = torch.ones((16 * 1024 * 1024,), device="mps", dtype=torch.float32)
        self._shader_cache: Dict[Tuple, object] = {}

    def _evict_cache(self) -> None:
        self._flush_buf.mul_(1.00001)
        torch.mps.synchronize()

    def benchmark(self, workload: WorkloadSpec, cfg: ScheduleConfig) -> KernelMeasurement:
        sol_ms = workload.sol_latency_s(self.hw) * 1e3
        torch.manual_seed(0)

        try:
            if workload.op_type == "gemm":
                return self._bench_gemm(workload, cfg, sol_ms)
            if workload.op_type == "rmsnorm":
                return self._bench_rmsnorm(workload, cfg, sol_ms)
            if workload.op_type == "softmax":
                return self._bench_softmax(workload, cfg, sol_ms)
            if workload.op_type == "swiglu":
                return self._bench_swiglu(workload, cfg, sol_ms)
            raise ValueError(f"Metal kernel generator not implemented for op_type={workload.op_type}")
        except Exception as exc:
            return KernelMeasurement(
                hardware="apple_m2_max",
                workload=workload.name,
                op_type=workload.op_type,
                schedule=cfg.as_dict(),
                verified=False,
                latency_ms=float("inf"),
                iqr_ms=0.0,
                sol_latency_ms=sol_ms,
                log_sol_eff=-6.0,
                max_abs_err=float("inf"),
                error_msg=str(exc),
            )

    def _time_runner(self, launch_fn) -> Tuple[float, float]:
        for _ in range(self.warmup):
            launch_fn()
        torch.mps.synchronize()

        times_ms = []
        for _ in range(self.iters):
            self._evict_cache()
            t0 = time.perf_counter()
            launch_fn()
            torch.mps.synchronize()
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1e3)

        arr = np.array(times_ms, dtype=np.float64)
        med = float(np.median(arr))
        iqr = float(np.percentile(arr, 75) - np.percentile(arr, 25))
        return med, iqr

    def _bench_gemm(self, wl: WorkloadSpec, cfg: ScheduleConfig, sol_ms: float) -> KernelMeasurement:
        src, (tx, ty, _), (bm, bn, _) = _generate_gemm_msl(cfg)
        key = ("gemm", cfg.as_tuple())
        if key not in self._shader_cache:
            self._shader_cache[key] = torch.mps.compile_shader(src)
        lib = self._shader_cache[key]

        A = torch.randn((wl.m, wl.k), device="mps", dtype=torch.float32) * 0.1
        B = torch.randn((wl.k, wl.n), device="mps", dtype=torch.float32) * 0.1
        C = torch.empty((wl.m, wl.n), device="mps", dtype=torch.float32)
        dims = torch.tensor([wl.m, wl.n, wl.k], device="mps", dtype=torch.uint32)

        grid_x = math.ceil(wl.n / bn) * tx
        grid_y = math.ceil(wl.m / bm) * ty

        def launch():
            lib.gemm_tiled(A, B, C, dims, threads=(grid_x, grid_y, 1), group_size=(tx, ty, 1))

        launch()
        torch.mps.synchronize()
        ref = torch.matmul(A, B)
        max_err = float((C - ref).abs().max().item())
        verified = bool(torch.allclose(C, ref, rtol=1e-2, atol=1e-2))
        if not verified:
            raise ValueError(f"Numerical verification failed: max_abs_err={max_err:.4e}")

        med_ms, iqr_ms = self._time_runner(launch)
        log_eff = float(math.log(max(sol_ms / max(med_ms, 1e-6), 1e-6)))
        return KernelMeasurement(
            hardware="apple_m2_max",
            workload=wl.name,
            op_type=wl.op_type,
            schedule=cfg.as_dict(),
            verified=True,
            latency_ms=med_ms,
            iqr_ms=iqr_ms,
            sol_latency_ms=sol_ms,
            log_sol_eff=min(-0.001, log_eff),
            max_abs_err=max_err,
        )

    def _bench_rmsnorm(self, wl: WorkloadSpec, cfg: ScheduleConfig, sol_ms: float) -> KernelMeasurement:
        src, rows_per_tg = _generate_rmsnorm_msl(cfg)
        key = ("rmsnorm", cfg.as_tuple())
        if key not in self._shader_cache:
            self._shader_cache[key] = torch.mps.compile_shader(src)
        lib = self._shader_cache[key]

        X = torch.randn((wl.m, wl.n), device="mps", dtype=torch.float32)
        Res = torch.randn((wl.m, wl.n), device="mps", dtype=torch.float32)
        W = torch.randn((wl.n,), device="mps", dtype=torch.float32)
        Y = torch.empty_like(X)
        dims = torch.tensor([wl.m, wl.n], device="mps", dtype=torch.uint32)

        n_threads = cfg.num_warps * 32
        n_groups = math.ceil(wl.m / rows_per_tg)

        def launch():
            lib.rmsnorm_res_fused(X, Res, W, Y, dims, threads=(n_groups * n_threads,), group_size=(n_threads,))

        launch()
        torch.mps.synchronize()
        h = X + Res
        ref = h * torch.rsqrt(h.pow(2).mean(dim=-1, keepdim=True) + 1e-5) * W
        max_err = float((Y - ref).abs().max().item())
        verified = bool(torch.allclose(Y, ref, rtol=1e-2, atol=1e-2))
        if not verified:
            raise ValueError(f"Numerical verification failed: max_abs_err={max_err:.4e}")

        med_ms, iqr_ms = self._time_runner(launch)
        log_eff = float(math.log(max(sol_ms / max(med_ms, 1e-6), 1e-6)))
        return KernelMeasurement(
            hardware="apple_m2_max",
            workload=wl.name,
            op_type=wl.op_type,
            schedule=cfg.as_dict(),
            verified=True,
            latency_ms=med_ms,
            iqr_ms=iqr_ms,
            sol_latency_ms=sol_ms,
            log_sol_eff=min(-0.001, log_eff),
            max_abs_err=max_err,
        )

    def _bench_softmax(self, wl: WorkloadSpec, cfg: ScheduleConfig, sol_ms: float) -> KernelMeasurement:
        src, rows_per_tg = _generate_softmax_msl(cfg)
        key = ("softmax", cfg.as_tuple())
        if key not in self._shader_cache:
            self._shader_cache[key] = torch.mps.compile_shader(src)
        lib = self._shader_cache[key]

        X = torch.randn((wl.m, wl.n), device="mps", dtype=torch.float32)
        Y = torch.empty_like(X)
        dims = torch.tensor([wl.m, wl.n], device="mps", dtype=torch.uint32)

        n_threads = cfg.num_warps * 32
        n_groups = math.ceil(wl.m / rows_per_tg)

        def launch():
            lib.softmax_fused(X, Y, dims, threads=(n_groups * n_threads,), group_size=(n_threads,))

        launch()
        torch.mps.synchronize()
        ref = torch.softmax(X, dim=-1)
        max_err = float((Y - ref).abs().max().item())
        verified = bool(torch.allclose(Y, ref, rtol=1e-2, atol=1e-2))
        if not verified:
            raise ValueError(f"Numerical verification failed: max_abs_err={max_err:.4e}")

        med_ms, iqr_ms = self._time_runner(launch)
        log_eff = float(math.log(max(sol_ms / max(med_ms, 1e-6), 1e-6)))
        return KernelMeasurement(
            hardware="apple_m2_max",
            workload=wl.name,
            op_type=wl.op_type,
            schedule=cfg.as_dict(),
            verified=True,
            latency_ms=med_ms,
            iqr_ms=iqr_ms,
            sol_latency_ms=sol_ms,
            log_sol_eff=min(-0.001, log_eff),
            max_abs_err=max_err,
        )

    def _bench_swiglu(self, wl: WorkloadSpec, cfg: ScheduleConfig, sol_ms: float) -> KernelMeasurement:
        src, ipt = _generate_swiglu_msl(cfg)
        key = ("swiglu", cfg.as_tuple())
        if key not in self._shader_cache:
            self._shader_cache[key] = torch.mps.compile_shader(src)
        lib = self._shader_cache[key]

        G = torch.randn((wl.m, wl.n), device="mps", dtype=torch.float32)
        U = torch.randn((wl.m, wl.n), device="mps", dtype=torch.float32)
        O = torch.empty_like(G)
        total = wl.m * wl.n
        dims = torch.tensor([total], device="mps", dtype=torch.uint32)

        n_threads_group = cfg.num_warps * 32
        work_items = math.ceil(total / max(cfg.vec_width * ipt, 1))
        total_grid = math.ceil(work_items / n_threads_group) * n_threads_group

        def launch():
            lib.swiglu_fused(G, U, O, dims, threads=(total_grid,), group_size=(n_threads_group,))

        launch()
        torch.mps.synchronize()
        ref = torch.nn.functional.silu(G) * U
        max_err = float((O - ref).abs().max().item())
        verified = bool(torch.allclose(O, ref, rtol=1e-2, atol=1e-2))
        if not verified:
            raise ValueError(f"Numerical verification failed: max_abs_err={max_err:.4e}")

        med_ms, iqr_ms = self._time_runner(launch)
        log_eff = float(math.log(max(sol_ms / max(med_ms, 1e-6), 1e-6)))
        return KernelMeasurement(
            hardware="apple_m2_max",
            workload=wl.name,
            op_type=wl.op_type,
            schedule=cfg.as_dict(),
            verified=True,
            latency_ms=med_ms,
            iqr_ms=iqr_ms,
            sol_latency_ms=sol_ms,
            log_sol_eff=min(-0.001, log_eff),
            max_abs_err=max_err,
        )
