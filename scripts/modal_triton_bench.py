"""
Serverless Modal + OpenAI Triton 5-Operator Multi-GPU Benchmark Runner for CrossArch-RL.

Collects 100% REAL, numerically verified, L2-cache-flushed OpenAI Triton kernel
measurements across NVIDIA T4 (SM75), L4 (SM89), A100 (SM80), and H100 (SM90)
spanning 5 core LLM operators:
  1. Tiled GEMM (decode, prefill, MLP shapes)
  2. Fused RMSNorm + Residual Add
  3. Fused Online Softmax
  4. Fused SwiGLU (silu(gate) * up)
  5. FlashAttention-2 style Tiled Attention (B*H*S, S_kv, head_dim)

Also records wall-clock compile + benchmark time per schedule so CrossArch-RL can
be compared directly against @triton.autotune end-to-end tuning time!

Usage:
    pip install modal
    modal setup
    modal run scripts/modal_triton_bench.py --gpu ALL
    # Or run one GPU at a time:
    modal run scripts/modal_triton_bench.py --gpu A100
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List

try:
    import modal
except ImportError:
    modal = None


# Hardware datasheet specs for computing exact SOL latency on each NVIDIA GPU
_NVIDIA_SPECS: Dict[str, Dict[str, float]] = {
    "nvidia_t4": {"fp16_tf": 65.0, "bw_gbps": 320.0},
    "nvidia_l4": {"fp16_tf": 121.0, "bw_gbps": 300.0},
    "nvidia_a100": {"fp16_tf": 312.0, "bw_gbps": 2039.0},
    "nvidia_h100": {"fp16_tf": 989.0, "bw_gbps": 3350.0},
}


def _sol_ms(op_type: str, m: int, n: int, k: int, dtype_bytes: int, gpu_key: str) -> float:
    spec = _NVIDIA_SPECS[gpu_key]
    b = float(dtype_bytes)
    if op_type == "gemm":
        flops = 2.0 * m * n * k
        nbytes = b * (m * k + k * n + m * n)
    elif op_type == "rmsnorm":
        flops = 6.0 * m * n
        nbytes = b * (3.0 * m * n + n)
    elif op_type == "softmax":
        flops = 5.0 * m * n
        nbytes = b * (2.0 * m * n)
    elif op_type == "swiglu":
        flops = 6.0 * m * n
        nbytes = b * (3.0 * m * n)
    elif op_type == "flash_attn":
        flops = m * n * (4.0 * k + 5.0)
        nbytes = b * (4.0 * m * k)
    else:
        raise ValueError(op_type)
    comp_s = flops / (spec["fp16_tf"] * 1e12)
    mem_s = nbytes / (spec["bw_gbps"] * 1e9)
    return max(comp_s, mem_s, 1e-7) * 1e3


if modal is not None:
    app = modal.App("crossarch-rl-triton-5op")
    image = modal.Image.debian_slim(python_version="3.11").pip_install(
        "torch>=2.4", "triton>=3.0", "numpy>=1.26"
    )

    def _run_full_triton_suite(gpu_key: str, max_per_op: int = 72) -> List[Dict[str, Any]]:
        import itertools
        import numpy as np
        import torch
        import triton
        import triton.language as tl

        device_name = torch.cuda.get_device_name(0)
        print(f"[{gpu_key}] Running on live silicon: {device_name}", flush=True)

        # ---- 1. Parameterized Tiled GEMM ----
        @triton.jit
        def gemm_kernel(
            a_ptr, b_ptr, c_ptr, M, N, K,
            stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
            GROUP_M: tl.constexpr,
        ):
            pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BLOCK_M)
            num_pid_n = tl.cdiv(N, BLOCK_N)
            num_pid_in_group = GROUP_M * num_pid_n
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * GROUP_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
            pid_m = first_pid_m + (pid % group_size_m)
            pid_n = (pid % num_pid_in_group) // group_size_m

            offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
            offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
            offs_k = tl.arange(0, BLOCK_K)
            a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
            b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k_i in range(0, tl.cdiv(K, BLOCK_K)):
                a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k_i * BLOCK_K, other=0.0)
                b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k_i * BLOCK_K, other=0.0)
                acc = tl.dot(a, b, acc)
                a_ptrs += BLOCK_K * stride_ak
                b_ptrs += BLOCK_K * stride_bk
            c = acc.to(tl.float16)
            offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
            c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
            tl.store(c_ptrs, c, mask=c_mask)

        # ---- 2. Parameterized Fused RMSNorm + Residual Add ----
        @triton.jit
        def rmsnorm_kernel(
            x_ptr, res_ptr, w_ptr, y_ptr, stride, N,
            BLOCK_N: tl.constexpr,
        ):
            row = tl.program_id(0)
            row_start = row * stride
            ss = tl.zeros([BLOCK_N], dtype=tl.float32)
            for off in range(0, N, BLOCK_N):
                cols = off + tl.arange(0, BLOCK_N)
                mask = cols < N
                x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
                r = tl.load(res_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
                h = x + r
                ss += h * h
            inv_rms = tl.rsqrt(tl.sum(ss, axis=0) / N + 1e-5)
            for off in range(0, N, BLOCK_N):
                cols = off + tl.arange(0, BLOCK_N)
                mask = cols < N
                x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
                r = tl.load(res_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
                w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
                y = (x + r) * inv_rms * w
                tl.store(y_ptr + row_start + cols, y.to(tl.float16), mask=mask)

        # ---- 3. Parameterized Fused Online Softmax ----
        @triton.jit
        def softmax_kernel(
            x_ptr, y_ptr, stride, N,
            BLOCK_N: tl.constexpr,
        ):
            row = tl.program_id(0)
            row_start = row * stride
            row_max = tl.full([BLOCK_N], -float("inf"), dtype=tl.float32)
            for off in range(0, N, BLOCK_N):
                cols = off + tl.arange(0, BLOCK_N)
                mask = cols < N
                x = tl.load(x_ptr + row_start + cols, mask=mask, other=-float("inf")).to(tl.float32)
                row_max = tl.maximum(row_max, x)
            m_val = tl.max(row_max, axis=0)
            row_sum = tl.zeros([BLOCK_N], dtype=tl.float32)
            for off in range(0, N, BLOCK_N):
                cols = off + tl.arange(0, BLOCK_N)
                mask = cols < N
                x = tl.load(x_ptr + row_start + cols, mask=mask, other=-float("inf")).to(tl.float32)
                row_sum += tl.where(mask, tl.exp(x - m_val), 0.0)
            inv_sum = 1.0 / tl.sum(row_sum, axis=0)
            for off in range(0, N, BLOCK_N):
                cols = off + tl.arange(0, BLOCK_N)
                mask = cols < N
                x = tl.load(x_ptr + row_start + cols, mask=mask, other=-float("inf")).to(tl.float32)
                y = tl.exp(x - m_val) * inv_sum
                tl.store(y_ptr + row_start + cols, y.to(tl.float16), mask=mask)

        # ---- 4. Parameterized Fused SwiGLU ----
        @triton.jit
        def swiglu_kernel(
            g_ptr, u_ptr, o_ptr, total,
            BLOCK_SIZE: tl.constexpr,
        ):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < total
            g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            u = tl.load(u_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            o = (g / (1.0 + tl.exp(-g))) * u
            tl.store(o_ptr + offs, o.to(tl.float16), mask=mask)

        # ---- 5. Parameterized FlashAttention-2 Tiled Kernel ----
        @triton.jit
        def flash_attn_kernel(
            Q, K, V, Out,
            stride_qz, stride_qm, stride_qk,
            stride_kz, stride_kn, stride_kk,
            stride_vz, stride_vn, stride_vk,
            stride_oz, stride_om, stride_ok,
            S, sm_scale,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
        ):
            pid_m = tl.program_id(0)
            pid_z = tl.program_id(1)
            offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_d = tl.arange(0, HEAD_DIM)
            q_ptrs = Q + pid_z * stride_qz + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
            q = tl.load(q_ptrs, mask=offs_m[:, None] < S, other=0.0)

            m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
            l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
            acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

            for start_n in range(0, S, BLOCK_N):
                offs_n = start_n + tl.arange(0, BLOCK_N)
                k_ptrs = K + pid_z * stride_kz + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
                v_ptrs = V + pid_z * stride_vz + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
                k = tl.load(k_ptrs, mask=offs_n[:, None] < S, other=0.0)
                v = tl.load(v_ptrs, mask=offs_n[:, None] < S, other=0.0)

                qk = tl.dot(q, tl.trans(k)) * sm_scale
                qk = tl.where(offs_n[None, :] < S, qk, -float("inf"))
                m_ij = tl.maximum(m_i, tl.max(qk, 1))
                p = tl.exp(qk - m_ij[:, None])
                l_ij = tl.sum(p, 1)
                alpha = tl.exp(m_i - m_ij)
                acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
                l_i = l_i * alpha + l_ij
                m_i = m_ij

            acc = acc / l_i[:, None]
            o_ptrs = Out + pid_z * stride_oz + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
            tl.store(o_ptrs, acc.to(tl.float16), mask=offs_m[:, None] < S)

        records: List[Dict[str, Any]] = []
        rng = np.random.default_rng(42)

        # ---- Benchmark GEMM (4 realistic LLM shapes) ----
        gemm_shapes = [
            ("gemm_decode_64x4096x4096", 64, 4096, 4096),
            ("gemm_prefill_512x2048x2048", 512, 2048, 2048),
            ("gemm_prefill_1024x1024x1024", 1024, 1024, 1024),
            ("gemm_mlp_1024x4096x1024", 1024, 4096, 1024),
        ]
        gemm_grid = [
            (bm, bn, bk, nw, ns, 4, 1, sw)
            for bm, bn, bk, nw, ns, sw in itertools.product(
                (16, 32, 64, 128, 256), (16, 32, 64, 128, 256), (16, 32, 64), (2, 4, 8), (1, 2, 3, 4), (0, 1)
            )
            if bm * bn <= 256 * 128
        ]

        for name, M, N, K in gemm_shapes:
            A = torch.randn((M, K), device="cuda", dtype=torch.float16) * 0.05
            B = torch.randn((K, N), device="cuda", dtype=torch.float16) * 0.05
            ref = torch.matmul(A, B)
            sol = _sol_ms("gemm", M, N, K, 2, gpu_key)
            valid_g = [c for c in gemm_grid if c[0] <= M and c[1] <= N and c[2] <= K]
            pick_idx = rng.choice(len(valid_g), size=min(max_per_op, len(valid_g)), replace=False)
            ok_cnt = 0
            for idx in sorted(pick_idx):
                bm, bn, bk, nw, ns, vw, sk, sw = valid_g[int(idx)]
                C = torch.empty((M, N), device="cuda", dtype=torch.float16)
                grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
                group_m = 8 if sw else 1
                t_c0 = time.perf_counter()
                try:
                    gemm_kernel[grid](
                        A, B, C, M, N, K,
                        A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
                        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=group_m,
                        num_warps=nw, num_stages=ns,
                    )
                    torch.cuda.synchronize()
                    compile_s = time.perf_counter() - t_c0
                    max_err = float((C - ref).abs().max().item())
                    if not torch.allclose(C, ref, rtol=5e-2, atol=5e-2):
                        continue
                    med_ms, q25, q75 = triton.testing.do_bench(
                        lambda: gemm_kernel[grid](
                            A, B, C, M, N, K,
                            A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
                            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=group_m,
                            num_warps=nw, num_stages=ns,
                        ),
                        quantiles=[0.5, 0.25, 0.75],
                    )
                    log_eff = float(math.log(max(sol / max(med_ms, 1e-6), 1e-6)))
                    records.append({
                        "hardware": gpu_key, "device_name": device_name,
                        "workload": name, "op_type": "gemm", "shape": [M, N, K], "dtype_bytes": 2,
                        "schedule": {"block_m": bm, "block_n": bn, "block_k": bk, "num_warps": nw, "num_stages": ns, "vec_width": vw, "split_k": sk, "swizzle": sw},
                        "verified": True, "latency_ms": float(med_ms), "iqr_ms": float(q75 - q25),
                        "compile_s": float(compile_s), "sol_latency_ms": float(sol),
                        "log_sol_eff": min(-0.001, log_eff), "max_abs_err": max_err,
                    })
                    ok_cnt += 1
                except Exception:
                    continue
            print(f"[{gpu_key}] {name:28s}: {ok_cnt} verified", flush=True)

        # ---- Benchmark RMSNorm + Residual (2 shapes) ----
        rmsnorm_shapes = [("rmsnorm_2048x4096", 2048, 4096), ("rmsnorm_1024x8192", 1024, 8192)]
        row_grid = list(itertools.product((1,), (64, 128, 256, 512, 1024, 2048), (8,), (1, 2, 4, 8), (1, 2, 3), (4,), (1,), (0,)))
        for name, M, N in rmsnorm_shapes:
            X = torch.randn((M, N), device="cuda", dtype=torch.float16)
            Res = torch.randn((M, N), device="cuda", dtype=torch.float16)
            W = torch.randn((N,), device="cuda", dtype=torch.float16)
            h = (X + Res).float()
            ref = (h * torch.rsqrt(h.pow(2).mean(dim=-1, keepdim=True) + 1e-5) * W.float()).half()
            sol = _sol_ms("rmsnorm", M, N, 1, 2, gpu_key)
            ok_cnt = 0
            for bm, bn, bk, nw, ns, vw, sk, sw in row_grid:
                if bn > N:
                    continue
                Y = torch.empty_like(X)
                t_c0 = time.perf_counter()
                try:
                    rmsnorm_kernel[(M,)](X, Res, W, Y, X.stride(0), N, BLOCK_N=bn, num_warps=nw, num_stages=ns)
                    torch.cuda.synchronize()
                    compile_s = time.perf_counter() - t_c0
                    max_err = float((Y - ref).abs().max().item())
                    if not torch.allclose(Y, ref, rtol=5e-2, atol=5e-2):
                        continue
                    med_ms, q25, q75 = triton.testing.do_bench(
                        lambda: rmsnorm_kernel[(M,)](X, Res, W, Y, X.stride(0), N, BLOCK_N=bn, num_warps=nw, num_stages=ns),
                        quantiles=[0.5, 0.25, 0.75],
                    )
                    log_eff = float(math.log(max(sol / max(med_ms, 1e-6), 1e-6)))
                    records.append({
                        "hardware": gpu_key, "device_name": device_name,
                        "workload": name, "op_type": "rmsnorm", "shape": [M, N, 1], "dtype_bytes": 2,
                        "schedule": {"block_m": bm, "block_n": bn, "block_k": bk, "num_warps": nw, "num_stages": ns, "vec_width": vw, "split_k": sk, "swizzle": sw},
                        "verified": True, "latency_ms": float(med_ms), "iqr_ms": float(q75 - q25),
                        "compile_s": float(compile_s), "sol_latency_ms": float(sol),
                        "log_sol_eff": min(-0.001, log_eff), "max_abs_err": max_err,
                    })
                    ok_cnt += 1
                except Exception:
                    continue
            print(f"[{gpu_key}] {name:28s}: {ok_cnt} verified", flush=True)

        # ---- Benchmark Online Softmax (2 shapes) ----
        softmax_shapes = [("softmax_2048x4096", 2048, 4096), ("softmax_1024x8192", 1024, 8192)]
        for name, M, N in softmax_shapes:
            X = torch.randn((M, N), device="cuda", dtype=torch.float16)
            ref = torch.softmax(X.float(), dim=-1).half()
            sol = _sol_ms("softmax", M, N, 1, 2, gpu_key)
            ok_cnt = 0
            for bm, bn, bk, nw, ns, vw, sk, sw in row_grid:
                if bn > N:
                    continue
                Y = torch.empty_like(X)
                t_c0 = time.perf_counter()
                try:
                    softmax_kernel[(M,)](X, Y, X.stride(0), N, BLOCK_N=bn, num_warps=nw, num_stages=ns)
                    torch.cuda.synchronize()
                    compile_s = time.perf_counter() - t_c0
                    max_err = float((Y - ref).abs().max().item())
                    if not torch.allclose(Y, ref, rtol=5e-2, atol=5e-2):
                        continue
                    med_ms, q25, q75 = triton.testing.do_bench(
                        lambda: softmax_kernel[(M,)](X, Y, X.stride(0), N, BLOCK_N=bn, num_warps=nw, num_stages=ns),
                        quantiles=[0.5, 0.25, 0.75],
                    )
                    log_eff = float(math.log(max(sol / max(med_ms, 1e-6), 1e-6)))
                    records.append({
                        "hardware": gpu_key, "device_name": device_name,
                        "workload": name, "op_type": "softmax", "shape": [M, N, 1], "dtype_bytes": 2,
                        "schedule": {"block_m": bm, "block_n": bn, "block_k": bk, "num_warps": nw, "num_stages": ns, "vec_width": vw, "split_k": sk, "swizzle": sw},
                        "verified": True, "latency_ms": float(med_ms), "iqr_ms": float(q75 - q25),
                        "compile_s": float(compile_s), "sol_latency_ms": float(sol),
                        "log_sol_eff": min(-0.001, log_eff), "max_abs_err": max_err,
                    })
                    ok_cnt += 1
                except Exception:
                    continue
            print(f"[{gpu_key}] {name:28s}: {ok_cnt} verified", flush=True)

        # ---- Benchmark Fused SwiGLU (2 shapes) ----
        swiglu_shapes = [("swiglu_2048x4096", 2048, 4096), ("swiglu_4096x4096", 4096, 4096)]
        ew_grid = list(itertools.product((1, 2, 4, 8), (64, 128, 256, 512), (8,), (1, 2, 4, 8), (1, 2), (4,), (1,), (0,)))
        for name, M, N in swiglu_shapes:
            G = torch.randn((M, N), device="cuda", dtype=torch.float16)
            U = torch.randn((M, N), device="cuda", dtype=torch.float16)
            ref = (torch.nn.functional.silu(G.float()) * U.float()).half()
            total = M * N
            sol = _sol_ms("swiglu", M, N, 1, 2, gpu_key)
            pick_idx = rng.choice(len(ew_grid), size=min(48, len(ew_grid)), replace=False)
            ok_cnt = 0
            for idx in sorted(pick_idx):
                bm, bn, bk, nw, ns, vw, sk, sw = ew_grid[int(idx)]
                bs = bm * bn
                O = torch.empty_like(G)
                grid = (triton.cdiv(total, bs),)
                t_c0 = time.perf_counter()
                try:
                    swiglu_kernel[grid](G, U, O, total, BLOCK_SIZE=bs, num_warps=nw, num_stages=ns)
                    torch.cuda.synchronize()
                    compile_s = time.perf_counter() - t_c0
                    max_err = float((O - ref).abs().max().item())
                    if not torch.allclose(O, ref, rtol=5e-2, atol=5e-2):
                        continue
                    med_ms, q25, q75 = triton.testing.do_bench(
                        lambda: swiglu_kernel[grid](G, U, O, total, BLOCK_SIZE=bs, num_warps=nw, num_stages=ns),
                        quantiles=[0.5, 0.25, 0.75],
                    )
                    log_eff = float(math.log(max(sol / max(med_ms, 1e-6), 1e-6)))
                    records.append({
                        "hardware": gpu_key, "device_name": device_name,
                        "workload": name, "op_type": "swiglu", "shape": [M, N, 1], "dtype_bytes": 2,
                        "schedule": {"block_m": bm, "block_n": bn, "block_k": bk, "num_warps": nw, "num_stages": ns, "vec_width": vw, "split_k": sk, "swizzle": sw},
                        "verified": True, "latency_ms": float(med_ms), "iqr_ms": float(q75 - q25),
                        "compile_s": float(compile_s), "sol_latency_ms": float(sol),
                        "log_sol_eff": min(-0.001, log_eff), "max_abs_err": max_err,
                    })
                    ok_cnt += 1
                except Exception:
                    continue
            print(f"[{gpu_key}] {name:28s}: {ok_cnt} verified", flush=True)

        # ---- Benchmark FlashAttention-2 Tiled Kernel (2 shapes) ----
        fa_shapes = [
            ("flash_attn_s512_d64", 16, 512, 64),
            ("flash_attn_s1024_d64", 8, 1024, 64),
        ]
        fa_grid = list(itertools.product((16, 32, 64, 128), (16, 32, 64, 128), (16,), (2, 4, 8), (1, 2, 3), (4,), (1,), (0,)))
        for name, Z, S, D in fa_shapes:
            Q = torch.randn((Z, S, D), device="cuda", dtype=torch.float16) * 0.2
            K_t = torch.randn((Z, S, D), device="cuda", dtype=torch.float16) * 0.2
            V_t = torch.randn((Z, S, D), device="cuda", dtype=torch.float16) * 0.2
            sm_scale = 1.0 / math.sqrt(D)
            ref = torch.nn.functional.scaled_dot_product_attention(Q, K_t, V_t)
            sol = _sol_ms("flash_attn", Z * S, S, D, 2, gpu_key)
            pick_idx = rng.choice(len(fa_grid), size=min(48, len(fa_grid)), replace=False)
            ok_cnt = 0
            for idx in sorted(pick_idx):
                bm, bn, bk, nw, ns, vw, sk, sw = fa_grid[int(idx)]
                Out = torch.empty_like(Q)
                grid = (triton.cdiv(S, bm), Z)
                t_c0 = time.perf_counter()
                try:
                    flash_attn_kernel[grid](
                        Q, K_t, V_t, Out,
                        Q.stride(0), Q.stride(1), Q.stride(2),
                        K_t.stride(0), K_t.stride(1), K_t.stride(2),
                        V_t.stride(0), V_t.stride(1), V_t.stride(2),
                        Out.stride(0), Out.stride(1), Out.stride(2),
                        S, sm_scale, BLOCK_M=bm, BLOCK_N=bn, HEAD_DIM=D,
                        num_warps=nw, num_stages=ns,
                    )
                    torch.cuda.synchronize()
                    compile_s = time.perf_counter() - t_c0
                    max_err = float((Out - ref).abs().max().item())
                    if not torch.allclose(Out, ref, rtol=5e-2, atol=5e-2):
                        continue
                    med_ms, q25, q75 = triton.testing.do_bench(
                        lambda: flash_attn_kernel[grid](
                            Q, K_t, V_t, Out,
                            Q.stride(0), Q.stride(1), Q.stride(2),
                            K_t.stride(0), K_t.stride(1), K_t.stride(2),
                            V_t.stride(0), V_t.stride(1), V_t.stride(2),
                            Out.stride(0), Out.stride(1), Out.stride(2),
                            S, sm_scale, BLOCK_M=bm, BLOCK_N=bn, HEAD_DIM=D,
                            num_warps=nw, num_stages=ns,
                        ),
                        quantiles=[0.5, 0.25, 0.75],
                    )
                    log_eff = float(math.log(max(sol / max(med_ms, 1e-6), 1e-6)))
                    records.append({
                        "hardware": gpu_key, "device_name": device_name,
                        "workload": name, "op_type": "flash_attn", "shape": [Z * S, S, D], "dtype_bytes": 2,
                        "schedule": {"block_m": bm, "block_n": bn, "block_k": bk, "num_warps": nw, "num_stages": ns, "vec_width": vw, "split_k": sk, "swizzle": sw},
                        "verified": True, "latency_ms": float(med_ms), "iqr_ms": float(q75 - q25),
                        "compile_s": float(compile_s), "sol_latency_ms": float(sol),
                        "log_sol_eff": min(-0.001, log_eff), "max_abs_err": max_err,
                    })
                    ok_cnt += 1
                except Exception:
                    continue
            print(f"[{gpu_key}] {name:28s}: {ok_cnt} verified", flush=True)

        print(f"[{gpu_key}] TOTAL VERIFIED TRITON TRACES: {len(records)}", flush=True)
        return records

    @app.function(image=image, gpu="T4", timeout=1800)
    def bench_t4(max_per_op: int = 72):
        return _run_full_triton_suite("nvidia_t4", max_per_op)

    @app.function(image=image, gpu="L4", timeout=1800)
    def bench_l4(max_per_op: int = 72):
        return _run_full_triton_suite("nvidia_l4", max_per_op)

    @app.function(image=image, gpu="A100-80GB", timeout=1800)
    def bench_a100(max_per_op: int = 72):
        return _run_full_triton_suite("nvidia_a100", max_per_op)

    @app.function(image=image, gpu="H100", timeout=1800)
    def bench_h100(max_per_op: int = 72):
        return _run_full_triton_suite("nvidia_h100", max_per_op)

    @app.local_entrypoint()
    def main(gpu: str = "ALL", max_per_op: int = 72):
        dispatch = {
            "T4": ("nvidia_t4", bench_t4),
            "L4": ("nvidia_l4", bench_l4),
            "A100": ("nvidia_a100", bench_a100),
            "H100": ("nvidia_h100", bench_h100),
        }
        targets = list(dispatch.keys()) if gpu.upper() == "ALL" else [gpu.upper()]
        # Launch all target GPUs concurrently on Modal!
        calls = [(name, dispatch[name][0], dispatch[name][1].spawn(max_per_op)) for name in targets]
        data_dir = Path(__file__).resolve().parents[1] / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        for name, hw_key, handle in calls:
            rows = handle.get()
            out_path = data_dir / f"traces_{hw_key}.jsonl"
            with out_path.open("w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            print(f"Saved {len(rows)} verified Triton measurements on {name} -> {out_path}")
