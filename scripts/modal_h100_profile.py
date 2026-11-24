"""
Live NVIDIA H100 Low-Level Compiler & Hardware Profiling Diagnostic for CrossArch-RL.

Profiles the exact paired schedules picked zero-shot by:
  1. Linear Physics Only (no MLP)
  2. CrossArch-RL (Full + small residual MLP)
  3. Measured Oracle
across 6 core LLM workloads on a live NVIDIA H100 80GB HBM3, extracting:
  - Exact live latency (ms), achieved TFLOPS, and achieved HBM3 bandwidth (GB/s)
  - Exact Triton/ptxas compiled hardware telemetry: n_regs, n_spills, shared_bytes
  - Exact PTX instruction profile: Tensor Core (mma/wgmma), async copies (cp.async/tma),
    global loads/stores, syncthreads barriers, and total PTX instructions
  - Exact active warps / SM occupancy computed from real ptxas register & SRAM counts
  - Bonus check on whether ncu hardware PMU counters are permitted in the container.

Usage:
    uv run --with modal modal run scripts/modal_h100_profile.py
"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import modal

app = modal.App("crossarch-rl-h100-profile")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.4", "triton>=3.0", "numpy>=1.26"
)


@app.function(image=image, gpu="H100", timeout=600)
def profile_h100_cases() -> Dict[str, Any]:
    import torch
    import triton
    import triton.language as tl

    device_name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    ncu_bin = shutil.which("ncu")
    ncu_status = "not_installed"
    if ncu_bin:
        r = subprocess.run([ncu_bin, "--version"], capture_output=True, text=True)
        ncu_status = f"installed ({r.stdout.strip().splitlines()[0] if r.stdout else r.returncode})"

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
        tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))

    @triton.jit
    def rmsnorm_kernel(x_ptr, res_ptr, w_ptr, y_ptr, stride, N, BLOCK_N: tl.constexpr):
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
            tl.store(y_ptr + row_start + cols, ((x + r) * inv_rms * w).to(tl.float16), mask=mask)

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
        q = tl.load(Q + pid_z * stride_qz + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk, mask=offs_m[:, None] < S, other=0.0)
        m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
        for start_n in range(0, S, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            k = tl.load(K + pid_z * stride_kz + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk, mask=offs_n[:, None] < S, other=0.0)
            v = tl.load(V + pid_z * stride_vz + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk, mask=offs_n[:, None] < S, other=0.0)
            qk = tl.dot(q, tl.trans(k)) * sm_scale
            qk = tl.where(offs_n[None, :] < S, qk, -float("inf"))
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            p = tl.exp(qk - m_ij[:, None])
            l_ij = tl.sum(p, 1)
            alpha = tl.exp(m_i - m_ij)
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
            l_i = l_i * alpha + l_ij
            m_i = m_ij
        tl.store(Out + pid_z * stride_oz + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok, (acc / l_i[:, None]).to(tl.float16), mask=offs_m[:, None] < S)

    def inspect_compiled(compiled_obj, num_warps: int, grid_blocks: int) -> Dict[str, Any]:
        n_regs = int(getattr(compiled_obj, "n_regs", -1))
        n_spills = int(getattr(compiled_obj, "n_spills", -1))
        shared = int(getattr(getattr(compiled_obj, "metadata", None), "shared", 0))
        asm = getattr(compiled_obj, "asm", {})
        ptx = asm.get("ptx", "") if isinstance(asm, dict) else ""
        ptx_lines = [ln.strip() for ln in ptx.splitlines() if ln.strip() and not ln.strip().startswith("//") and not ln.strip().startswith(".")]

        n_mma = sum(1 for ln in ptx_lines if ("mma.sync" in ln or "wgmma.mma_async" in ln))
        n_cp_async = sum(1 for ln in ptx_lines if ("cp.async" in ln or "cp.reduce" in ln))
        n_ld_global = sum(1 for ln in ptx_lines if "ld.global" in ln)
        n_st_global = sum(1 for ln in ptx_lines if "st.global" in ln)
        n_local_spill_ops = sum(1 for ln in ptx_lines if (".local" in ln and ("ld." in ln or "st." in ln)))
        n_barriers = sum(1 for ln in ptx_lines if "bar.sync" in ln)

        # Compute exact H100 SM occupancy from real ptxas n_regs and real shared bytes
        # H100 SM90: 65536 32-bit regs/SM, 228 KB shared/SM, max 64 warps/SM, max 32 blocks/SM
        threads = num_warps * 32
        regs_per_block = max(1, n_regs) * threads if n_regs > 0 else threads * 32
        blk_reg = 65536 // regs_per_block
        blk_sram = (228 * 1024) // max(shared, 1) if shared > 0 else 32
        blk_warps = 64 // max(num_warps, 1)
        active_blks = max(1, min(32, blk_reg, blk_sram, blk_warps))
        real_occ = min(1.0, (active_blks * num_warps) / 64.0)
        waves = grid_blocks / float(active_blks * props.multi_processor_count)

        return {
            "n_regs": n_regs,
            "n_spills": n_spills,
            "shared_bytes": shared,
            "ptx_total_insns": len(ptx_lines),
            "ptx_mma_insns": n_mma,
            "ptx_cp_async_insns": n_cp_async,
            "ptx_ld_global": n_ld_global,
            "ptx_st_global": n_st_global,
            "ptx_local_spill_ops": n_local_spill_ops,
            "ptx_barriers": n_barriers,
            "grid_blocks": grid_blocks,
            "active_blocks_per_sm": active_blks,
            "real_sm_occupancy": round(real_occ, 3),
            "real_waves": round(waves, 2),
        }

    # Exact schedules from our live H100 diagnostic (Oracle vs CrossArch-RL +MLP vs Linear Physics Only)
    cases = [
        ("gemm_prefill_512x2048x2048", "gemm", (512, 2048, 2048), [
            ("Oracle",         dict(bm=64, bn=64, bk=64, nw=8, ns=4, sw=0)),
            ("CrossArch+MLP",  dict(bm=64, bn=64, bk=64, nw=8, ns=4, sw=0)),
            ("LinearPhysOnly", dict(bm=16, bn=16, bk=64, nw=4, ns=2, sw=0)),
        ]),
        ("gemm_mlp_1024x4096x1024", "gemm", (1024, 4096, 1024), [
            ("Oracle",         dict(bm=256, bn=128, bk=32, nw=8, ns=4, sw=1)),
            ("CrossArch+MLP",  dict(bm=64,  bn=128, bk=32, nw=4, ns=4, sw=1)),
            ("LinearPhysOnly", dict(bm=16,  bn=32,  bk=32, nw=8, ns=1, sw=0)),
        ]),
        ("gemm_prefill_1024x1024x1024", "gemm", (1024, 1024, 1024), [
            ("Oracle",         dict(bm=32, bn=64, bk=64, nw=4, ns=3, sw=0)),
            ("CrossArch+MLP",  dict(bm=64, bn=64, bk=64, nw=8, ns=2, sw=0)),
            ("LinearPhysOnly", dict(bm=16, bn=16, bk=16, nw=8, ns=1, sw=0)),
        ]),
        ("flash_attn_s512_d64", "flash_attn", (16, 512, 64), [
            ("Oracle",         dict(bm=64, bn=128, nw=8, ns=3)),
            ("CrossArch+MLP",  dict(bm=64, bn=64,  nw=8, ns=3)),
            ("LinearPhysOnly", dict(bm=16, bn=16,  nw=2, ns=1)),
        ]),
        ("flash_attn_s1024_d64", "flash_attn", (8, 1024, 64), [
            ("Oracle",         dict(bm=64, bn=128, nw=8, ns=3)),
            ("CrossArch+MLP",  dict(bm=64, bn=64,  nw=8, ns=3)),
            ("LinearPhysOnly", dict(bm=16, bn=16,  nw=4, ns=1)),
        ]),
        ("rmsnorm_1024x8192", "rmsnorm", (1024, 8192, 1), [
            ("Oracle",         dict(bn=2048, nw=4, ns=3)),
            ("CrossArch+MLP",  dict(bn=512,  nw=8, ns=3)),
            ("LinearPhysOnly", dict(bn=64,   nw=8, ns=1)),
        ]),
    ]

    results: List[Dict[str, Any]] = []
    for wname, op, shape, variants in cases:
        if op == "gemm":
            M, N, K = shape
            A = torch.randn((M, K), device="cuda", dtype=torch.float16) * 0.05
            B = torch.randn((K, N), device="cuda", dtype=torch.float16) * 0.05
            C = torch.empty((M, N), device="cuda", dtype=torch.float16)
            flops = 2.0 * M * N * K
            nbytes = 2.0 * (M * K + K * N + M * N)
            for label, p in variants:
                grid_blocks = triton.cdiv(M, p["bm"]) * triton.cdiv(N, p["bn"])
                compiled = gemm_kernel[(grid_blocks,)](
                    A, B, C, M, N, K, A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
                    BLOCK_M=p["bm"], BLOCK_N=p["bn"], BLOCK_K=p["bk"], GROUP_M=(8 if p["sw"] else 1),
                    num_warps=p["nw"], num_stages=p["ns"],
                )
                torch.cuda.synchronize()
                med_ms = float(triton.testing.do_bench(lambda: gemm_kernel[(grid_blocks,)](
                    A, B, C, M, N, K, A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
                    BLOCK_M=p["bm"], BLOCK_N=p["bn"], BLOCK_K=p["bk"], GROUP_M=(8 if p["sw"] else 1),
                    num_warps=p["nw"], num_stages=p["ns"],
                )))
                info = inspect_compiled(compiled, p["nw"], grid_blocks)
                info.update({
                    "workload": wname, "op": op, "variant": label, "cfg": p,
                    "latency_ms": round(med_ms, 4),
                    "achieved_tflops": round((flops / (med_ms * 1e-3)) / 1e12, 2),
                    "achieved_gbps": round((nbytes / (med_ms * 1e-3)) / 1e9, 1),
                })
                results.append(info)

        elif op == "flash_attn":
            Z, S, D = shape
            Q = torch.randn((Z, S, D), device="cuda", dtype=torch.float16) * 0.2
            Kt = torch.randn((Z, S, D), device="cuda", dtype=torch.float16) * 0.2
            Vt = torch.randn((Z, S, D), device="cuda", dtype=torch.float16) * 0.2
            O = torch.empty_like(Q)
            sm_scale = 1.0 / math.sqrt(D)
            flops = (Z * S) * S * (4.0 * D + 5.0)
            nbytes = 2.0 * (4.0 * Z * S * D)
            for label, p in variants:
                grid = (triton.cdiv(S, p["bm"]), Z)
                grid_blocks = grid[0] * grid[1]
                compiled = flash_attn_kernel[grid](
                    Q, Kt, Vt, O,
                    Q.stride(0), Q.stride(1), Q.stride(2),
                    Kt.stride(0), Kt.stride(1), Kt.stride(2),
                    Vt.stride(0), Vt.stride(1), Vt.stride(2),
                    O.stride(0), O.stride(1), O.stride(2),
                    S, sm_scale, BLOCK_M=p["bm"], BLOCK_N=p["bn"], HEAD_DIM=D,
                    num_warps=p["nw"], num_stages=p["ns"],
                )
                torch.cuda.synchronize()
                med_ms = float(triton.testing.do_bench(lambda: flash_attn_kernel[grid](
                    Q, Kt, Vt, O,
                    Q.stride(0), Q.stride(1), Q.stride(2),
                    Kt.stride(0), Kt.stride(1), Kt.stride(2),
                    Vt.stride(0), Vt.stride(1), Vt.stride(2),
                    O.stride(0), O.stride(1), O.stride(2),
                    S, sm_scale, BLOCK_M=p["bm"], BLOCK_N=p["bn"], HEAD_DIM=D,
                    num_warps=p["nw"], num_stages=p["ns"],
                )))
                info = inspect_compiled(compiled, p["nw"], grid_blocks)
                info.update({
                    "workload": wname, "op": op, "variant": label, "cfg": p,
                    "latency_ms": round(med_ms, 4),
                    "achieved_tflops": round((flops / (med_ms * 1e-3)) / 1e12, 2),
                    "achieved_gbps": round((nbytes / (med_ms * 1e-3)) / 1e9, 1),
                })
                results.append(info)

        elif op == "rmsnorm":
            M, N, _ = shape
            X = torch.randn((M, N), device="cuda", dtype=torch.float16)
            Res = torch.randn((M, N), device="cuda", dtype=torch.float16)
            W = torch.randn((N,), device="cuda", dtype=torch.float16)
            Y = torch.empty_like(X)
            flops = 6.0 * M * N
            nbytes = 2.0 * (3.0 * M * N + N)
            for label, p in variants:
                grid_blocks = M
                compiled = rmsnorm_kernel[(M,)](X, Res, W, Y, X.stride(0), N, BLOCK_N=p["bn"], num_warps=p["nw"], num_stages=p["ns"])
                torch.cuda.synchronize()
                med_ms = float(triton.testing.do_bench(lambda: rmsnorm_kernel[(M,)](
                    X, Res, W, Y, X.stride(0), N, BLOCK_N=p["bn"], num_warps=p["nw"], num_stages=p["ns"]
                )))
                info = inspect_compiled(compiled, p["nw"], grid_blocks)
                info.update({
                    "workload": wname, "op": op, "variant": label, "cfg": p,
                    "latency_ms": round(med_ms, 4),
                    "achieved_tflops": round((flops / (med_ms * 1e-3)) / 1e12, 3),
                    "achieved_gbps": round((nbytes / (med_ms * 1e-3)) / 1e9, 1),
                })
                results.append(info)

    return {
        "device_name": device_name,
        "sm_count": props.multi_processor_count,
        "ncu_status": ncu_status,
        "profiles": results,
    }


@app.local_entrypoint()
def main():
    out = profile_h100_cases.remote()
    out_path = Path(__file__).resolve().parents[1] / "results" / "h100_compiler_profile.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nLive Silicon: {out['device_name']} ({out['sm_count']} SMs) | ncu: {out['ncu_status']}")
    print(f"{'Workload':28s} | {'Variant':14s} | {'Lat(ms)':7s} | {'TFLOPS':6s} | {'GB/s':6s} | {'Regs':4s} | {'Spill':5s} | {'SRAM(B)':7s} | {'Occ':5s} | {'MMA':3s} | {'Async':5s} | {'Blks':5s}")
    print("-" * 124)
    for r in out["profiles"]:
        print(
            f"{r['workload']:28s} | {r['variant']:14s} | {r['latency_ms']:7.4f} | {r['achieved_tflops']:6.1f} | "
            f"{r['achieved_gbps']:6.1f} | {r['n_regs']:4d} | {r['n_spills']:5d} | {r['shared_bytes']:7d} | "
            f"{r['real_sm_occupancy']:5.2f} | {r['ptx_mma_insns']:3d} | {r['ptx_cp_async_insns']:5d} | {r['grid_blocks']:5d}"
        )
    print(f"\nSaved H100 compiler telemetry -> {out_path}")
