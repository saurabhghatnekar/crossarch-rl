"""Core LLM kernel workloads and Speed-of-Light (SOL) roofline bounds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from crossarch_rl.hardware import HardwareSpec


OP_TYPES = ("gemm", "rmsnorm", "softmax", "swiglu", "flash_attn")


@dataclass(frozen=True)
class WorkloadSpec:
    """Specification of a single tensor operator workload."""

    name: str
    op_type: str  # one of OP_TYPES
    m: int  # batch/rows (or B*H*S for attention)
    n: int  # cols / hidden / vocab (or S for attention)
    k: int  # reduction dim (or head_dim for attention)
    dtype_bytes: int = 4  # 2 for FP16/BF16, 4 for FP32

    def __post_init__(self) -> None:
        if self.op_type not in OP_TYPES:
            raise ValueError(f"Unsupported op_type '{self.op_type}'. Expected one of {OP_TYPES}")

    @property
    def flops(self) -> float:
        if self.op_type == "gemm":
            return 2.0 * self.m * self.n * self.k
        if self.op_type == "rmsnorm":
            # square, sum, rsqrt, scale, residual add ~ 6 ops per element
            return 6.0 * self.m * self.n
        if self.op_type == "softmax":
            # max, sub, exp, sum, div ~ 5 ops per element
            return 5.0 * self.m * self.n
        if self.op_type == "swiglu":
            # elementwise fused silu(gate) * up ~ 6 ops per element
            return 6.0 * self.m * self.n
        if self.op_type == "flash_attn":
            # QK^T (2*S*S*D) + softmax (~5*S*S) + PV (2*S*S*D) per batch*head
            # m = B*H*S, n = S_kv, k = head_dim
            return self.m * self.n * (4.0 * self.k + 5.0)
        raise AssertionError("Unreachable")

    @property
    def bytes_moved(self) -> float:
        b = float(self.dtype_bytes)
        if self.op_type == "gemm":
            return b * (self.m * self.k + self.k * self.n + self.m * self.n)
        if self.op_type == "rmsnorm":
            # read x, residual, weight(n); write y, out_res
            return b * (3.0 * self.m * self.n + self.n)
        if self.op_type == "softmax":
            # read x, write y
            return b * (2.0 * self.m * self.n)
        if self.op_type == "swiglu":
            # read gate, up; write out
            return b * (3.0 * self.m * self.n)
        if self.op_type == "flash_attn":
            # read Q (m*k), K (m_kv*k), V (m_kv*k), write O (m*k)
            # approximate m_kv ~ m (self-attention)
            return b * (4.0 * self.m * self.k)
        raise AssertionError("Unreachable")

    @property
    def arithmetic_intensity(self) -> float:
        return self.flops / max(self.bytes_moved, 1.0)

    def sol_latency_s(self, hw: HardwareSpec) -> float:
        """Analytic Speed-of-Light lower bound latency in seconds."""
        compute_s = self.flops / (hw.peak_tflops(self.dtype_bytes) * 1e12)
        mem_s = self.bytes_moved / (hw.peak_bandwidth_gbps * 1e9)
        return max(compute_s, mem_s, 1e-7)

    def op_one_hot(self) -> List[float]:
        return [1.0 if self.op_type == op else 0.0 for op in OP_TYPES]


def default_llm_workloads(dtype_bytes: int = 4) -> List[WorkloadSpec]:
    """Curated suite of LLM workloads spanning compute-bound and memory-bound regimes."""
    return [
        # GEMM: decode (memory-bound), medium prefill, large prefill (compute-bound)
        WorkloadSpec("gemm_decode_64x4096x4096", "gemm", 64, 4096, 4096, dtype_bytes),
        WorkloadSpec("gemm_prefill_512x2048x2048", "gemm", 512, 2048, 2048, dtype_bytes),
        WorkloadSpec("gemm_prefill_1024x1024x1024", "gemm", 1024, 1024, 1024, dtype_bytes),
        WorkloadSpec("gemm_mlp_1024x4096x1024", "gemm", 1024, 4096, 1024, dtype_bytes),
        WorkloadSpec("gemm_tall_2048x1024x512", "gemm", 2048, 1024, 512, dtype_bytes),
        # Fused RMSNorm + Residual
        WorkloadSpec("rmsnorm_2048x4096", "rmsnorm", 2048, 4096, 1, dtype_bytes),
        WorkloadSpec("rmsnorm_4096x2048", "rmsnorm", 4096, 2048, 1, dtype_bytes),
        WorkloadSpec("rmsnorm_1024x8192", "rmsnorm", 1024, 8192, 1, dtype_bytes),
        # Fused Online Softmax
        WorkloadSpec("softmax_2048x4096", "softmax", 2048, 4096, 1, dtype_bytes),
        WorkloadSpec("softmax_1024x8192", "softmax", 1024, 8192, 1, dtype_bytes),
        WorkloadSpec("softmax_512x16384", "softmax", 512, 16384, 1, dtype_bytes),
        # Fused SwiGLU
        WorkloadSpec("swiglu_2048x4096", "swiglu", 2048, 4096, 1, dtype_bytes),
        WorkloadSpec("swiglu_4096x4096", "swiglu", 4096, 4096, 1, dtype_bytes),
        # FlashAttention-style Tiled Attention (m = B*H*S, n = S_kv, k = head_dim)
        WorkloadSpec("flash_attn_s512_d64", "flash_attn", 16 * 512, 512, 64, dtype_bytes),
        WorkloadSpec("flash_attn_s1024_d64", "flash_attn", 8 * 1024, 1024, 64, dtype_bytes),
        WorkloadSpec("flash_attn_s512_d128", "flash_attn", 8 * 512, 512, 128, dtype_bytes),
    ]
