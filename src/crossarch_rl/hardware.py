"""GPU hardware descriptors, datasheet specs, and schedule feasibility checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class HardwareSpec:
    """Published hardware datasheet specification for a target GPU architecture."""

    name: str
    vendor: str  # "apple" | "nvidia"
    arch: str  # e.g., "apple_m2_max", "sm75", "sm80", "sm89", "sm90"
    peak_fp32_tflops: float
    peak_fp16_tflops: float
    peak_bandwidth_gbps: float
    sram_per_cu_bytes: int  # Threadgroup memory (Metal) or Shared memory per SM (CUDA)
    num_cus: int  # GPU cores (Apple) or SMs (NVIDIA)
    simd_width: int  # 32 for both Apple simdgroup and NVIDIA warp
    max_threads_per_block: int
    registers_per_cu: int  # 32-bit registers per CU/SM
    l2_cache_bytes: int
    max_warps_per_cu: int
    has_tensor_cores: bool
    has_async_copy: bool  # cp.async / TMA pipelining

    @property
    def ridge_point_fp16(self) -> float:
        """Roofline ridge point (FLOPs / Byte) for FP16/BF16 workloads."""
        return (self.peak_fp16_tflops * 1e12) / (self.peak_bandwidth_gbps * 1e9)

    @property
    def ridge_point_fp32(self) -> float:
        """Roofline ridge point (FLOPs / Byte) for FP32 workloads."""
        return (self.peak_fp32_tflops * 1e12) / (self.peak_bandwidth_gbps * 1e9)

    def peak_tflops(self, dtype_bytes: int) -> float:
        return self.peak_fp16_tflops if dtype_bytes <= 2 else self.peak_fp32_tflops

    def ridge_point(self, dtype_bytes: int) -> float:
        return self.ridge_point_fp16 if dtype_bytes <= 2 else self.ridge_point_fp32

    def datasheet_vector(self) -> List[float]:
        """Normalized log-scale hardware descriptor vector u_h."""
        import math

        return [
            math.log2(self.peak_fp16_tflops),
            math.log2(self.peak_fp32_tflops),
            math.log2(self.peak_bandwidth_gbps),
            math.log2(self.ridge_point_fp16),
            math.log2(self.sram_per_cu_bytes / 1024.0),
            math.log2(self.num_cus),
            math.log2(self.l2_cache_bytes / (1024.0 * 1024.0)),
            math.log2(self.max_warps_per_cu),
            1.0 if self.has_tensor_cores else 0.0,
            1.0 if self.has_async_copy else 0.0,
        ]


HARDWARE_CATALOG: Dict[str, HardwareSpec] = {
    "apple_m2_max_cpu": HardwareSpec(
        name="Apple M2 Max CPU (12-Core ARM64)",
        vendor="apple_cpu",
        arch="apple_m2_max_cpu",
        peak_fp32_tflops=0.85,  # 12-core NEON FP32 throughput
        peak_fp16_tflops=1.70,
        peak_bandwidth_gbps=200.0,  # CPU-side unified memory bandwidth
        sram_per_cu_bytes=128 * 1024,  # 128 KB L1D cache per P-core
        num_cus=12,
        simd_width=4,  # 128-bit NEON FP32 lane width
        max_threads_per_block=64,
        registers_per_cu=4096,
        l2_cache_bytes=32 * 1024 * 1024,  # 32 MB L2
        max_warps_per_cu=16,
        has_tensor_cores=False,
        has_async_copy=False,
    ),
    "apple_m2_max": HardwareSpec(
        name="Apple M2 Max",
        vendor="apple",
        arch="apple_m2_max",
        peak_fp32_tflops=13.6,
        peak_fp16_tflops=27.2,
        peak_bandwidth_gbps=400.0,
        sram_per_cu_bytes=32 * 1024,  # 32 KB threadgroup memory
        num_cus=38,
        simd_width=32,
        max_threads_per_block=1024,
        registers_per_cu=65536,
        l2_cache_bytes=48 * 1024 * 1024,  # 48 MB SLC
        max_warps_per_cu=48,
        has_tensor_cores=False,  # simdgroup_matrix ALU path, lower ridge than Hopper
        has_async_copy=False,
    ),
    "nvidia_t4": HardwareSpec(
        name="NVIDIA T4",
        vendor="nvidia",
        arch="sm75",
        peak_fp32_tflops=8.1,
        peak_fp16_tflops=65.0,
        peak_bandwidth_gbps=320.0,
        sram_per_cu_bytes=64 * 1024,
        num_cus=40,
        simd_width=32,
        max_threads_per_block=1024,
        registers_per_cu=65536,
        l2_cache_bytes=4 * 1024 * 1024,
        max_warps_per_cu=32,
        has_tensor_cores=True,
        has_async_copy=False,
    ),
    "nvidia_l4": HardwareSpec(
        name="NVIDIA L4",
        vendor="nvidia",
        arch="sm89",
        peak_fp32_tflops=30.3,
        peak_fp16_tflops=121.0,
        peak_bandwidth_gbps=300.0,
        sram_per_cu_bytes=100 * 1024,
        num_cus=58,
        simd_width=32,
        max_threads_per_block=1024,
        registers_per_cu=65536,
        l2_cache_bytes=48 * 1024 * 1024,
        max_warps_per_cu=48,
        has_tensor_cores=True,
        has_async_copy=True,
    ),
    "nvidia_a10g": HardwareSpec(
        name="NVIDIA A10G",
        vendor="nvidia",
        arch="sm86",
        peak_fp32_tflops=31.2,
        peak_fp16_tflops=125.0,
        peak_bandwidth_gbps=600.0,
        sram_per_cu_bytes=100 * 1024,
        num_cus=80,
        simd_width=32,
        max_threads_per_block=1024,
        registers_per_cu=65536,
        l2_cache_bytes=6 * 1024 * 1024,
        max_warps_per_cu=48,
        has_tensor_cores=True,
        has_async_copy=True,
    ),
    "nvidia_a100": HardwareSpec(
        name="NVIDIA A100-80GB",
        vendor="nvidia",
        arch="sm80",
        peak_fp32_tflops=19.5,
        peak_fp16_tflops=312.0,
        peak_bandwidth_gbps=2039.0,
        sram_per_cu_bytes=164 * 1024,
        num_cus=108,
        simd_width=32,
        max_threads_per_block=1024,
        registers_per_cu=65536,
        l2_cache_bytes=40 * 1024 * 1024,
        max_warps_per_cu=64,
        has_tensor_cores=True,
        has_async_copy=True,
    ),
    "nvidia_h100": HardwareSpec(
        name="NVIDIA H100-SXM",
        vendor="nvidia",
        arch="sm90",
        peak_fp32_tflops=67.0,
        peak_fp16_tflops=989.0,
        peak_bandwidth_gbps=3350.0,
        sram_per_cu_bytes=228 * 1024,
        num_cus=132,
        simd_width=32,
        max_threads_per_block=1024,
        registers_per_cu=65536,
        l2_cache_bytes=50 * 1024 * 1024,
        max_warps_per_cu=64,
        has_tensor_cores=True,
        has_async_copy=True,
    ),
}


def get_hardware(key: str) -> HardwareSpec:
    if key not in HARDWARE_CATALOG:
        raise KeyError(f"Unknown hardware '{key}'. Available: {list(HARDWARE_CATALOG)}")
    return HARDWARE_CATALOG[key]
