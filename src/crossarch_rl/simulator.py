"""Calibrated multi-architecture GPU physics simulator with nonlinear cliffs & hardware nuances."""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np

from crossarch_rl.hardware import HardwareSpec
from crossarch_rl.schedule_mdp import (
    ScheduleConfig,
    check_feasibility,
    estimate_resources,
    extract_physics_basis,
)
from crossarch_rl.workloads import WorkloadSpec


# True hardware-specific bottleneck weight profiles (used to construct realistic
# non-identical cross-architecture latency surfaces WITH added nonlinearities).
# Each architecture has genuinely different sensitivity to wave quantization,
# register spills, tensor-core alignment, async pipelining, and L2/SLC thrashing.
ARCH_PHYSICS_PROFILES: Dict[str, Dict[str, float]] = {
    "apple_m2_max": {
        "base_eff": -0.22,
        "wave_w": 0.72,
        "occ_w": 0.42,
        "reg_w": 0.65,
        "sram_w": 1.40,
        "mem_starve_w": 0.24,
        "tile_over_w": 0.14,
        "vec_w": 0.18,
        "pipe_w": 0.10,
        "splitk_w": 0.14,
        "swizzle_w": 0.04,  # Apple M2 bank conflicts are milder than older CUDA
        "l2_w": 0.08,       # 48MB unified SLC is forgiving
        "tc_align_pen": 0.02,
        "spill_threshold": 96.0,
        "spill_cliff": 0.55,
        "opt_sram_frac": 0.55,
    },
    "nvidia_t4": {
        "base_eff": -0.35,
        "wave_w": 0.85,
        "occ_w": 0.55,
        "reg_w": 0.90,
        "sram_w": 1.60,
        "mem_starve_w": 0.36,
        "tile_over_w": 0.18,
        "vec_w": 0.22,
        "pipe_w": 0.12,
        "splitk_w": 0.18,
        "swizzle_w": 0.14,
        "l2_w": 0.24,       # Small 4MB L2 -> high thrash penalty
        "tc_align_pen": 0.48,  # Strict Tensor Core 16x16 alignment
        "spill_threshold": 80.0,
        "spill_cliff": 0.85,
        "opt_sram_frac": 0.62,
    },
    "nvidia_l4": {
        "base_eff": -0.28,
        "wave_w": 0.80,
        "occ_w": 0.48,
        "reg_w": 0.75,
        "sram_w": 1.30,
        "mem_starve_w": 0.34,  # 300 GB/s bandwidth is tight vs 121 TFLOPS!
        "tile_over_w": 0.12,
        "vec_w": 0.24,
        "pipe_w": 0.16,
        "splitk_w": 0.15,
        "swizzle_w": 0.12,
        "l2_w": 0.07,       # Huge 48MB L2 cache on Ada!
        "tc_align_pen": 0.45,
        "spill_threshold": 96.0,
        "spill_cliff": 0.70,
        "opt_sram_frac": 0.68,
    },
    "nvidia_a10g": {
        "base_eff": -0.30,
        "wave_w": 0.88,
        "occ_w": 0.52,
        "reg_w": 0.82,
        "sram_w": 1.45,
        "mem_starve_w": 0.28,
        "tile_over_w": 0.14,
        "vec_w": 0.20,
        "pipe_w": 0.18,
        "splitk_w": 0.16,
        "swizzle_w": 0.13,
        "l2_w": 0.22,       # Small 6MB L2
        "tc_align_pen": 0.46,
        "spill_threshold": 88.0,
        "spill_cliff": 0.78,
        "opt_sram_frac": 0.65,
    },
    "nvidia_a100": {
        "base_eff": -0.24,
        "wave_w": 0.92,     # 108 SMs -> tail waves hurt heavily
        "occ_w": 0.45,
        "reg_w": 0.78,
        "sram_w": 1.25,
        "mem_starve_w": 0.26,
        "tile_over_w": 0.10,
        "vec_w": 0.22,
        "pipe_w": 0.24,     # cp.async multi-stage pipelining shines on Ampere
        "splitk_w": 0.12,
        "swizzle_w": 0.15,
        "l2_w": 0.10,       # 40MB L2
        "tc_align_pen": 0.52,
        "spill_threshold": 104.0,
        "spill_cliff": 0.80,
        "opt_sram_frac": 0.72,
    },
    "nvidia_h100": {
        "base_eff": -0.20,
        "wave_w": 0.96,     # 132 SMs -> wave quantization critical
        "occ_w": 0.40,
        "reg_w": 0.72,
        "sram_w": 1.15,
        "mem_starve_w": 0.30,  # High ridge point (989 TF / 3.35 TB/s)
        "tile_over_w": 0.08,
        "vec_w": 0.20,
        "pipe_w": 0.30,     # Hopper TMA / deep pipelining dominates
        "splitk_w": 0.11,
        "swizzle_w": 0.18,  # WGMMA swizzle patterns matter strongly
        "l2_w": 0.09,
        "tc_align_pen": 0.60,
        "spill_threshold": 112.0,
        "spill_cliff": 0.85,
        "opt_sram_frac": 0.76,
    },
}


class CalibratedHardwareSimulator:
    """
    Calibrated nonlinear multi-architecture GPU performance model.

    Produces normalized log-SOL efficiency R_h(x, c) = log(tau_SOL / tau_h) <= 0
    and wall-clock latency tau_h(x, c), combining:
      1. Architecture-specific bottleneck weights over dimensionless physics basis
      2. Genuine nonlinear effects (hard register-spill cliff, Tensor Core alignment
         step-drop, non-monotonic SRAM/tile interaction, and op-specific coupling)
      3. Optional realistic sub-Gaussian hardware timing jitter
    """

    def __init__(self, noise_std: float = 0.015, seed: int = 42) -> None:
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

    def _profile_for(self, hw: HardwareSpec) -> Dict[str, float]:
        for key, prof in ARCH_PHYSICS_PROFILES.items():
            if hw.arch in key or key.endswith(hw.arch) or hw.name.lower().replace("-", "_").replace(" ", "_").startswith(key):
                return prof
        # Fallback match by arch code
        arch_map = {
            "apple_m2_max": "apple_m2_max",
            "sm75": "nvidia_t4",
            "sm89": "nvidia_l4",
            "sm86": "nvidia_a10g",
            "sm80": "nvidia_a100",
            "sm90": "nvidia_h100",
        }
        return ARCH_PHYSICS_PROFILES[arch_map[hw.arch]]

    def true_log_sol_efficiency(
        self, workload: WorkloadSpec, cfg: ScheduleConfig, hw: HardwareSpec
    ) -> float:
        """Deterministic ground-truth log-SOL efficiency R_h(x, c) <= 0."""
        ok, res = check_feasibility(workload, cfg, hw)
        if not ok:
            return -6.0

        prof = self._profile_for(hw)
        phi = extract_physics_basis(workload, cfg, hw)
        w = np.array(
            [
                prof["base_eff"],
                prof["wave_w"],
                prof["occ_w"],
                prof["reg_w"],
                prof["sram_w"],
                prof["mem_starve_w"],
                prof["tile_over_w"],
                prof["vec_w"],
                prof["pipe_w"],
                prof["splitk_w"],
                prof["swizzle_w"],
                prof["l2_w"],
            ],
            dtype=np.float64,
        )
        linear_part = float(np.dot(phi, w))

        # --- Genuine Nonlinear Hardware Effects (non-zero eps_fact!) ---
        nonlinear = 0.0

        # 1. Hard register-spill cliff above architecture threshold (sharper & architecture-specific)
        regs = res["regs_per_thread"]
        if regs > prof["spill_threshold"]:
            overshoot = (regs - prof["spill_threshold"]) / 20.0
            nonlinear -= prof["spill_cliff"] * (1.0 - math.exp(-overshoot))

        # 2. Tensor Core tile alignment & pipelining regime shift across GPU generations
        if hw.has_tensor_cores and workload.op_type in ("gemm", "flash_attn"):
            aligned = (cfg.block_m >= 32 and cfg.block_n >= 32 and cfg.block_m % 16 == 0 and cfg.block_n % 16 == 0)
            if not aligned:
                nonlinear -= prof["tc_align_pen"]
            if workload.op_type == "gemm" and cfg.block_k < 32 and hw.arch in ("sm80", "sm90"):
                nonlinear -= 0.32  # Ampere/Hopper MMA prefers BK >= 32
        elif hw.vendor == "apple" and workload.op_type == "gemm":
            # Apple M2 Max favors smaller compact 32x32 / 64x32 tiles without deep staging
            if cfg.num_stages >= 3:
                nonlinear -= 0.42
            if cfg.block_m * cfg.block_n > 64 * 64:
                nonlinear -= 0.35

        # 3. Hopper/Ampere deep async pipeline + swizzle synergy vs Turing/Apple penalty
        if hw.arch == "sm90":
            if cfg.num_stages >= 3 and cfg.swizzle == 1 and cfg.block_m >= 128:
                nonlinear += 0.28
            elif cfg.num_stages == 1:
                nonlinear -= 0.38
        elif hw.arch == "sm80":
            if cfg.num_stages >= 3 and cfg.swizzle == 1:
                nonlinear += 0.18
            elif cfg.num_stages == 1:
                nonlinear -= 0.22
        elif hw.arch == "sm75":
            # Turing lacks cp.async: deep staging wastes precious 64KB SRAM
            if cfg.num_stages >= 3:
                nonlinear -= 0.35
        elif hw.arch == "sm89":
            # L4 is severely memory-bandwidth constrained (300 GB/s vs 121 TFLOPS) -> split-K / high reuse wins
            if workload.op_type == "gemm" and cfg.block_m * cfg.block_n >= 64 * 128 and cfg.num_stages >= 2:
                nonlinear += 0.20

        # 4. Non-monotonic sweet-spot interaction between SRAM fill and occupancy
        sram_dev = res["sram_frac"] - prof["opt_sram_frac"]
        if res["sram_bytes"] > 0:
            nonlinear -= 0.32 * (sram_dev ** 2)

        # 5. Interaction between num_warps and tile aspect ratio for GEMM/FlashAttn
        if workload.op_type in ("gemm", "flash_attn"):
            aspect = abs(math.log2(max(cfg.block_m, 1) / max(cfg.block_n, 1)))
            nonlinear -= 0.055 * aspect
            if cfg.block_m * cfg.block_n >= 128 * 128 and cfg.num_warps < 4:
                nonlinear -= 0.38
        else:
            if cfg.vec_width == 1:
                nonlinear -= 0.22
            if workload.op_type in ("rmsnorm", "softmax") and cfg.block_n < 128:
                nonlinear -= 0.24

        # Clamp to physically valid range (<= -0.03 since no real kernel exceeds 97% SOL)
        return float(min(-0.03, max(-5.5, linear_part + nonlinear)))

    def evaluate(
        self,
        workload: WorkloadSpec,
        cfg: ScheduleConfig,
        hw: HardwareSpec,
        noisy: bool = True,
    ) -> float:
        """Measured log-SOL efficiency R_h(x, c) with optional timing jitter."""
        val = self.true_log_sol_efficiency(workload, cfg, hw)
        if noisy and self.noise_std > 0.0:
            val += float(self.rng.normal(0.0, self.noise_std))
        return float(min(-0.01, max(-6.0, val)))

    def latency_ms(
        self,
        workload: WorkloadSpec,
        cfg: ScheduleConfig,
        hw: HardwareSpec,
        noisy: bool = False,
    ) -> float:
        """Simulated execution latency in milliseconds."""
        r = self.evaluate(workload, cfg, hw, noisy=noisy)
        sol_ms = workload.sol_latency_s(hw) * 1e3
        return float(sol_ms * math.exp(-r))
