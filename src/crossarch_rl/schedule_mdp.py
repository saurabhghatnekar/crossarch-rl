"""Contextual Kernel Schedule MDP, feasibility checker, and physics-normalized features."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from crossarch_rl.hardware import HardwareSpec
from crossarch_rl.workloads import WorkloadSpec


# Discrete schedule axes shared across Metal, CPU, and Triton backends
BLOCK_M_CHOICES = (8, 16, 32, 64, 128, 256)
BLOCK_N_CHOICES = (8, 16, 32, 64, 128, 256)
BLOCK_K_CHOICES = (8, 16, 32, 64)
NUM_WARPS_CHOICES = (1, 2, 4, 8, 16)
NUM_STAGES_CHOICES = (1, 2, 3, 4)
VEC_WIDTH_CHOICES = (1, 2, 4)
SPLIT_K_CHOICES = (1, 2, 4)
SWIZZLE_CHOICES = (0, 1)


@dataclass(frozen=True)
class ScheduleConfig:
    """Unified parameterized GPU kernel schedule configuration."""

    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int
    vec_width: int
    split_k: int = 1
    swizzle: int = 0

    def as_tuple(self) -> Tuple[int, ...]:
        return (
            self.block_m,
            self.block_n,
            self.block_k,
            self.num_warps,
            self.num_stages,
            self.vec_width,
            self.split_k,
            self.swizzle,
        )

    def as_dict(self) -> Dict[str, int]:
        return {
            "block_m": self.block_m,
            "block_n": self.block_n,
            "block_k": self.block_k,
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
            "vec_width": self.vec_width,
            "split_k": self.split_k,
            "swizzle": self.swizzle,
        }


def estimate_resources(
    workload: WorkloadSpec, cfg: ScheduleConfig, hw: HardwareSpec
) -> Dict[str, float]:
    """Analytically estimate hardware resource footprint of a schedule on target hardware."""
    b = float(workload.dtype_bytes)
    threads_per_block = cfg.num_warps * hw.simd_width
    stages = max(cfg.num_stages, 1)

    if workload.op_type == "gemm":
        # Double/multi-buffered A (BM x BK) and B (BK x BN) tiles in SRAM/shared memory
        sram_bytes = stages * b * (cfg.block_m * cfg.block_k + cfg.block_k * cfg.block_n)
        if cfg.swizzle:
            # Padding/swizzle overhead ~ 6%
            sram_bytes *= 1.06
        grid_m = math.ceil(workload.m / cfg.block_m)
        grid_n = math.ceil(workload.n / cfg.block_n)
        num_blocks = grid_m * grid_n * cfg.split_k
        # Accumulator registers per thread (FP32 accumulators)
        accum_per_thread = (cfg.block_m * cfg.block_n) / max(threads_per_block, 1)
        regs_per_thread = 24.0 + 1.15 * accum_per_thread + 4.0 * stages
        tile_flops = 2.0 * cfg.block_m * cfg.block_n * cfg.block_k
        tile_bytes = b * (cfg.block_m * cfg.block_k + cfg.block_k * cfg.block_n)
        reuse_factor = (cfg.block_m * cfg.block_n) / max(cfg.block_m + cfg.block_n, 1)

    elif workload.op_type == "flash_attn":
        # Q tile (BM x D), K/V tiles (BN x D) buffered by stages, plus score scratch (BM x BN)
        d = workload.k
        sram_bytes = b * (cfg.block_m * d + stages * 2.0 * cfg.block_n * d) + 4.0 * cfg.block_m * cfg.block_n
        grid_m = math.ceil(workload.m / cfg.block_m)
        num_blocks = grid_m
        accum_per_thread = (cfg.block_m * d + cfg.block_m * cfg.block_n * 0.5) / max(threads_per_block, 1)
        regs_per_thread = 32.0 + 1.1 * accum_per_thread + 4.0 * stages
        tile_flops = 4.0 * cfg.block_m * cfg.block_n * d
        tile_bytes = b * (cfg.block_m * d + 2.0 * cfg.block_n * d)
        reuse_factor = (cfg.block_m * cfg.block_n) / max(cfg.block_m + cfg.block_n, 1)

    elif workload.op_type in ("rmsnorm", "softmax"):
        # Row-wise reduction: each block processes block_m rows, caching chunk of width block_n
        sram_bytes = stages * b * (cfg.block_m * cfg.block_n) + 4.0 * cfg.num_warps * cfg.block_m
        grid_m = math.ceil(workload.m / cfg.block_m)
        num_blocks = grid_m
        elems_per_thread = (cfg.block_m * cfg.block_n) / max(threads_per_block, 1)
        regs_per_thread = 20.0 + 1.5 * elems_per_thread + 2.0 * cfg.vec_width
        tile_flops = 5.5 * cfg.block_m * cfg.block_n
        tile_bytes = 2.5 * b * cfg.block_m * cfg.block_n
        reuse_factor = float(cfg.vec_width)

    elif workload.op_type == "swiglu":
        # Pure elementwise fused kernel: block processes block_m * block_n elements
        sram_bytes = (stages - 1) * b * cfg.block_m * cfg.block_n if stages > 1 else 0.0
        total_elems = workload.m * workload.n
        tile_elems = max(cfg.block_m * cfg.block_n * cfg.vec_width, threads_per_block)
        num_blocks = math.ceil(total_elems / tile_elems)
        regs_per_thread = 16.0 + 3.0 * cfg.vec_width + 2.0 * stages
        tile_flops = 6.0 * tile_elems
        tile_bytes = 3.0 * b * tile_elems
        reuse_factor = float(cfg.vec_width)

    else:
        raise ValueError(f"Unknown op_type {workload.op_type}")

    sram_frac = sram_bytes / float(hw.sram_per_cu_bytes)
    regs_per_block = regs_per_thread * threads_per_block
    reg_frac = regs_per_block / float(hw.registers_per_cu)

    # Active blocks per CU constrained by SRAM, registers, and max warps
    blocks_by_sram = math.floor(hw.sram_per_cu_bytes / max(sram_bytes, 1.0)) if sram_bytes > 0 else 16
    blocks_by_reg = math.floor(hw.registers_per_cu / max(regs_per_block, 1.0))
    blocks_by_warps = math.floor(hw.max_warps_per_cu / max(cfg.num_warps, 1))
    active_blocks_per_cu = max(0, min(16, blocks_by_sram, blocks_by_reg, blocks_by_warps))
    occupancy = min(1.0, (active_blocks_per_cu * cfg.num_warps) / float(hw.max_warps_per_cu))

    # Wave quantization efficiency across CUs
    blocks_per_wave = max(1, active_blocks_per_cu * hw.num_cus)
    waves = num_blocks / float(blocks_per_wave)
    ceil_waves = max(1, math.ceil(waves))
    wave_efficiency = waves / float(ceil_waves)

    # Tile arithmetic intensity relative to hardware roofline ridge point
    tile_ai = tile_flops / max(tile_bytes, 1.0)
    ridge = hw.ridge_point(workload.dtype_bytes)

    # Working set relative to L2/SLC cache
    ws_bytes = min(workload.bytes_moved, float(num_blocks) * tile_bytes * 0.25 + sram_bytes * hw.num_cus)
    l2_frac = ws_bytes / float(hw.l2_cache_bytes)

    return {
        "threads_per_block": float(threads_per_block),
        "sram_bytes": float(sram_bytes),
        "sram_frac": float(sram_frac),
        "regs_per_thread": float(regs_per_thread),
        "reg_frac": float(reg_frac),
        "active_blocks_per_cu": float(active_blocks_per_cu),
        "occupancy": float(occupancy),
        "num_blocks": float(num_blocks),
        "waves": float(waves),
        "wave_efficiency": float(wave_efficiency),
        "tile_ai": float(tile_ai),
        "ridge": float(ridge),
        "ai_ratio": float(tile_ai / max(ridge, 1e-6)),
        "reuse_factor": float(reuse_factor),
        "l2_frac": float(l2_frac),
    }


def check_feasibility(
    workload: WorkloadSpec, cfg: ScheduleConfig, hw: HardwareSpec
) -> Tuple[bool, Dict[str, float]]:
    """Analytically verify whether schedule cfg is hardware-feasible on hw."""
    res = estimate_resources(workload, cfg, hw)
    threads = res["threads_per_block"]
    if threads > hw.max_threads_per_block or threads < hw.simd_width:
        return False, res
    if res["sram_bytes"] > hw.sram_per_cu_bytes:
        return False, res
    # Hard register file upper limit (severe spill / compile failure above 255 regs/thread)
    if res["regs_per_thread"] > 255.0:
        return False, res
    # Operator-specific structural constraints
    if workload.op_type == "gemm":
        if cfg.block_m > workload.m or cfg.block_n > workload.n or cfg.block_k > workload.k:
            return False, res
        if cfg.split_k > 1 and (workload.k // cfg.split_k) < cfg.block_k:
            return False, res
    elif workload.op_type == "flash_attn":
        if cfg.block_m > workload.m or cfg.block_n > workload.n or cfg.split_k > 1:
            return False, res
    else:
        if cfg.split_k > 1:
            return False, res
        if cfg.block_m > workload.m or cfg.block_n > workload.n:
            return False, res
    return True, res


# Number of fixed analytic bottleneck features in phi_phys
PHYS_BASIS_DIM = 12


def extract_physics_basis(
    workload: WorkloadSpec, cfg: ScheduleConfig, hw: HardwareSpec
) -> np.ndarray:
    """
    Analytic hardware-relative bottleneck feature vector phi_phys(s, a; h) in R^12.

    Every coordinate is dimensionless and oriented so that negative values / penalties
    correspond linearly (in log-SOL space) to real hardware slowdowns:
      0: constant intercept (baseline kernel efficiency)
      1: log(wave_efficiency)  — tail-wave quantization slowdown
      2: log(max(occupancy, 0.05)) — low occupancy / latency-hiding deficit
      3: -max(0, regs_per_thread - 64)/64 — register pressure / spill cliff onset
      4: -max(0, sram_frac - 0.75)^2 — SRAM capacity congestion
      5: -max(0, log2(ridge / tile_ai)) — memory-bandwidth starvation below ridge
      6: -max(0, log2(tile_ai / ridge)) — excess compute tile overhead above ridge
      7: log2(vec_width) / 2.0 — vectorized memory coalescing bonus
      8: pipeline_hiding_gain — benefit of multi-stage pipelining (async copy aware)
      9: -log2(split_k) — split-K atomic reduction overhead
     10: swizzle_bank_benefit — shared-memory bank conflict reduction (if SRAM > 0)
     11: -max(0, log2(l2_frac + 1e-3)) — L2/SLC cache thrashing pressure
    """
    res = estimate_resources(workload, cfg, hw)
    wave_eff = max(res["wave_efficiency"], 0.05)
    occ = max(res["occupancy"], 0.04)
    reg_excess = max(0.0, (res["regs_per_thread"] - 64.0) / 64.0)
    sram_cong = max(0.0, res["sram_frac"] - 0.75) ** 2
    log_ai_ratio = math.log2(max(res["ai_ratio"], 1e-4))
    mem_starve = max(0.0, -log_ai_ratio)
    tile_overshoot = max(0.0, log_ai_ratio)
    vec_bonus = math.log2(max(cfg.vec_width, 1)) / 2.0

    # Multi-stage pipelining helps when async copy exists and stages >= 2, else minor overhead
    if hw.has_async_copy:
        pipe_gain = math.log2(cfg.num_stages) * (1.0 if res["sram_bytes"] > 0 else 0.0)
    else:
        pipe_gain = -0.35 * max(0, cfg.num_stages - 2)

    split_k_pen = -math.log2(max(cfg.split_k, 1))
    swizzle_gain = float(cfg.swizzle) * (1.0 if res["sram_bytes"] > 0 else 0.0)
    l2_pressure = max(0.0, math.log2(max(res["l2_frac"], 1e-3)))

    return np.array(
        [
            1.0,
            math.log(wave_eff),
            math.log(occ),
            -reg_excess,
            -sram_cong,
            -mem_starve,
            -tile_overshoot,
            vec_bonus,
            pipe_gain,
            split_k_pen,
            swizzle_gain,
            -l2_pressure,
        ],
        dtype=np.float32,
    )


def extract_normalized_state_features(
    workload: WorkloadSpec, cfg: ScheduleConfig, hw: HardwareSpec
) -> np.ndarray:
    """Dimensionless hardware-relative state representation z_norm(x, c, h)."""
    res = estimate_resources(workload, cfg, hw)
    phi = extract_physics_basis(workload, cfg, hw)
    extra = np.array(
        [
            res["sram_frac"],
            res["reg_frac"],
            res["occupancy"],
            res["wave_efficiency"],
            math.log2(max(res["waves"], 1e-2)) / 8.0,
            math.log2(max(res["reuse_factor"], 1.0)) / 7.0,
            math.log2(max(workload.arithmetic_intensity / max(hw.ridge_point(workload.dtype_bytes), 1e-3), 1e-3)) / 6.0,
            1.0 if (cfg.block_m % 16 == 0 and cfg.block_n % 16 == 0 and cfg.block_k % 16 == 0) else 0.0,
            1.0 if hw.has_tensor_cores else 0.0,
            1.0 if hw.has_async_copy else 0.0,
            *workload.op_one_hot(),
        ],
        dtype=np.float32,
    )
    return np.concatenate([phi, extra], axis=0)


def extract_raw_state_features(
    workload: WorkloadSpec, cfg: ScheduleConfig, hw: HardwareSpec
) -> np.ndarray:
    """Raw unnormalized tile parameters (used for 'w/o Dimensionless Normalization' ablation)."""
    return np.array(
        [
            math.log2(cfg.block_m) / 8.0,
            math.log2(cfg.block_n) / 8.0,
            math.log2(cfg.block_k) / 6.0,
            math.log2(cfg.num_warps) / 4.0,
            float(cfg.num_stages) / 4.0,
            math.log2(cfg.vec_width) / 2.0,
            math.log2(cfg.split_k) / 2.0,
            float(cfg.swizzle),
            math.log2(max(workload.m, 1)) / 14.0,
            math.log2(max(workload.n, 1)) / 14.0,
            math.log2(max(workload.k, 1)) / 14.0,
            *workload.op_one_hot(),
        ],
        dtype=np.float32,
    )


def enumerate_feasible_schedules(
    workload: WorkloadSpec,
    hw: HardwareSpec,
    max_configs: Optional[int] = None,
    seed: int = 0,
) -> List[ScheduleConfig]:
    """Enumerate valid hardware-feasible schedule space C_h(x) for a workload."""
    if workload.op_type == "gemm":
        bms = (16, 32, 64, 128, 256)
        bns = (16, 32, 64, 128, 256)
        bks = (16, 32, 64)
        warps = (2, 4, 8)
        stages = (1, 2, 3, 4)
        vecs = (1, 2, 4)
        split_ks = (1, 2)
        swizzles = (0, 1)
    elif workload.op_type == "flash_attn":
        bms = (16, 32, 64, 128)
        bns = (16, 32, 64, 128)
        bks = (16,)  # unused for flash_attn; head_dim comes from workload.k
        warps = (2, 4, 8)
        stages = (1, 2, 3)
        vecs = (2, 4)
        split_ks = (1,)
        swizzles = (0, 1)
    elif workload.op_type in ("rmsnorm", "softmax"):
        bms = (1, 2, 4, 8, 16)
        bns = (64, 128, 256, 512, 1024)
        bks = (8,)
        warps = (1, 2, 4, 8)
        stages = (1, 2)
        vecs = (1, 2, 4)
        split_ks = (1,)
        swizzles = (0,)
    elif workload.op_type == "swiglu":
        bms = (1, 2, 4, 8, 16)
        bns = (32, 64, 128, 256)
        bks = (8,)
        warps = (1, 2, 4, 8)
        stages = (1, 2)
        vecs = (1, 2, 4)
        split_ks = (1,)
        swizzles = (0,)
    else:
        raise ValueError(f"Unknown op_type {workload.op_type}")

    feasible: List[ScheduleConfig] = []
    for bm, bn, bk, nw, ns, vw, sk, sw in itertools.product(
        bms, bns, bks, warps, stages, vecs, split_ks, swizzles
    ):
        cfg = ScheduleConfig(bm, bn, bk, nw, ns, vw, sk, sw)
        ok, _ = check_feasibility(workload, cfg, hw)
        if ok:
            feasible.append(cfg)

    if max_configs is not None and len(feasible) > max_configs:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(feasible), size=max_configs, replace=False)
        feasible = [feasible[int(i)] for i in sorted(idx)]
    return feasible


def get_axis_choices(workload: WorkloadSpec) -> Tuple[Tuple[int, ...], ...]:
    if workload.op_type == "gemm":
        return ((16, 32, 64, 128, 256), (16, 32, 64, 128, 256), (16, 32, 64), (2, 4, 8), (1, 2, 3, 4), (1, 2, 4), (1, 2), (0, 1))
    if workload.op_type == "flash_attn":
        return ((16, 32, 64, 128), (16, 32, 64, 128), (16,), (2, 4, 8), (1, 2, 3), (2, 4), (1,), (0, 1))
    if workload.op_type in ("rmsnorm", "softmax"):
        return ((1, 2, 4, 8, 16), (64, 128, 256, 512, 1024), (8,), (1, 2, 4, 8), (1, 2), (1, 2, 4), (1,), (0,))
    return ((1, 2, 4, 8, 16), (32, 64, 128, 256), (8,), (1, 2, 4, 8), (1, 2), (1, 2, 4), (1,), (0,))


class ScheduleMDP:
    """
    Finite-horizon Contextual MDP for sequential GPU kernel schedule transformation.

    Actions (17 total):
      0..15 : increment (+1) or decrement (-1) along each of the 8 schedule axes
      16    : STOP (terminate and deploy current schedule)
    """

    NUM_AXES = 8
    STOP_ACTION = 16
    NUM_ACTIONS = 17

    def __init__(
        self,
        workload: WorkloadSpec,
        hw: HardwareSpec,
        eval_fn: Callable[[WorkloadSpec, ScheduleConfig, HardwareSpec], float],
        horizon: int = 12,
    ) -> None:
        self.workload = workload
        self.hw = hw
        self.eval_fn = eval_fn  # returns normalized log-SOL reward R_h(x, c) <= 0
        self.horizon = horizon
        self.axis_choices = get_axis_choices(workload)
        self.current_cfg: Optional[ScheduleConfig] = None
        self.current_score: float = -10.0
        self.step_count: int = 0

    def default_initial_config(self) -> ScheduleConfig:
        """Conservative feasible starting schedule."""
        feasible = enumerate_feasible_schedules(self.workload, self.hw)
        if not feasible:
            raise RuntimeError(f"No feasible schedules for {self.workload.name} on {self.hw.name}")
        return feasible[0]

    def reset(self, init_cfg: Optional[ScheduleConfig] = None) -> ScheduleConfig:
        self.current_cfg = init_cfg if init_cfg is not None else self.default_initial_config()
        self.current_score = self.eval_fn(self.workload, self.current_cfg, self.hw)
        self.step_count = 0
        return self.current_cfg

    def apply_action(self, cfg: ScheduleConfig, action: int) -> Optional[ScheduleConfig]:
        if action == self.STOP_ACTION:
            return cfg
        axis = action // 2
        direction = 1 if (action % 2 == 0) else -1
        choices = self.axis_choices[axis]
        vals = list(cfg.as_tuple())
        if vals[axis] not in choices:
            return None
        idx = choices.index(vals[axis]) + direction
        if idx < 0 or idx >= len(choices):
            return None
        vals[axis] = choices[idx]
        nxt = ScheduleConfig(*vals)
        ok, _ = check_feasibility(self.workload, nxt, self.hw)
        return nxt if ok else None

    def valid_actions(self, cfg: ScheduleConfig) -> List[int]:
        valid = [self.STOP_ACTION]
        for a in range(self.STOP_ACTION):
            if self.apply_action(cfg, a) is not None:
                valid.append(a)
        return valid

    def step(self, action: int) -> Tuple[ScheduleConfig, float, bool, Dict[str, float]]:
        assert self.current_cfg is not None, "Call reset() before step()."
        self.step_count += 1
        if action == self.STOP_ACTION:
            return self.current_cfg, 0.0, True, {"score": self.current_score}
        nxt = self.apply_action(self.current_cfg, action)
        if nxt is None:
            # Invalid transition penalty
            return self.current_cfg, -0.5, self.step_count >= self.horizon, {"score": self.current_score, "invalid": 1.0}
        nxt_score = self.eval_fn(self.workload, nxt, self.hw)
        reward = nxt_score - self.current_score
        self.current_cfg = nxt
        self.current_score = nxt_score
        done = self.step_count >= self.horizon
        return self.current_cfg, float(reward), done, {"score": self.current_score}
