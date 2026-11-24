"""Real native compiled ARM64 CPU (Clang -O3 + pthreads) tiled kernel benchmarker."""

from __future__ import annotations

import ctypes
import math
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from crossarch_rl.hardware import get_hardware
from crossarch_rl.metal_bench import KernelMeasurement
from crossarch_rl.schedule_mdp import ScheduleConfig
from crossarch_rl.workloads import WorkloadSpec


_C_SOURCE = r"""
#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int tid;
    int n_threads;
    int M, N, K;
    int BM, BN, BK;
    int vec;
    int swizzle;
    const float* A;
    const float* B;
    const float* W;
    float* C;
} TaskArgs;

static void* gemm_worker(void* arg) {
    TaskArgs* t = (TaskArgs*)arg;
    const int M = t->M, N = t->N, K = t->K;
    const int BM = t->BM, BN = t->BN, BK = t->BK;
    const float* A = t->A;
    const float* B = t->B;
    float* C = t->C;

    int num_mb = (M + BM - 1) / BM;
    for (int mb = t->tid; mb < num_mb; mb += t->n_threads) {
        int i0 = mb * BM;
        int i1 = (i0 + BM < M) ? (i0 + BM) : M;
        for (int i = i0; i < i1; ++i) {
            memset(C + i * N, 0, sizeof(float) * N);
        }
        for (int k0 = 0; k0 < K; k0 += BK) {
            int k1 = (k0 + BK < K) ? (k0 + BK) : K;
            for (int j0 = 0; j0 < N; j0 += BN) {
                int j1 = (j0 + BN < N) ? (j0 + BN) : N;
                for (int i = i0; i < i1; ++i) {
                    float* ci = C + i * N;
                    for (int k = k0; k < k1; ++k) {
                        float aik = A[i * K + k];
                        const float* bk = B + k * N;
                        int j = j0;
                        if (t->vec >= 4) {
                            for (; j + 3 < j1; j += 4) {
                                ci[j]     += aik * bk[j];
                                ci[j + 1] += aik * bk[j + 1];
                                ci[j + 2] += aik * bk[j + 2];
                                ci[j + 3] += aik * bk[j + 3];
                            }
                        } else if (t->vec == 2) {
                            for (; j + 1 < j1; j += 2) {
                                ci[j]     += aik * bk[j];
                                ci[j + 1] += aik * bk[j + 1];
                            }
                        }
                        for (; j < j1; ++j) {
                            ci[j] += aik * bk[j];
                        }
                    }
                }
            }
        }
    }
    return NULL;
}

void run_gemm(
    const float* A, const float* B, float* C,
    int M, int N, int K, int BM, int BN, int BK, int n_threads, int vec, int swizzle
) {
    if (n_threads < 1) n_threads = 1;
    if (n_threads > 12) n_threads = 12;
    pthread_t threads[12];
    TaskArgs args[12];
    for (int t = 0; t < n_threads; ++t) {
        args[t] = (TaskArgs){t, n_threads, M, N, K, BM, BN, BK, vec, swizzle, A, B, NULL, C};
        pthread_create(&threads[t], NULL, gemm_worker, &args[t]);
    }
    for (int t = 0; t < n_threads; ++t) {
        pthread_join(threads[t], NULL);
    }
}

static void* rmsnorm_worker(void* arg) {
    TaskArgs* t = (TaskArgs*)arg;
    const int M = t->M, N = t->N, BM = t->BM;
    int num_mb = (M + BM - 1) / BM;
    for (int mb = t->tid; mb < num_mb; mb += t->n_threads) {
        int i0 = mb * BM;
        int i1 = (i0 + BM < M) ? (i0 + BM) : M;
        for (int r = i0; r < i1; ++r) {
            const float* x = t->A + r * N;
            const float* res = t->B + r * N;
            const float* w = t->W;
            float* y = t->C + r * N;
            float ss = 0.0f;
            for (int j = 0; j < N; ++j) {
                float v = x[j] + res[j];
                ss += v * v;
            }
            float inv = 1.0f / sqrtf(ss / (float)N + 1e-5f);
            for (int j = 0; j < N; ++j) {
                y[j] = (x[j] + res[j]) * inv * w[j];
            }
        }
    }
    return NULL;
}

void run_rmsnorm(
    const float* X, const float* Res, const float* W, float* Y,
    int M, int N, int BM, int n_threads, int vec
) {
    if (n_threads < 1) n_threads = 1;
    if (n_threads > 12) n_threads = 12;
    pthread_t threads[12];
    TaskArgs args[12];
    for (int t = 0; t < n_threads; ++t) {
        args[t] = (TaskArgs){t, n_threads, M, N, 1, BM, 1, 1, vec, 0, X, Res, W, Y};
        pthread_create(&threads[t], NULL, rmsnorm_worker, &args[t]);
    }
    for (int t = 0; t < n_threads; ++t) pthread_join(threads[t], NULL);
}

static void* softmax_worker(void* arg) {
    TaskArgs* t = (TaskArgs*)arg;
    const int M = t->M, N = t->N, BM = t->BM;
    int num_mb = (M + BM - 1) / BM;
    for (int mb = t->tid; mb < num_mb; mb += t->n_threads) {
        int i0 = mb * BM;
        int i1 = (i0 + BM < M) ? (i0 + BM) : M;
        for (int r = i0; r < i1; ++r) {
            const float* x = t->A + r * N;
            float* y = t->C + r * N;
            float mx = -INFINITY;
            for (int j = 0; j < N; ++j) if (x[j] > mx) mx = x[j];
            float sm = 0.0f;
            for (int j = 0; j < N; ++j) {
                float e = expf(x[j] - mx);
                y[j] = e;
                sm += e;
            }
            float inv = 1.0f / sm;
            for (int j = 0; j < N; ++j) y[j] *= inv;
        }
    }
    return NULL;
}

void run_softmax(
    const float* X, float* Y, int M, int N, int BM, int n_threads, int vec
) {
    if (n_threads < 1) n_threads = 1;
    if (n_threads > 12) n_threads = 12;
    pthread_t threads[12];
    TaskArgs args[12];
    for (int t = 0; t < n_threads; ++t) {
        args[t] = (TaskArgs){t, n_threads, M, N, 1, BM, 1, 1, vec, 0, X, NULL, NULL, Y};
        pthread_create(&threads[t], NULL, softmax_worker, &args[t]);
    }
    for (int t = 0; t < n_threads; ++t) pthread_join(threads[t], NULL);
}

static void* swiglu_worker(void* arg) {
    TaskArgs* t = (TaskArgs*)arg;
    const int total = t->M * t->N;
    const int chunk = (t->BM * t->BN > 256) ? (t->BM * t->BN) : 256;
    int num_chunks = (total + chunk - 1) / chunk;
    for (int ch = t->tid; ch < num_chunks; ch += t->n_threads) {
        int i0 = ch * chunk;
        int i1 = (i0 + chunk < total) ? (i0 + chunk) : total;
        for (int i = i0; i < i1; ++i) {
            float g = t->A[i];
            t->C[i] = (g / (1.0f + expf(-g))) * t->B[i];
        }
    }
    return NULL;
}

void run_swiglu(
    const float* G, const float* U, float* O, int M, int N, int BM, int BN, int n_threads, int vec
) {
    if (n_threads < 1) n_threads = 1;
    if (n_threads > 12) n_threads = 12;
    pthread_t threads[12];
    TaskArgs args[12];
    for (int t = 0; t < n_threads; ++t) {
        args[t] = (TaskArgs){t, n_threads, M, N, 1, BM, BN, 1, vec, 0, G, U, NULL, O};
        pthread_create(&threads[t], NULL, swiglu_worker, &args[t]);
    }
    for (int t = 0; t < n_threads; ++t) pthread_join(threads[t], NULL);
}
"""


class Arm64CpuBenchmarker:
    """Real Apple M2 Max ARM64 multi-core native C/pthreads tiled kernel benchmarker."""

    def __init__(self, warmup: int = 3, iters: int = 10) -> None:
        self.hw = get_hardware("apple_m2_max_cpu")
        self.warmup = warmup
        self.iters = iters
        self._tmpdir = tempfile.TemporaryDirectory(prefix="crossarch_cpu_")
        self._lib = self._compile_library()
        # 40 MB L2 cache eviction array (> 32 MB M2 Max L2)
        self._flush_arr = np.ones((10 * 1024 * 1024,), dtype=np.float32)

    def _compile_library(self) -> ctypes.CDLL:
        c_path = Path(self._tmpdir.name) / "kernels.c"
        so_path = Path(self._tmpdir.name) / "libkernels.dylib"
        c_path.write_text(_C_SOURCE)
        sdk_candidates = [
            None,
            "/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk",
            "/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk",
            "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk",
        ]
        last_err = ""
        for sdk in sdk_candidates:
            if sdk is not None and not Path(sdk).exists():
                continue
            cmd = ["clang", "-O3", "-ffast-math", "-shared", "-fPIC", "-pthread"]
            if sdk is not None:
                cmd.extend(["-isysroot", sdk])
            cmd.extend([str(c_path), "-o", str(so_path)])
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0:
                break
            last_err = res.stderr
        else:
            raise RuntimeError(f"Failed to compile ARM64 C kernels:\n{last_err}")
        lib = ctypes.CDLL(str(so_path))

        f_ptr = ctypes.POINTER(ctypes.c_float)
        c_i = ctypes.c_int
        lib.run_gemm.argtypes = [f_ptr, f_ptr, f_ptr, c_i, c_i, c_i, c_i, c_i, c_i, c_i, c_i, c_i]
        lib.run_rmsnorm.argtypes = [f_ptr, f_ptr, f_ptr, f_ptr, c_i, c_i, c_i, c_i, c_i]
        lib.run_softmax.argtypes = [f_ptr, f_ptr, c_i, c_i, c_i, c_i, c_i]
        lib.run_swiglu.argtypes = [f_ptr, f_ptr, f_ptr, c_i, c_i, c_i, c_i, c_i, c_i]
        return lib

    def _evict_cache(self) -> None:
        self._flush_arr *= 1.00001

    def _time_fn(self, fn) -> Tuple[float, float]:
        for _ in range(self.warmup):
            fn()
        times_ms = []
        for _ in range(self.iters):
            self._evict_cache()
            t0 = time.perf_counter()
            fn()
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1e3)
        arr = np.array(times_ms, dtype=np.float64)
        return float(np.median(arr)), float(np.percentile(arr, 75) - np.percentile(arr, 25))

    def benchmark(self, wl: WorkloadSpec, cfg: ScheduleConfig) -> KernelMeasurement:
        sol_ms = wl.sol_latency_s(self.hw) * 1e3
        rng = np.random.default_rng(0)
        n_threads = min(12, max(1, cfg.num_warps))
        f_ptr = ctypes.POINTER(ctypes.c_float)

        try:
            if wl.op_type == "gemm":
                A = (rng.standard_normal((wl.m, wl.k)).astype(np.float32)) * 0.1
                B = (rng.standard_normal((wl.k, wl.n)).astype(np.float32)) * 0.1
                C = np.zeros((wl.m, wl.n), dtype=np.float32)

                def launch():
                    self._lib.run_gemm(
                        A.ctypes.data_as(f_ptr), B.ctypes.data_as(f_ptr), C.ctypes.data_as(f_ptr),
                        wl.m, wl.n, wl.k, cfg.block_m, cfg.block_n, cfg.block_k, n_threads, cfg.vec_width, cfg.swizzle
                    )

                launch()
                ref = A @ B
                max_err = float(np.max(np.abs(C - ref)))
                verified = bool(np.allclose(C, ref, rtol=1e-2, atol=1e-2))

            elif wl.op_type == "rmsnorm":
                X = rng.standard_normal((wl.m, wl.n)).astype(np.float32)
                Res = rng.standard_normal((wl.m, wl.n)).astype(np.float32)
                W = rng.standard_normal((wl.n,)).astype(np.float32)
                Y = np.zeros_like(X)

                def launch():
                    self._lib.run_rmsnorm(
                        X.ctypes.data_as(f_ptr), Res.ctypes.data_as(f_ptr),
                        W.ctypes.data_as(f_ptr), Y.ctypes.data_as(f_ptr),
                        wl.m, wl.n, cfg.block_m, n_threads, cfg.vec_width
                    )

                launch()
                h = X + Res
                ref = h / np.sqrt(np.mean(h * h, axis=-1, keepdims=True) + 1e-5) * W
                max_err = float(np.max(np.abs(Y - ref)))
                verified = bool(np.allclose(Y, ref, rtol=1e-2, atol=1e-2))

            elif wl.op_type == "softmax":
                X = rng.standard_normal((wl.m, wl.n)).astype(np.float32)
                Y = np.zeros_like(X)

                def launch():
                    self._lib.run_softmax(
                        X.ctypes.data_as(f_ptr), Y.ctypes.data_as(f_ptr),
                        wl.m, wl.n, cfg.block_m, n_threads, cfg.vec_width
                    )

                launch()
                ex = np.exp(X - np.max(X, axis=-1, keepdims=True))
                ref = ex / np.sum(ex, axis=-1, keepdims=True)
                max_err = float(np.max(np.abs(Y - ref)))
                verified = bool(np.allclose(Y, ref, rtol=1e-2, atol=1e-2))

            elif wl.op_type == "swiglu":
                G = rng.standard_normal((wl.m, wl.n)).astype(np.float32)
                U = rng.standard_normal((wl.m, wl.n)).astype(np.float32)
                O = np.zeros_like(G)

                def launch():
                    self._lib.run_swiglu(
                        G.ctypes.data_as(f_ptr), U.ctypes.data_as(f_ptr), O.ctypes.data_as(f_ptr),
                        wl.m, wl.n, cfg.block_m, cfg.block_n, n_threads, cfg.vec_width
                    )

                launch()
                ref = (G / (1.0 + np.exp(-G))) * U
                max_err = float(np.max(np.abs(O - ref)))
                verified = bool(np.allclose(O, ref, rtol=1e-2, atol=1e-2))
            else:
                raise ValueError(f"Unsupported op_type {wl.op_type}")

            if not verified:
                raise ValueError(f"Verification failed: max_err={max_err:.4e}")

            med_ms, iqr_ms = self._time_fn(launch)
            log_eff = float(math.log(max(sol_ms / max(med_ms, 1e-6), 1e-6)))
            return KernelMeasurement(
                hardware="apple_m2_max_cpu",
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
        except Exception as exc:
            return KernelMeasurement(
                hardware="apple_m2_max_cpu",
                workload=wl.name,
                op_type=wl.op_type,
                schedule=cfg.as_dict(),
                verified=False,
                latency_ms=float("inf"),
                iqr_ms=0.0,
                sol_latency_ms=sol_ms,
                log_sol_eff=-6.0,
                max_abs_err=float("inf"),
                error_msg=str(exc),
            )
