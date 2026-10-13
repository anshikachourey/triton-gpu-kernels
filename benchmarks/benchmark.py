"""Benchmarking framework for the custom Triton matmul kernel (Phase 3).

Runs in Google Colab on the Tesla T4. Kiro has no local GPU and makes no claim
about measurements until the developer pastes the observed output.

Methodology (per the approved design):
  - Benchmark the IDENTICAL operation C = A @ B on BOTH sides with PREALLOCATED
    FP16 output buffers:
        torch.matmul(a, b, out=c_torch)
        matmul(a, b, config, out=c_triton)
    so neither side pays for an output allocation inside the timed region.
  - PRIMARY latency: triton.testing.do_bench(fn, return_mode="median").
  - Additionally record quantiles p20/p50/p80 from a SEPARATE
    do_bench(fn, quantiles=[0.2, 0.5, 0.8]) invocation. The quantile-run p50 is
    recorded independently and is NOT assumed equal to the primary median.
  - Exclude one-time Triton compilation from steady-state via a throwaway launch
    before timing (do_bench also warms up internally).
  - TFLOPS = (2*M*N*K) / (latency_seconds * 1e12). Speedup from measured
    latencies only.
  - Attempt every size/config. Catch triton.runtime.errors.OutOfResources,
    Triton compilation failures, and torch.cuda.OutOfMemoryError; record a
    skipped row carrying the ACTUAL error type + message in skip_reason; free
    memory and continue. Regressions and unsupported configs are valid findings.

This module does NOT implement plotting (that is Phase 5) and does not change
the kernel, input generation, or the six configurations.
"""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass, asdict

import torch
import triton
import triton.testing

from kernels.matmul import MatmulConfig, matmul

# Import OutOfResources defensively (its import path can vary across Triton
# versions). We still catch any Exception as a fallback below.
try:
    from triton.runtime.errors import OutOfResources  # type: ignore
except Exception:  # pragma: no cover - path differs across versions
    class OutOfResources(Exception):
        """Fallback placeholder if the real class is not importable."""


# ----------------------------------------------------------------------------
# Data models
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class ShapeCase:
    m: int
    n: int
    k: int
    label: str
    seed: int = 0


@dataclass(frozen=True)
class EnvInfo:
    gpu_model: str
    compute_capability: str
    torch_version: str
    cuda_version: str
    triton_version: str
    cuda_available: bool


@dataclass
class BenchmarkRecord:
    label: str
    m: int
    n: int
    k: int
    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int
    impl: str                 # "triton_custom" | "torch"
    status: str               # "ok" | "skipped"
    skip_reason: str          # "" | actual error type + message
    latency_ms_primary_median: float
    latency_ms_p20: float
    latency_ms_p50: float
    latency_ms_p80: float
    tflops: float
    speedup_vs_torch: float
    gpu_model: str
    compute_capability: str
    torch_version: str
    cuda_version: str
    triton_version: str


# Benchmark sizes from the design.
DEFAULT_SIZES = [
    ShapeCase(512, 512, 512, "512^3", seed=0),
    ShapeCase(1024, 1024, 1024, "1024^3", seed=0),
    ShapeCase(2048, 2048, 2048, "2048^3", seed=0),
    ShapeCase(4096, 4096, 4096, "4096^3", seed=0),
    ShapeCase(1000, 1500, 700, "rect_1000x1500x700", seed=2),
]

# The six fixed configurations (num_stages=2). For the default run we time the
# baseline config; Phase 4 sweeps the full matrix.
DEFAULT_CONFIGS = [
    MatmulConfig(64, 64, 32, 4, 2),
    MatmulConfig(64, 64, 32, 8, 2),
    MatmulConfig(128, 128, 32, 4, 2),
    MatmulConfig(128, 128, 32, 8, 2),
    MatmulConfig(128, 64, 32, 4, 2),
    MatmulConfig(128, 64, 32, 8, 2),
]


# ----------------------------------------------------------------------------
# Metrics / timing primitives
# ----------------------------------------------------------------------------
def tflops(m: int, n: int, k: int, latency_ms: float) -> float:
    """Effective throughput: (2*M*N*K) / (latency_seconds * 1e12)."""
    latency_s = latency_ms / 1000.0
    return (2.0 * m * n * k) / (latency_s * 1e12)


def bench_median(fn) -> float:
    """PRIMARY latency in ms via do_bench(..., return_mode='median')."""
    return float(triton.testing.do_bench(fn, return_mode="median"))


def bench_quantiles(fn):
    """Latency quantiles (p20, p50, p80) in ms via a SEPARATE do_bench run."""
    q = triton.testing.do_bench(fn, quantiles=[0.2, 0.5, 0.8])
    p20, p50, p80 = (float(x) for x in q)
    return p20, p50, p80


def environment_info() -> EnvInfo:
    """Collect GPU + software environment metadata."""
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        gpu_model = torch.cuda.get_device_name(0)
        cc = torch.cuda.get_device_capability(0)
        compute_capability = f"{cc[0]}.{cc[1]}"
    else:
        gpu_model = "N/A"
        compute_capability = "N/A"
    return EnvInfo(
        gpu_model=gpu_model,
        compute_capability=compute_capability,
        torch_version=torch.__version__,
        cuda_version=str(torch.version.cuda),
        triton_version=triton.__version__,
        cuda_available=cuda_available,
    )


def _make_inputs(m, k, n, seed):
    """Deterministic FP16 inputs (same scheme as the correctness suite)."""
    torch.manual_seed(seed)
    a = torch.randn((m, k), dtype=torch.float16, device="cuda")
    b = torch.randn((k, n), dtype=torch.float16, device="cuda")
    return a, b


# ----------------------------------------------------------------------------
# Benchmark runner
# ----------------------------------------------------------------------------
def run_benchmarks(sizes=None, configs=None):
    """Benchmark custom kernel vs torch.matmul for each size x config.

    For each size and config:
      - allocate FP16 inputs + a preallocated FP16 output per side
      - time the Triton kernel (preallocated out) and torch.matmul (preallocated
        out) with do_bench median + quantiles
      - record TFLOPS and the Triton-vs-torch speedup from measured medians
      - on OutOfResources / compile failure / OOM: record a skipped Triton row
        with the actual error type + message, free memory, continue

    Returns a list of BenchmarkRecord (two rows per attempted size x config:
    one triton_custom, one torch; the torch row is skipped only if its own
    allocation/launch fails).
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Run this in Google Colab with a Tesla T4 GPU."
        )
    sizes = sizes if sizes is not None else DEFAULT_SIZES
    configs = configs if configs is not None else DEFAULT_CONFIGS
    env = environment_info()
    records = []

    for case in sizes:
        for cfg in configs:
            m, n, k = case.m, case.n, case.k
            base = dict(
                label=case.label, m=m, n=n, k=k,
                block_m=cfg.block_m, block_n=cfg.block_n, block_k=cfg.block_k,
                num_warps=cfg.num_warps, num_stages=cfg.num_stages,
                gpu_model=env.gpu_model, compute_capability=env.compute_capability,
                torch_version=env.torch_version, cuda_version=env.cuda_version,
                triton_version=env.triton_version,
            )

            triton_median = math.nan

            # ---- Triton custom kernel ------------------------------------
            try:
                a, b = _make_inputs(m, k, n, case.seed)
                c_triton = torch.empty((m, n), dtype=torch.float16, device="cuda")

                def run_triton():
                    matmul(a, b, cfg, out=c_triton)

                run_triton()                 # throwaway launch: exclude compile
                torch.cuda.synchronize()
                triton_median = bench_median(run_triton)
                p20, p50, p80 = bench_quantiles(run_triton)

                records.append(BenchmarkRecord(
                    **base, impl="triton_custom", status="ok", skip_reason="",
                    latency_ms_primary_median=triton_median,
                    latency_ms_p20=p20, latency_ms_p50=p50, latency_ms_p80=p80,
                    tflops=tflops(m, n, k, triton_median),
                    speedup_vs_torch=math.nan,  # filled after torch row
                ))
            except (OutOfResources, torch.cuda.OutOfMemoryError, Exception) as exc:
                records.append(BenchmarkRecord(
                    **base, impl="triton_custom", status="skipped",
                    skip_reason=f"{type(exc).__name__}: {exc}",
                    latency_ms_primary_median=math.nan,
                    latency_ms_p20=math.nan, latency_ms_p50=math.nan,
                    latency_ms_p80=math.nan,
                    tflops=math.nan, speedup_vs_torch=math.nan,
                ))
                triton_median = math.nan
            finally:
                torch.cuda.empty_cache()

            # ---- torch.matmul (preallocated out, same operation) ---------
            torch_median = math.nan
            try:
                a, b = _make_inputs(m, k, n, case.seed)
                c_torch = torch.empty((m, n), dtype=torch.float16, device="cuda")

                def run_torch():
                    torch.matmul(a, b, out=c_torch)

                run_torch()
                torch.cuda.synchronize()
                torch_median = bench_median(run_torch)
                p20, p50, p80 = bench_quantiles(run_torch)

                records.append(BenchmarkRecord(
                    **base, impl="torch", status="ok", skip_reason="",
                    latency_ms_primary_median=torch_median,
                    latency_ms_p20=p20, latency_ms_p50=p50, latency_ms_p80=p80,
                    tflops=tflops(m, n, k, torch_median),
                    speedup_vs_torch=math.nan,  # speedup is defined on the triton row
                ))
            except (torch.cuda.OutOfMemoryError, Exception) as exc:
                records.append(BenchmarkRecord(
                    **base, impl="torch", status="skipped",
                    skip_reason=f"{type(exc).__name__}: {exc}",
                    latency_ms_primary_median=math.nan,
                    latency_ms_p20=math.nan, latency_ms_p50=math.nan,
                    latency_ms_p80=math.nan,
                    tflops=math.nan, speedup_vs_torch=math.nan,
                ))
            finally:
                torch.cuda.empty_cache()

            # ---- Speedup (triton row) from measured medians --------------
            if not math.isnan(triton_median) and not math.isnan(torch_median) \
                    and triton_median > 0:
                speedup = torch_median / triton_median
                # Attach to the most recent triton_custom "ok" row for this pair.
                for rec in reversed(records):
                    if rec.impl == "triton_custom" and rec.label == case.label \
                            and rec.block_m == cfg.block_m \
                            and rec.num_warps == cfg.num_warps \
                            and rec.status == "ok":
                        rec.speedup_vs_torch = speedup
                        break

    return records


_CSV_FIELDS = [
    "label", "m", "n", "k", "block_m", "block_n", "block_k", "num_warps",
    "num_stages", "impl", "status", "skip_reason",
    "latency_ms_primary_median", "latency_ms_p20", "latency_ms_p50",
    "latency_ms_p80", "tflops", "speedup_vs_torch",
    "gpu_model", "compute_capability", "torch_version", "cuda_version",
    "triton_version",
]


def write_csv(records, path="results/benchmarks.csv"):
    """Write benchmark records to CSV with the fixed schema."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for rec in records:
            writer.writerow({key: asdict(rec)[key] for key in _CSV_FIELDS})
    return path


def print_benchmarks(records) -> None:
    """Compact preview of benchmark records."""
    print("label                 impl           status   "
          "median_ms   tflops     speedup  skip_reason")
    for r in records:
        med = "%.4f" % r.latency_ms_primary_median \
            if not math.isnan(r.latency_ms_primary_median) else "nan"
        tfl = "%.3f" % r.tflops if not math.isnan(r.tflops) else "nan"
        spd = "%.3f" % r.speedup_vs_torch \
            if not math.isnan(r.speedup_vs_torch) else "-"
        print("%-21s %-14s %-8s %-11s %-10s %-8s %s" % (
            r.label, r.impl, r.status, med, tfl, spd, r.skip_reason))
