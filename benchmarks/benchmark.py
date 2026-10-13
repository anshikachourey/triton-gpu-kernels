"""Benchmarking framework for the custom Triton matmul kernel.

STUB — implemented in Phase 3 (after the correctness suite is confirmed on the
T4). It will provide: tflops(), bench_median() via do_bench(return_mode="median"),
bench_quantiles() via a separate do_bench(quantiles=[0.2, 0.5, 0.8]) run,
environment_info(), run_benchmarks() (preallocated FP16 output on both the torch
and triton paths, OOM / OutOfResources / compile-failure handling that records
the actual error type and message), BenchmarkRecord, write_csv(), plot_latency(),
and plot_throughput().

Intentionally left minimal for Phase 0.
"""

from __future__ import annotations


def tflops(m: int, n: int, k: int, latency_ms: float) -> float:
    """Effective throughput: (2*M*N*K) / (latency_seconds * 1e12)."""
    latency_s = latency_ms / 1000.0
    return (2.0 * m * n * k) / (latency_s * 1e12)
