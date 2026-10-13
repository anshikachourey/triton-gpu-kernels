"""Harness for the naive CUDA C++ WMMA matmul extension.

Loads libwmma_matmul.so via ctypes (NO torch.utils.cpp_extension, NO installs)
and feeds the SAME torch FP16 tensors to the CUDA, Triton, and PyTorch paths so
all three compute the identical operation on identical inputs.

Scope: square, 16-aligned workloads only (256x256, 1024x1024 to start). This is
an investigation extension; it does not modify the Triton kernel or any original
result. All new artifacts go under results/diagnostics/cuda/.

Nothing here claims correctness or performance — it reports measured numbers for
the developer to review after running on the Tesla T4.
"""

from __future__ import annotations

import ctypes
import json
import os
import statistics

import torch

# Triton kernel from the existing project (unchanged).
from kernels.matmul import MatmulConfig, matmul as triton_matmul

# Error codes mirrored from wmma_matmul.cu.
_WMMA_OK = 0
_ERR = {0: "OK", -1: "null pointer", -2: "non-positive dim",
        -3: "dimension not a multiple of 16"}

_DEFAULT_LIB = os.path.join(os.path.dirname(__file__), "libwmma_matmul.so")
_CUDA_DIAG_DIR = "results/diagnostics/cuda"


def load_library(lib_path: str = _DEFAULT_LIB):
    """Load the compiled shared library and declare the C signature."""
    if not os.path.exists(lib_path):
        raise FileNotFoundError(
            f"{lib_path} not found. Compile it first with:\n"
            f"  nvcc -arch=sm_75 -Xcompiler -fPIC -shared "
            f"cuda/wmma_matmul.cu -o cuda/libwmma_matmul.so"
        )
    lib = ctypes.CDLL(lib_path)
    # int wmma_matmul_launch(const half* A, const half* B, half* C, int M,N,K)
    lib.wmma_matmul_launch.restype = ctypes.c_int
    lib.wmma_matmul_launch.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    return lib


def cuda_matmul(lib, a: torch.Tensor, b: torch.Tensor,
                out: torch.Tensor | None = None) -> torch.Tensor:
    """Call the CUDA WMMA kernel on torch FP16 tensors via their data_ptr().

    Raises on a non-zero error code so unsupported shapes never pass silently.
    """
    assert a.dtype == torch.float16 and b.dtype == torch.float16
    assert a.is_cuda and b.is_cuda and a.is_contiguous() and b.is_contiguous()
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = out if out is not None else torch.empty((M, N), dtype=torch.float16,
                                                device=a.device)
    assert c.is_contiguous()
    code = lib.wmma_matmul_launch(
        ctypes.c_void_p(a.data_ptr()),
        ctypes.c_void_p(b.data_ptr()),
        ctypes.c_void_p(c.data_ptr()),
        ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K),
    )
    torch.cuda.synchronize()
    if code != _WMMA_OK:
        raise RuntimeError(
            f"wmma_matmul_launch returned {code} ({_ERR.get(code, 'launch error')})"
        )
    return c


def _make_inputs(n, seed=0):
    """Deterministic FP16 square inputs (same scheme as the project)."""
    torch.manual_seed(seed)
    a = torch.randn((n, n), dtype=torch.float16, device="cuda")
    b = torch.randn((n, n), dtype=torch.float16, device="cuda")
    return a, b


# ---------------------------------------------------------------------------
# Phase 2: correctness (CUDA vs FP64-derived reference and vs Triton)
# ---------------------------------------------------------------------------
def correctness(lib, sizes=(256, 1024), seed=0, atol=1e-2, rtol=1e-2,
                out_dir=_CUDA_DIAG_DIR):
    """Compare CUDA WMMA output vs FP64->FP16 reference and vs the Triton output.

    Uses the SAME inputs for all paths. Reports max abs error, tolerance
    violations, and exact-match counts (bitwise FP16) against the rounded FP64
    reference. Makes no pass/fail claim beyond what the numbers show.
    """
    os.makedirs(out_dir, exist_ok=True)
    cfg = MatmulConfig(64, 64, 32, 4, 2)
    records = []
    for n in sizes:
        a, b = _make_inputs(n, seed)

        c_cuda = cuda_matmul(lib, a, b)
        c_triton = triton_matmul(a, b, cfg)
        ref_fp64 = torch.matmul(a.double(), b.double())
        ref_fp16 = ref_fp64.half()

        def stats(c):
            diff = (c.float() - ref_fp16.float()).abs()
            allowed = atol + rtol * ref_fp16.float().abs()
            viol = int((diff > allowed).sum().item())
            exact = int((c == ref_fp16).sum().item())
            return {
                "max_abs_error": float(diff.max().item()),
                "mean_abs_error": float(diff.mean().item()),
                "tol_violations": viol,
                "exact_matches_vs_fp64rounded": exact,
                "exact_match_pct": 100.0 * exact / c.numel(),
            }

        cuda_vs_ref = stats(c_cuda)
        triton_vs_ref = stats(c_triton)
        cuda_vs_triton_max_abs = float(
            (c_cuda.float() - c_triton.float()).abs().max().item())
        cuda_equals_triton = bool(torch.equal(c_cuda, c_triton))

        records.append({
            "size": n, "seed": seed, "atol": atol, "rtol": rtol,
            "cuda_vs_fp64rounded": cuda_vs_ref,
            "triton_vs_fp64rounded": triton_vs_ref,
            "cuda_vs_triton_max_abs": cuda_vs_triton_max_abs,
            "cuda_bitwise_equals_triton": cuda_equals_triton,
        })

    with open(os.path.join(out_dir, "cuda_correctness.json"), "w") as f:
        json.dump(records, f, indent=2)
    return records


# ---------------------------------------------------------------------------
# Phase 4: performance (CUDA vs Triton vs PyTorch), interleaved repeats
# ---------------------------------------------------------------------------
def _gpu_clock_mhz():
    try:
        fn = getattr(torch.cuda, "clock_rate", None)
        return fn(0) if fn else None
    except Exception:
        return None


def _time_ms(fn, warmup=10, iters=50):
    """Warmed-up, synchronized median latency in ms (manual CUDA-event timing)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))  # ms
    return statistics.median(times)


def performance(lib, sizes=(256, 1024), seed=0, repeats=5,
                out_dir=_CUDA_DIAG_DIR):
    """Interleaved CUDA / Triton / PyTorch timing with preallocated FP16 output.

    For each repeat we measure the three paths back-to-back (interleaved order)
    and capture the SM clock, so clock drift is visible per repeat. Preallocated
    output buffers are reused on all three paths where practical.
    """
    os.makedirs(out_dir, exist_ok=True)
    cfg = MatmulConfig(64, 64, 32, 4, 2)
    rows = []
    for n in sizes:
        a, b = _make_inputs(n, seed)
        c_cuda = torch.empty((n, n), dtype=torch.float16, device="cuda")
        c_tri = torch.empty((n, n), dtype=torch.float16, device="cuda")
        c_torch = torch.empty((n, n), dtype=torch.float16, device="cuda")

        for r in range(repeats):
            clk_before = _gpu_clock_mhz()
            t_cuda = _time_ms(lambda: cuda_matmul(lib, a, b, out=c_cuda))
            t_tri = _time_ms(lambda: triton_matmul(a, b, cfg, out=c_tri))
            t_torch = _time_ms(lambda: torch.matmul(a, b, out=c_torch))
            clk_after = _gpu_clock_mhz()

            def tflops(ms):
                return (2.0 * n * n * n) / ((ms / 1000.0) * 1e12)

            rows.append({
                "size": n, "repeat": r,
                "cuda_ms": t_cuda, "triton_ms": t_tri, "torch_ms": t_torch,
                "cuda_tflops": tflops(t_cuda),
                "triton_tflops": tflops(t_tri),
                "torch_tflops": tflops(t_torch),
                # Speedup = how many times FASTER CUDA is than the other path,
                # i.e. other_ms / cuda_ms. A value > 1 means CUDA is faster.
                "cuda_speedup_over_torch": t_torch / t_cuda,   # torch_ms / cuda_ms
                "cuda_speedup_over_triton": t_tri / t_cuda,    # triton_ms / cuda_ms
                "sm_clock_mhz_before": clk_before,
                "sm_clock_mhz_after": clk_after,
            })

    with open(os.path.join(out_dir, "cuda_performance.json"), "w") as f:
        json.dump(rows, f, indent=2)
    return rows


def print_correctness(records):
    print("CUDA WMMA correctness (vs FP64->FP16 reference)")
    for rec in records:
        c = rec["cuda_vs_fp64rounded"]
        print(f"  size={rec['size']}: max_abs={c['max_abs_error']:.6e} "
              f"violations={c['tol_violations']} "
              f"exact={c['exact_matches_vs_fp64rounded']} "
              f"({c['exact_match_pct']:.4f}%) | "
              f"cuda==triton(bitwise)={rec['cuda_bitwise_equals_triton']} "
              f"cuda-vs-triton max_abs={rec['cuda_vs_triton_max_abs']:.6e}")


def print_performance(rows):
    # The two ratio columns are "how many times faster CUDA is" =
    # torch_ms/cuda_ms and triton_ms/cuda_ms (value > 1 => CUDA faster).
    print("CUDA WMMA performance (median ms; interleaved repeats)")
    print("  size repeat cuda_ms   triton_ms torch_ms  "
          "cuda_faster_than_torch(x) cuda_faster_than_triton(x) clk_before")
    for r in rows:
        # Accept either the new keys or the earlier (now-renamed) keys.
        s_torch = r.get("cuda_speedup_over_torch", r.get("cuda_vs_torch_speedup"))
        s_triton = r.get("cuda_speedup_over_triton", r.get("cuda_vs_triton_speedup"))
        print("  %-4d %-6d %-9.4f %-9.4f %-9.4f %-25.3f %-26.3f %s" % (
            r["size"], r["repeat"], r["cuda_ms"], r["triton_ms"], r["torch_ms"],
            s_torch, s_triton, str(r["sm_clock_mhz_before"])))
