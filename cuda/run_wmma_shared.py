"""Harness for the OPTIMIZED shared-memory CUDA WMMA kernel.

Adds the shared-memory kernel to the comparison WITHOUT touching the existing
run_wmma.py (naive baseline harness) or any kernel source. Reuses run_wmma's
input generation, Triton call, and timing helpers so all paths see identical
inputs and the same measurement methodology.

Four-way comparison:
  * naive CUDA WMMA        (libwmma_matmul.so, wmma_matmul_launch)
  * shared CUDA WMMA       (libwmma_matmul_shared.so, wmma_matmul_shared_launch)
  * Triton kernel          (kernels.matmul.matmul)
  * PyTorch matmul

Nothing here claims an improvement. It reports measured numbers for review on
the Tesla T4. Writes to results/diagnostics/cuda/ (separate files; the naive
harness outputs are not overwritten).
"""

from __future__ import annotations

import ctypes
import json
import os
import statistics

import torch

from kernels.matmul import MatmulConfig, matmul as triton_matmul

# Reuse the baseline harness helpers (naive loader, timing, inputs, triton).
import run_wmma  # same directory; loaded by file path in Colab

_OK = 0
_ERR = {0: "OK", -1: "null pointer", -2: "non-positive dim",
        -3: "not a multiple of 16", -4: "not a multiple of the 64 block tile"}

_SHARED_LIB = os.path.join(os.path.dirname(__file__), "libwmma_matmul_shared.so")
_CUDA_DIAG_DIR = "results/diagnostics/cuda"


def load_shared_library(lib_path: str = _SHARED_LIB):
    """Load the shared-memory kernel library and declare its C signature."""
    if not os.path.exists(lib_path):
        raise FileNotFoundError(
            f"{lib_path} not found. Compile it first with:\n"
            f"  nvcc -arch=sm_75 -Xcompiler -fPIC -shared "
            f"cuda/wmma_matmul_shared.cu -o cuda/libwmma_matmul_shared.so"
        )
    lib = ctypes.CDLL(lib_path)
    lib.wmma_matmul_shared_launch.restype = ctypes.c_int
    lib.wmma_matmul_shared_launch.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    return lib


def cuda_matmul_shared(lib, a: torch.Tensor, b: torch.Tensor,
                       out: torch.Tensor | None = None) -> torch.Tensor:
    """Call the shared-memory CUDA WMMA kernel via ctypes + data_ptr()."""
    assert a.dtype == torch.float16 and b.dtype == torch.float16
    assert a.is_cuda and b.is_cuda and a.is_contiguous() and b.is_contiguous()
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = out if out is not None else torch.empty((M, N), dtype=torch.float16,
                                                device=a.device)
    assert c.is_contiguous()
    code = lib.wmma_matmul_shared_launch(
        ctypes.c_void_p(a.data_ptr()),
        ctypes.c_void_p(b.data_ptr()),
        ctypes.c_void_p(c.data_ptr()),
        ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K),
    )
    torch.cuda.synchronize()
    if code != _OK:
        raise RuntimeError(
            f"wmma_matmul_shared_launch returned {code} "
            f"({_ERR.get(code, 'launch error')})"
        )
    return c


# ---------------------------------------------------------------------------
# Correctness: shared CUDA vs FP64->FP16, vs naive CUDA, vs Triton
# ---------------------------------------------------------------------------
def correctness(naive_lib, shared_lib, sizes=(256, 1024), seed=0,
                atol=1e-2, rtol=1e-2, out_dir=_CUDA_DIAG_DIR):
    os.makedirs(out_dir, exist_ok=True)
    cfg = MatmulConfig(64, 64, 32, 4, 2)
    records = []
    for n in sizes:
        a, b = run_wmma._make_inputs(n, seed)

        c_naive = run_wmma.cuda_matmul(naive_lib, a, b)
        c_shared = cuda_matmul_shared(shared_lib, a, b)
        c_triton = triton_matmul(a, b, cfg)
        ref_fp16 = torch.matmul(a.double(), b.double()).half()

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

        records.append({
            "size": n, "seed": seed, "atol": atol, "rtol": rtol,
            "shared_vs_fp64rounded": stats(c_shared),
            "naive_vs_fp64rounded": stats(c_naive),
            "triton_vs_fp64rounded": stats(c_triton),
            "shared_vs_naive_max_abs":
                float((c_shared.float() - c_naive.float()).abs().max().item()),
            "shared_bitwise_equals_naive": bool(torch.equal(c_shared, c_naive)),
            "shared_bitwise_equals_triton": bool(torch.equal(c_shared, c_triton)),
        })

    with open(os.path.join(out_dir, "cuda_shared_correctness.json"), "w") as f:
        json.dump(records, f, indent=2)
    return records


# ---------------------------------------------------------------------------
# Performance: interleaved shared / naive / Triton / PyTorch
# ---------------------------------------------------------------------------
def performance(naive_lib, shared_lib, sizes=(256, 1024), seed=0, repeats=5,
                out_dir=_CUDA_DIAG_DIR):
    os.makedirs(out_dir, exist_ok=True)
    cfg = MatmulConfig(64, 64, 32, 4, 2)
    rows = []
    for n in sizes:
        a, b = run_wmma._make_inputs(n, seed)
        c_sh = torch.empty((n, n), dtype=torch.float16, device="cuda")
        c_nv = torch.empty((n, n), dtype=torch.float16, device="cuda")
        c_tr = torch.empty((n, n), dtype=torch.float16, device="cuda")
        c_to = torch.empty((n, n), dtype=torch.float16, device="cuda")

        for r in range(repeats):
            clk_before = run_wmma._gpu_clock_mhz()
            # Interleaved order each repeat to reduce clock bias.
            t_sh = run_wmma._time_ms(lambda: cuda_matmul_shared(shared_lib, a, b, out=c_sh))
            t_nv = run_wmma._time_ms(lambda: run_wmma.cuda_matmul(naive_lib, a, b, out=c_nv))
            t_tr = run_wmma._time_ms(lambda: triton_matmul(a, b, cfg, out=c_tr))
            t_to = run_wmma._time_ms(lambda: torch.matmul(a, b, out=c_to))
            clk_after = run_wmma._gpu_clock_mhz()

            def tflops(ms):
                return (2.0 * n * n * n) / ((ms / 1000.0) * 1e12)

            rows.append({
                "size": n, "repeat": r,
                "shared_ms": t_sh, "naive_ms": t_nv,
                "triton_ms": t_tr, "torch_ms": t_to,
                "shared_tflops": tflops(t_sh), "naive_tflops": tflops(t_nv),
                # Ratio names state the formula and direction explicitly:
                # value > 1 means the shared kernel is FASTER than the other.
                "shared_faster_than_naive": t_nv / t_sh,    # naive_ms / shared_ms
                "shared_faster_than_triton": t_tr / t_sh,   # triton_ms / shared_ms
                "shared_faster_than_torch": t_to / t_sh,    # torch_ms / shared_ms
                "sm_clock_mhz_before": clk_before,
                "sm_clock_mhz_after": clk_after,
            })

    with open(os.path.join(out_dir, "cuda_shared_performance.json"), "w") as f:
        json.dump(rows, f, indent=2)
    return rows


def print_correctness(records):
    print("Shared-memory CUDA WMMA correctness (vs FP64->FP16 reference)")
    for rec in records:
        s = rec["shared_vs_fp64rounded"]
        print(f"  size={rec['size']}: max_abs={s['max_abs_error']:.6e} "
              f"violations={s['tol_violations']} "
              f"exact={s['exact_matches_vs_fp64rounded']} "
              f"({s['exact_match_pct']:.4f}%) | "
              f"shared==naive(bitwise)={rec['shared_bitwise_equals_naive']} "
              f"shared-vs-naive max_abs={rec['shared_vs_naive_max_abs']:.6e}")


def print_performance(rows):
    # Ratio columns: value > 1 means the shared kernel is faster than that path.
    print("Shared-memory CUDA WMMA performance (median ms; interleaved repeats)")
    print("  size rep shared_ms naive_ms  triton_ms torch_ms  "
          "shared_faster_than_naive(x) shared_faster_than_torch(x) clk_before")
    for r in rows:
        print("  %-4d %-3d %-9.4f %-9.4f %-9.4f %-9.4f %-27.3f %-27.3f %s" % (
            r["size"], r["repeat"], r["shared_ms"], r["naive_ms"],
            r["triton_ms"], r["torch_ms"],
            r["shared_faster_than_naive"], r["shared_faster_than_torch"],
            str(r["sm_clock_mhz_before"])))
