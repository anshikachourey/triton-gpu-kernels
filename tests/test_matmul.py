"""Correctness tests for the custom Triton matmul kernel.

Phase 0 provides the 256x256 smoke test (`smoke_test_256`). Phase 2 adds the
full correctness suite below: deterministic inputs, FP16 (primary) and FP32
(diagnostic) references, FP32-evaluated error metrics, the required shape cases
(square, rectangular, small non-square, non-multiple-of-tile), the six fixed
tile/warp configurations (num_stages=2), the fixed-tolerance elementwise rule
(atol=rtol=1e-2), per-case records, and a CSV writer.

Everything here runs in Google Colab on the Tesla T4. Kiro has no local GPU and
makes no claim about results until the developer pastes the observed output.

Design rules honored:
  - Compare in FP32 (cast both outputs to float()) so the error calculation
    itself never uses FP16 arithmetic.
  - Fixed tolerances atol=1e-2, rtol=1e-2 are NEVER loosened to force a pass.
  - Record max absolute error and a robust relative error vs BOTH ref_fp16
    (primary / pass-fail) and ref_fp32 (diagnostic), regardless of pass/fail.
  - Attempt EVERY config; record actual compilation / unsupported-config /
    numerical failures rather than hiding them.
  - This does NOT implement benchmarking or plotting.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, asdict

import torch

from kernels.matmul import MatmulConfig, matmul


# Fixed acceptance tolerances (never loosened to force a pass).
ATOL = 1e-2
RTOL = 1e-2

# Avoids division by near-zero reference magnitudes in the robust relative error.
_REL_EPS = 1e-3


# ----------------------------------------------------------------------------
# Data models
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class ShapeCase:
    """One matmul shape to validate."""
    m: int
    n: int
    k: int
    label: str
    seed: int = 0


@dataclass
class CorrectnessRecord:
    """Per (shape x config) correctness result. Mirrors the design schema."""
    label: str
    m: int
    n: int
    k: int
    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int
    seed: int
    status: str                 # "ok" | "error"
    error_reason: str           # "" | actual error type + message
    max_abs_error_fp16: float
    rel_error_fp16: float
    max_abs_error_fp32: float
    rel_error_fp32: float
    atol: float
    rtol: float
    passed: bool
    note: str


# The six fixed tile/warp configurations, num_stages=2 for all.
DEFAULT_CONFIGS = [
    MatmulConfig(64, 64, 32, 4, 2),
    MatmulConfig(64, 64, 32, 8, 2),
    MatmulConfig(128, 128, 32, 4, 2),
    MatmulConfig(128, 128, 32, 8, 2),
    MatmulConfig(128, 64, 32, 4, 2),
    MatmulConfig(128, 64, 32, 8, 2),
]

# Required shape cases: square, rectangular, small non-square, non-multiple-of-tile.
DEFAULT_CASES = [
    ShapeCase(256, 256, 256, "square_256", seed=0),
    ShapeCase(512, 512, 512, "square_512", seed=1),
    ShapeCase(1000, 1500, 700, "rect_1000x1500x700", seed=2),
    ShapeCase(64, 128, 96, "small_nonsquare_64x128x96", seed=3),
    ShapeCase(130, 70, 50, "non_multiple_of_tile_130x70x50", seed=4),
]


# ----------------------------------------------------------------------------
# Inputs and references
# ----------------------------------------------------------------------------
def make_inputs(m: int, k: int, n: int, seed: int = 0):
    """Deterministic FP16 inputs in an FP16-safe magnitude band.

    A: [m, k], B: [k, n]. Seeded per case for reproducibility. Standard-normal
    values keep magnitudes modest so the FP16 reference and the kernel agree
    within hardware precision rather than overflowing FP16.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Run this in Google Colab with a Tesla T4 GPU."
        )
    torch.manual_seed(seed)
    a = torch.randn((m, k), dtype=torch.float16, device="cuda")
    b = torch.randn((k, n), dtype=torch.float16, device="cuda")
    return a, b


def ref_fp16(a, b):
    """PRIMARY reference: torch.matmul on FP16 inputs/output (pass/fail basis)."""
    return torch.matmul(a, b)


def ref_fp32(a, b):
    """DIAGNOSTIC reference: torch.matmul in FP32.

    Helps understand the combined effect of FP32 accumulation and the final
    FP16 output rounding. This does NOT perfectly isolate every source of
    numerical error.
    """
    return torch.matmul(a.float(), b.float())


def error_metrics(c_test, c_ref):
    """Return (max_abs_error, robust_rel_error), both computed in FP32.

    Both operands are cast to FP32 first, so no FP16 arithmetic enters the
    error calculation. The relative error clamps the denominator away from
    near-zero reference magnitudes.
    """
    t = c_test.float()
    r = c_ref.float()
    diff = (t - r).abs()
    max_abs = diff.max().item()
    denom = r.abs().clamp_min(_REL_EPS)
    rel = (diff / denom).max().item()
    return max_abs, rel


# ----------------------------------------------------------------------------
# Correctness runner
# ----------------------------------------------------------------------------
def run_correctness(cases=None, configs=None):
    """Run every (shape x config) pair and return a list of CorrectnessRecord.

    For each pair: attempt the kernel; on any Triton/CUDA error, record
    status="error" with the actual error type + message (never hidden). On
    success, compute FP32-evaluated errors vs ref_fp16 (primary) and ref_fp32
    (diagnostic), apply the fixed-tolerance elementwise rule to ref_fp16, and
    record passed/failed with the raw metrics preserved either way.
    Tolerances are NEVER loosened to force a pass.
    """
    cases = cases if cases is not None else DEFAULT_CASES
    configs = configs if configs is not None else DEFAULT_CONFIGS
    records = []

    for case in cases:
        a, b = make_inputs(case.m, case.k, case.n, seed=case.seed)
        r16 = ref_fp16(a, b)
        r32 = ref_fp32(a, b)

        for cfg in configs:
            base = dict(
                label=case.label, m=case.m, n=case.n, k=case.k,
                block_m=cfg.block_m, block_n=cfg.block_n, block_k=cfg.block_k,
                num_warps=cfg.num_warps, num_stages=cfg.num_stages, seed=case.seed,
                atol=ATOL, rtol=RTOL,
            )
            try:
                c = torch.empty((case.m, case.n), dtype=torch.float16, device="cuda")
                matmul(a, b, cfg, out=c)
                torch.cuda.synchronize()
            except Exception as exc:  # compilation / unsupported-config / launch
                records.append(CorrectnessRecord(
                    **base, status="error",
                    error_reason=f"{type(exc).__name__}: {exc}",
                    max_abs_error_fp16=float("nan"), rel_error_fp16=float("nan"),
                    max_abs_error_fp32=float("nan"), rel_error_fp32=float("nan"),
                    passed=False, note="kernel raised; see error_reason",
                ))
                torch.cuda.empty_cache()
                continue

            max_abs_16, rel_16 = error_metrics(c, r16)
            max_abs_32, rel_32 = error_metrics(c, r32)

            # Fixed-tolerance elementwise rule against the FP16 primary reference,
            # evaluated in FP32. Never loosened.
            passed = bool(
                torch.allclose(c.float(), r16.float(), atol=ATOL, rtol=RTOL)
            )
            note = "" if passed else (
                "elementwise tolerance exceeded; investigate (not loosened)"
            )
            records.append(CorrectnessRecord(
                **base, status="ok", error_reason="",
                max_abs_error_fp16=max_abs_16, rel_error_fp16=rel_16,
                max_abs_error_fp32=max_abs_32, rel_error_fp32=rel_32,
                passed=passed, note=note,
            ))

    return records


_CSV_FIELDS = [
    "label", "m", "n", "k", "block_m", "block_n", "block_k", "num_warps",
    "num_stages", "seed", "status", "error_reason",
    "max_abs_error_fp16", "rel_error_fp16", "max_abs_error_fp32", "rel_error_fp32",
    "atol", "rtol", "passed", "note",
]


def write_correctness_csv(records, path="results/correctness.csv"):
    """Write correctness records to CSV with the fixed schema."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for rec in records:
            writer.writerow({k: asdict(rec)[k] for k in _CSV_FIELDS})
    return path


# ----------------------------------------------------------------------------
# Phase 0 smoke test (retained)
# ----------------------------------------------------------------------------
def smoke_test_256(seed: int = 0) -> dict:
    """Minimal 256x256 smoke test with the conservative default config.

    Config: BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, num_warps=4, num_stages=2.
    Runs the custom kernel, compares against torch.matmul in FP32, and applies
    the explicit elementwise tolerance check (atol=rtol=1e-2).
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Run this in Google Colab with a Tesla T4 GPU."
        )

    torch.manual_seed(seed)
    device = "cuda"

    a = torch.randn((256, 256), dtype=torch.float16, device=device)
    b = torch.randn((256, 256), dtype=torch.float16, device=device)

    config = MatmulConfig(block_m=64, block_n=64, block_k=32, num_warps=4, num_stages=2)

    c_triton = torch.empty((256, 256), dtype=torch.float16, device=device)
    matmul(a, b, config, out=c_triton)

    c_ref = torch.matmul(a, b)

    diff = (c_triton.float() - c_ref.float()).abs()
    max_abs_error = diff.max().item()
    denom = c_ref.float().abs().clamp_min(_REL_EPS)
    robust_rel_error = (diff / denom).max().item()

    try:
        torch.testing.assert_close(
            c_triton.float(), c_ref.float(), atol=ATOL, rtol=RTOL
        )
        passed = True
        check_message = "PASS"
    except AssertionError as exc:
        passed = False
        check_message = f"FAIL: {exc}"

    return {
        "shape": tuple(c_triton.shape),
        "dtype": str(c_triton.dtype),
        "config": (config.block_m, config.block_n, config.block_k,
                   config.num_warps, config.num_stages),
        "max_abs_error_fp32eval": max_abs_error,
        "robust_rel_error_fp32eval": robust_rel_error,
        "atol": ATOL,
        "rtol": RTOL,
        "passed": passed,
        "check_message": check_message,
    }


def summarize(records) -> dict:
    """Small summary for printing: counts of ok/error/passed/failed."""
    total = len(records)
    errored = sum(1 for r in records if r.status == "error")
    passed = sum(1 for r in records if r.status == "ok" and r.passed)
    failed = sum(1 for r in records if r.status == "ok" and not r.passed)
    return {"total": total, "passed": passed, "failed": failed, "errored": errored}


if __name__ == "__main__":
    out = smoke_test_256()
    print("Smoke test 256x256")
    print(f"  output shape         : {out['shape']}")
    print(f"  output dtype         : {out['dtype']}")
    print(f"  config (BM,BN,BK,W,S): {out['config']}")
    print(f"  max abs error (FP32) : {out['max_abs_error_fp32eval']:.6e}")
    print(f"  robust rel error     : {out['robust_rel_error_fp32eval']:.6e}")
    print(f"  tolerance (atol,rtol): ({out['atol']}, {out['rtol']})")
    print(f"  elementwise check    : {out['check_message']}")
    print(f"  passed               : {out['passed']}")

    print("\nFull correctness suite")
    recs = run_correctness()
    path = write_correctness_csv(recs)
    s = summarize(recs)
    print(f"  wrote {path}")
    print(f"  total={s['total']} passed={s['passed']} "
          f"failed={s['failed']} errored={s['errored']}")
    for r in recs:
        tag = r.status if r.status == "error" else ("PASS" if r.passed else "FAIL")
        print(f"  [{tag}] {r.label} cfg=({r.block_m},{r.block_n},{r.block_k},"
              f"{r.num_warps},{r.num_stages}) "
              f"max_abs_fp16={r.max_abs_error_fp16} "
              f"max_abs_fp32={r.max_abs_error_fp32} "
              f"{r.error_reason}")
