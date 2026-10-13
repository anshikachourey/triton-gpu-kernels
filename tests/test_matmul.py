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


def diagnose_case(m: int = 1000, n: int = 1500, k: int = 700,
                  seed: int = 2,
                  config: MatmulConfig = MatmulConfig(64, 64, 32, 4, 2),
                  top: int = 10) -> dict:
    """Focused diagnostic for the failing rectangular case (default 1000x1500x700).

    Uses the SAME deterministic inputs (make_inputs + the case seed) and the
    SAME fixed tolerances (ATOL, RTOL). Does NOT loosen tolerances, change the
    kernel, or alter input generation. The elementwise rule evaluated is the
    identical one used by run_correctness:

        violation  <=>  |c_triton - c_ref| > atol + rtol*|c_ref|   (in FP32)

    Reports, for the Triton-vs-FP16-reference comparison:
      - count and percentage of violating elements
      - maximum EXCESS error above the allowed per-element tolerance
      - the worst `top` mismatching coordinates with Triton / FP16-ref /
        FP32-ref values, absolute diff, and the per-element allowed tolerance
      - whether violations cluster near matrix boundaries vs are distributed
    And separately compares the PyTorch FP16 reference against the FP32
    reference under the SAME tolerance rule (how many elements the references
    themselves disagree on).
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Run this in Google Colab with a Tesla T4 GPU."
        )

    # Same deterministic inputs as the correctness suite for this case.
    a, b = make_inputs(m, k, n, seed=seed)
    r16 = ref_fp16(a, b)
    r32 = ref_fp32(a, b)

    c = torch.empty((m, n), dtype=torch.float16, device="cuda")
    matmul(a, b, config, out=c)
    torch.cuda.synchronize()

    # Everything evaluated in FP32.
    cf = c.float()
    r16f = r16.float()
    r32f = r32.float()

    # ---- Triton vs FP16 reference (the primary pass/fail comparison) -------
    allowed = ATOL + RTOL * r16f.abs()          # per-element allowed tolerance
    abs_diff = (cf - r16f).abs()
    violation = abs_diff > allowed              # boolean mask of failures
    excess = (abs_diff - allowed)               # how far over the line (can be <0)

    total = violation.numel()
    n_viol = int(violation.sum().item())
    pct_viol = 100.0 * n_viol / total
    max_excess = float(excess.max().item())     # max amount ABOVE tolerance

    # Worst `top` violating coordinates by absolute difference.
    worst = []
    if n_viol > 0:
        flat_excess = excess.flatten()
        # Rank by excess so the "most over tolerance" show first.
        k_top = min(top, n_viol)
        vals, idxs = torch.topk(flat_excess, k_top)
        for idx in idxs.tolist():
            i = idx // n
            j = idx % n
            worst.append({
                "row": i, "col": j,
                "triton": float(cf[i, j].item()),
                "ref_fp16": float(r16f[i, j].item()),
                "ref_fp32": float(r32f[i, j].item()),
                "abs_diff_vs_fp16": float(abs_diff[i, j].item()),
                "allowed_tol": float(allowed[i, j].item()),
                "excess": float(excess[i, j].item()),
            })

    # ---- Boundary clustering test -----------------------------------------
    # "Near boundary" = within one default tile of any edge (last partial tile
    # along M or N, or the final K-slice region). We use the kernel's tile
    # sizes so this probes the masked/partial-tile regions specifically.
    bm, bn = config.block_m, config.block_n
    m_last = m - (m % bm) if (m % bm) else m - bm   # start row of last M tile
    n_last = n - (n % bn) if (n % bn) else n - bn   # start col of last N tile
    rows = torch.arange(m, device="cuda").view(m, 1)
    cols = torch.arange(n, device="cuda").view(1, n)
    near_boundary = (rows >= m_last) | (cols >= n_last)
    near_boundary = near_boundary.expand(m, n)

    viol_near = int((violation & near_boundary).sum().item())
    viol_interior = n_viol - viol_near
    boundary_cells = int(near_boundary.sum().item())
    interior_cells = total - boundary_cells
    # Densities let us compare clustering fairly despite different region sizes.
    dens_near = (viol_near / boundary_cells) if boundary_cells else 0.0
    dens_interior = (viol_interior / interior_cells) if interior_cells else 0.0

    # ---- FP16 reference vs FP32 reference under the SAME rule --------------
    allowed_ref = ATOL + RTOL * r32f.abs()
    ref_abs_diff = (r16f - r32f).abs()
    ref_violation = ref_abs_diff > allowed_ref
    n_ref_viol = int(ref_violation.sum().item())
    pct_ref_viol = 100.0 * n_ref_viol / total
    ref_max_excess = float((ref_abs_diff - allowed_ref).max().item())

    # How many of the Triton-vs-FP16 violations coincide with places where the
    # two references themselves already disagree beyond tolerance.
    overlap = int((violation & ref_violation).sum().item())

    return {
        "shape": (m, n, k),
        "seed": seed,
        "config": (config.block_m, config.block_n, config.block_k,
                   config.num_warps, config.num_stages),
        "atol": ATOL, "rtol": RTOL,
        "total_elements": total,
        # Triton vs FP16 reference
        "n_violations": n_viol,
        "pct_violations": pct_viol,
        "max_excess_over_tol": max_excess,
        "worst": worst,
        # boundary clustering
        "viol_near_boundary": viol_near,
        "viol_interior": viol_interior,
        "boundary_cells": boundary_cells,
        "interior_cells": interior_cells,
        "violation_density_near_boundary": dens_near,
        "violation_density_interior": dens_interior,
        # FP16 ref vs FP32 ref
        "ref16_vs_ref32_violations": n_ref_viol,
        "ref16_vs_ref32_pct": pct_ref_viol,
        "ref16_vs_ref32_max_excess": ref_max_excess,
        "triton_viol_overlapping_ref_disagreement": overlap,
    }


def print_diagnosis(d: dict) -> None:
    """Pretty-print a diagnose_case() result."""
    print("Rectangular-case diagnostic")
    print(f"  shape (M,N,K)        : {d['shape']}")
    print(f"  seed                 : {d['seed']}")
    print(f"  config (BM,BN,BK,W,S): {d['config']}")
    print(f"  tolerance (atol,rtol): ({d['atol']}, {d['rtol']})")
    print(f"  total elements       : {d['total_elements']}")
    print()
    print("  -- Triton vs FP16 reference (primary pass/fail) --")
    print(f"  violating elements   : {d['n_violations']} "
          f"({d['pct_violations']:.4f}%)")
    print(f"  max excess over tol  : {d['max_excess_over_tol']:.6e}")
    print()
    print("  -- boundary clustering --")
    print(f"  viol near boundary   : {d['viol_near_boundary']} "
          f"(density {d['violation_density_near_boundary']:.3e} "
          f"over {d['boundary_cells']} cells)")
    print(f"  viol interior        : {d['viol_interior']} "
          f"(density {d['violation_density_interior']:.3e} "
          f"over {d['interior_cells']} cells)")
    print()
    print("  -- FP16 reference vs FP32 reference (same rule) --")
    print(f"  ref16 vs ref32 viol  : {d['ref16_vs_ref32_violations']} "
          f"({d['ref16_vs_ref32_pct']:.4f}%)")
    print(f"  ref16 vs ref32 max excess : {d['ref16_vs_ref32_max_excess']:.6e}")
    print(f"  triton viols that overlap ref disagreement : "
          f"{d['triton_viol_overlapping_ref_disagreement']}")
    print()
    print(f"  -- worst {len(d['worst'])} mismatching coordinates --")
    hdr = ("    (row,col)      triton        ref_fp16      ref_fp32      "
           "abs_diff      allowed_tol")
    print(hdr)
    for w in d["worst"]:
        print("    (%4d,%4d)  % .6e  % .6e  % .6e  %.6e  %.6e" % (
            w["row"], w["col"], w["triton"], w["ref_fp16"], w["ref_fp32"],
            w["abs_diff_vs_fp16"], w["allowed_tol"]))


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
