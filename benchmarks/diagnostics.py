"""Isolated hardware + numerical diagnostics for the Triton matmul kernel.

This module is READ-ONLY with respect to the project: it does NOT modify
kernels/matmul.py, the public matmul() wrapper, the correctness tests, or any
committed result file. It invokes the kernel's @triton.jit function
`_matmul_kernel[grid](...)` directly so it can capture the compiled-kernel
object and inspect version-specific metadata, PTX, and TTGIR.

All diagnostic artifacts are written under results/diagnostics/ so the original
results/ files are preserved.

Nothing here concludes whether Tensor Cores are used, whether spilling occurs,
or what causes the performance gap. It only gathers evidence; conclusions wait
until the actual Colab output is reviewed.

Two phases:
  * Hardware diagnostics (Phase 2): compiled-object attributes, register/spill/
    shared-memory metadata, PTX instruction inspection (mma.sync / HMMA / FMA),
    TTGIR dot-lowering inspection, with a cuobjdump fallback.
  * Numerical diagnostics (Phase 3): compare custom Triton output vs several
    PyTorch references (FP16, FP16 with reduced-precision reduction disabled,
    FP64) on the 1000x1500x700 workload, plus a torch.equal cross-config check.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess

import torch
import triton
import triton.language as tl

# Import the kernel's @triton.jit function and config DIRECTLY. We never call or
# modify the public matmul() wrapper.
from kernels.matmul import _matmul_kernel, MatmulConfig


DIAG_DIR_DEFAULT = "results/diagnostics"

# The two configurations under investigation (both already in the project).
CONFIG_64 = MatmulConfig(block_m=64, block_n=64, block_k=32, num_warps=8, num_stages=2)
CONFIG_128 = MatmulConfig(block_m=128, block_n=128, block_k=32, num_warps=4, num_stages=2)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _make_inputs(m, k, n, seed=0):
    """Deterministic FP16 inputs — same scheme as the correctness suite."""
    torch.manual_seed(seed)
    a = torch.randn((m, k), dtype=torch.float16, device="cuda")
    b = torch.randn((k, n), dtype=torch.float16, device="cuda")
    return a, b


# ----------------------------------------------------------------------------
# Phase 2: hardware diagnostics
# ----------------------------------------------------------------------------
def compile_and_capture(config: MatmulConfig, m: int = 1024, n: int = 1024,
                        k: int = 1024, seed: int = 0):
    """Launch _matmul_kernel directly and return the compiled-kernel handle.

    Triton's indexed-launch (`kernel[grid](...)`) returns the compiled kernel
    object for the launched specialization. We replicate the exact launch
    signature used by matmul() WITHOUT calling matmul() itself.
    """
    a, b = _make_inputs(m, k, n, seed)
    c = torch.empty((m, n), dtype=torch.float16, device="cuda")
    grid = (triton.cdiv(m, config.block_m) * triton.cdiv(n, config.block_n),)

    compiled = _matmul_kernel[grid](
        a, b, c,
        m, n, k,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=config.block_m,
        BLOCK_N=config.block_n,
        BLOCK_K=config.block_k,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )
    torch.cuda.synchronize()
    return compiled


def _safe_getattr(obj, name):
    try:
        val = getattr(obj, name)
    except Exception as exc:  # some attributes raise on access
        return f"<error accessing: {type(exc).__name__}: {exc}>"
    return val


def inspect_compiled(compiled) -> dict:
    """Report the compiled object's PUBLIC attributes and resource metadata.

    Lists `dir(compiled)` (non-dunder) FIRST so we can see what the installed
    Triton actually exposes before reaching for version-specific fields. Then
    defensively pulls n_regs / n_spills / shared and any `metadata` sub-fields.
    Everything is reported as-is; unavailable fields are marked unavailable.
    """
    public_attrs = [a for a in dir(compiled) if not a.startswith("__")]

    report = {
        "type": type(compiled).__name__,
        "public_attributes": public_attrs,
        "n_regs": None,
        "n_spills": None,
        "shared_bytes": None,
        "num_warps": None,
        "num_stages": None,
        "metadata_fields": None,
        "notes": [],
    }

    # Direct attributes (older/newer Triton expose some of these on the handle).
    for attr, dst in [("n_regs", "n_regs"), ("n_spills", "n_spills"),
                      ("shared", "shared_bytes"), ("num_warps", "num_warps"),
                      ("num_stages", "num_stages")]:
        if hasattr(compiled, attr):
            report[dst] = _safe_getattr(compiled, attr)

    # Many Triton versions carry these inside a `metadata` namedtuple/dataclass.
    meta = _safe_getattr(compiled, "metadata") if hasattr(compiled, "metadata") else None
    if meta is not None and not isinstance(meta, str):
        meta_fields = {}
        for name in dir(meta):
            if name.startswith("__"):
                continue
            v = _safe_getattr(meta, name)
            # Keep only simple scalar-ish values in the report.
            if isinstance(v, (int, float, str, bool)) or v is None:
                meta_fields[name] = v
        report["metadata_fields"] = meta_fields
        for src, dst in [("n_regs", "n_regs"), ("n_spills", "n_spills"),
                         ("shared", "shared_bytes"), ("num_warps", "num_warps"),
                         ("num_stages", "num_stages")]:
            if report[dst] is None and src in meta_fields:
                report[dst] = meta_fields[src]

    if report["n_regs"] is None and report["metadata_fields"] is None:
        report["notes"].append(
            "No register/metadata fields found on the compiled handle; "
            "see public_attributes and the cuobjdump fallback."
        )
    return report


def dump_asm(compiled, kinds=("ptx", "ttgir", "ttir", "llir")) -> dict:
    """Return available IR/ASM strings from the compiled object.

    Triton typically stores these in `compiled.asm` (a dict keyed by 'ptx',
    'ttgir', 'ttir', 'llir', 'cubin', ...). We copy out the requested text
    kinds; missing kinds are marked unavailable. The raw text is returned so the
    caller can save it to files for inspection.
    """
    out = {}
    asm = _safe_getattr(compiled, "asm") if hasattr(compiled, "asm") else None
    if isinstance(asm, dict):
        out["available_kinds"] = sorted(asm.keys())
        for kind in kinds:
            out[kind] = asm.get(kind, None)
    else:
        out["available_kinds"] = []
        for kind in kinds:
            out[kind] = None
        out["note"] = "compiled.asm not available as a dict on this build"
    return out


def scan_ptx(ptx: str) -> dict:
    """Count MMA vs ordinary FP instructions in PTX.

    Reports raw substring counts for mma.sync / wmma / hmma (Tensor Core MMA)
    and fma / mad (ordinary FP). A COUNT ALONE IS NOT CONCLUSIVE — Tensor Core
    usage must be judged from the actual instruction forms, not just presence of
    a substring. We therefore also capture a few example lines for review.
    """
    if not ptx:
        return {"available": False}
    lines = ptx.splitlines()

    def count(sub):
        return sum(1 for ln in lines if sub in ln)

    def examples(sub, limit=5):
        ex = [ln.strip() for ln in lines if sub in ln]
        return ex[:limit]

    return {
        "available": True,
        "n_lines": len(lines),
        "count_mma_sync": count("mma.sync"),
        "count_wmma": count("wmma"),
        "count_hmma": count("hmma"),
        "count_fma": count("fma."),
        "count_mad": count("mad."),
        "examples_mma_sync": examples("mma.sync"),
        "examples_fma": examples("fma."),
        "caveat": ("Substring counts are indicative only; confirm Tensor Core "
                   "use from actual instruction forms, not presence alone."),
    }


def scan_ttgir(ttgir: str) -> dict:
    """Look for dot lowering / MMA vs blocked layouts in TTGIR.

    Reports raw counts for `tt.dot`, `mma`, and `blocked` layout mentions, with
    example lines. As with PTX, counts are indicative only, not conclusive.
    """
    if not ttgir:
        return {"available": False}
    lines = ttgir.splitlines()

    def count(sub):
        return sum(1 for ln in lines if sub in ln)

    def examples(sub, limit=5):
        return [ln.strip() for ln in lines if sub in ln][:limit]

    return {
        "available": True,
        "n_lines": len(lines),
        "count_tt_dot": count("tt.dot"),
        "count_mma": count("mma"),
        "count_blocked": count("blocked"),
        "examples_mma": examples("mma"),
        "examples_dot": examples("tt.dot"),
        "caveat": "A simple string count is not conclusive evidence by itself.",
    }


def cuobjdump_fallback(compiled, out_dir: str) -> dict:
    """If a cubin is available and cuobjdump exists, dump -res-usage and -sass.

    Writes the cubin and the cuobjdump text outputs under out_dir. Reports
    whether each step succeeded; everything degrades gracefully to 'unavailable'
    rather than guessing.
    """
    result = {"cuobjdump_available": False, "cubin_written": False,
              "res_usage": None, "sass_head": None, "notes": []}

    cuobjdump = shutil.which("cuobjdump")
    if cuobjdump is None:
        result["notes"].append("cuobjdump not found on PATH")
        return result
    result["cuobjdump_available"] = True

    asm = _safe_getattr(compiled, "asm") if hasattr(compiled, "asm") else None
    cubin = asm.get("cubin") if isinstance(asm, dict) else None
    if cubin is None:
        result["notes"].append("no cubin in compiled.asm")
        return result

    _ensure_dir(out_dir)
    cubin_path = os.path.join(out_dir, "kernel.cubin")
    mode = "wb" if isinstance(cubin, (bytes, bytearray)) else "w"
    with open(cubin_path, mode) as f:
        f.write(cubin)
    result["cubin_written"] = True

    for flag, key in [("--dump-resource-usage", "res_usage"), ("-sass", "sass_head")]:
        try:
            proc = subprocess.run([cuobjdump, flag, cubin_path],
                                  capture_output=True, text=True, timeout=120)
            text = proc.stdout or proc.stderr
            result[key] = text if key == "res_usage" else "\n".join(
                text.splitlines()[:80])
        except Exception as exc:
            result["notes"].append(f"{flag} failed: {type(exc).__name__}: {exc}")
    return result


def run_hardware_diagnostics(out_dir: str = DIAG_DIR_DEFAULT) -> dict:
    """Full Phase 2 pass for both configurations under investigation.

    Writes per-config PTX/TTGIR text + a JSON summary under out_dir, and returns
    the summary dict. Makes NO conclusions about Tensor Core use or bottlenecks.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Run in Colab on the Tesla T4.")
    _ensure_dir(out_dir)

    summary = {"env": {
        "gpu": torch.cuda.get_device_name(0),
        "cc": ".".join(map(str, torch.cuda.get_device_capability(0))),
        "torch": torch.__version__, "triton": triton.__version__,
        "cuda": str(torch.version.cuda),
    }, "configs": {}}

    for tag, cfg in [("64x64x32_w8_s2", CONFIG_64), ("128x128x32_w4_s2", CONFIG_128)]:
        compiled = compile_and_capture(cfg)
        meta = inspect_compiled(compiled)
        asm = dump_asm(compiled)

        # Save raw IR text for manual inspection.
        for kind in ("ptx", "ttgir", "ttir", "llir"):
            text = asm.get(kind)
            if isinstance(text, str) and text:
                with open(os.path.join(out_dir, f"{tag}.{kind}"), "w") as f:
                    f.write(text)

        ptx_scan = scan_ptx(asm.get("ptx") or "")
        ttgir_scan = scan_ttgir(asm.get("ttgir") or "")
        cuobj = cuobjdump_fallback(compiled, os.path.join(out_dir, tag))

        summary["configs"][tag] = {
            "config": (cfg.block_m, cfg.block_n, cfg.block_k,
                       cfg.num_warps, cfg.num_stages),
            "compiled": meta,
            "asm_available_kinds": asm.get("available_kinds"),
            "ptx_scan": ptx_scan,
            "ttgir_scan": ttgir_scan,
            "cuobjdump": {k: (v if k != "res_usage" else ("<saved>" if v else None))
                          for k, v in cuobj.items()},
        }

    with open(os.path.join(out_dir, "hardware_diagnostics.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary


def print_hardware_summary(summary: dict) -> None:
    """Human-readable preview of the hardware diagnostics."""
    print("Hardware diagnostics")
    print("  env:", summary["env"])
    for tag, c in summary["configs"].items():
        print(f"\n  === {tag}  config={c['config']} ===")
        comp = c["compiled"]
        print("  compiled type   :", comp["type"])
        print("  n_regs          :", comp["n_regs"])
        print("  n_spills        :", comp["n_spills"])
        print("  shared_bytes    :", comp["shared_bytes"])
        print("  num_warps       :", comp["num_warps"])
        print("  metadata_fields :", comp["metadata_fields"])
        print("  public_attrs    :", comp["public_attributes"])
        print("  asm kinds       :", c["asm_available_kinds"])
        print("  PTX scan        :", {k: c["ptx_scan"].get(k) for k in
              ("available", "count_mma_sync", "count_wmma", "count_hmma",
               "count_fma", "count_mad")})
        print("  TTGIR scan      :", {k: c["ttgir_scan"].get(k) for k in
              ("available", "count_tt_dot", "count_mma", "count_blocked")})
        print("  cuobjdump       :", c["cuobjdump"])
    print("\n  NOTE: no conclusion about Tensor Core use or bottlenecks is drawn "
          "here; review the saved PTX/TTGIR/SASS before interpreting.")


# ----------------------------------------------------------------------------
# Phase 3: numerical diagnostics (1000x1500x700, original input scheme)
# ----------------------------------------------------------------------------
# Compares the custom Triton output against several PyTorch references. Writes to
# results/diagnostics/; NEVER overwrites results/correctness.csv or reinterprets
# its pass/fail flags.

_RECT = (1000, 1500, 700)   # M, N, K
_RECT_SEED = 2              # same seed the correctness suite used for this shape
_ATOL = 1e-2
_RTOL = 1e-2


def _launch_direct(a, b, config: MatmulConfig):
    """Run the kernel via a direct _matmul_kernel launch (no matmul() wrapper)."""
    m, k = a.shape
    _, n = b.shape
    c = torch.empty((m, n), dtype=torch.float16, device="cuda")
    grid = (triton.cdiv(m, config.block_m) * triton.cdiv(n, config.block_n),)
    _matmul_kernel[grid](
        a, b, c, m, n, k,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=config.block_m, BLOCK_N=config.block_n, BLOCK_K=config.block_k,
        num_warps=config.num_warps, num_stages=config.num_stages,
    )
    torch.cuda.synchronize()
    return c


def _fp16_ulp_error(c_test_fp16: torch.Tensor, ref_rounded_fp16: torch.Tensor) -> dict:
    """Approximate FP16-ULP error between two FP16 tensors.

    Reinterprets the FP16 bit patterns as int16 ordinal values (sign-magnitude
    mapped to a monotonic ordering) and reports the max/mean absolute ordinal
    distance. This is a standard ULP-distance approximation and is only
    meaningful where both operands are finite.
    """
    a = c_test_fp16.contiguous().view(torch.int16).to(torch.int32)
    b = ref_rounded_fp16.contiguous().view(torch.int16).to(torch.int32)

    # Map sign-magnitude to a monotonic ordering so adjacent floats differ by 1.
    def mono(x):
        return torch.where(x < 0, torch.tensor(-2147483648, device=x.device) - x, x)

    da = mono(a)
    db = mono(b)
    ulp = (da - db).abs()
    return {
        "max_ulp": int(ulp.max().item()),
        "mean_ulp": float(ulp.float().mean().item()),
        "frac_within_1_ulp": float((ulp <= 1).float().mean().item()),
        "frac_within_2_ulp": float((ulp <= 2).float().mean().item()),
    }


def _error_stats(c_test: torch.Tensor, c_ref: torch.Tensor) -> dict:
    """Max abs error, relative-error stats, and tolerance-violation count, in FP32."""
    t = c_test.float()
    r = c_ref.float()
    diff = (t - r).abs()
    allowed = _ATOL + _RTOL * r.abs()
    viol = diff > allowed
    denom = r.abs().clamp_min(1e-3)
    rel = diff / denom
    return {
        "max_abs_error": float(diff.max().item()),
        "mean_abs_error": float(diff.mean().item()),
        "rel_error_max_robust": float(rel.max().item()),
        "rel_error_mean_robust": float(rel.mean().item()),
        "rel_error_p99_robust": float(torch.quantile(rel.flatten(), 0.99).item()),
        "tol_violations": int(viol.sum().item()),
        "tol_violation_pct": float(100.0 * viol.sum().item() / viol.numel()),
    }


def run_numerical_diagnostics(config: MatmulConfig = CONFIG_64,
                              out_dir: str = DIAG_DIR_DEFAULT) -> dict:
    """Compare custom Triton output vs PyTorch FP16 / FP16-no-reduced / FP64.

    References:
      * ref_fp16   = torch.matmul(a, b)                      (default FP16)
      * ref_fp16_nr= same, with allow_fp16_reduced_precision_reduction=False
                     (if supported; restored in a finally block)
      * ref_fp64   = torch.matmul(a.double(), b.double())    (true-math anchor)

    For each implementation, compares against the FP64 reference ROUNDED TO FP16
    (round-to-nearest-even via .half()), and reports how many Triton elements
    EXACTLY match that rounded FP64 value. Writes a JSON summary; does not touch
    correctness.csv.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Run in Colab on the Tesla T4.")
    _ensure_dir(out_dir)

    m, n, k = _RECT
    a, b = _make_inputs(m, k, n, seed=_RECT_SEED)

    # Custom Triton output (direct launch).
    c_triton = _launch_direct(a, b, config)

    # PyTorch FP16 reference (default reduced-precision reduction behavior).
    ref_fp16 = torch.matmul(a, b)

    # PyTorch FP16 with reduced-precision reduction DISABLED, if supported.
    ref_fp16_nr = None
    nr_supported = hasattr(torch.backends.cuda.matmul,
                           "allow_fp16_reduced_precision_reduction")
    if nr_supported:
        prev = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        try:
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
            ref_fp16_nr = torch.matmul(a, b)
        finally:
            # Always restore the backend setting.
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = prev

    # FP64 true-math anchor, and its round-to-FP16 image.
    ref_fp64 = torch.matmul(a.double(), b.double())
    ref_fp64_as_fp16 = ref_fp64.half()

    # Exact matches: Triton FP16 elements equal to rounded-FP64 FP16 bitwise.
    exact_matches = int((c_triton == ref_fp64_as_fp16).sum().item())
    total = c_triton.numel()

    summary = {
        "config": (config.block_m, config.block_n, config.block_k,
                   config.num_warps, config.num_stages),
        "shape": {"m": m, "n": n, "k": k, "seed": _RECT_SEED},
        "reduced_precision_reduction_toggle_supported": nr_supported,
        "vs_fp16_ref": _error_stats(c_triton, ref_fp16),
        "vs_fp64_ref_rounded_fp16": _error_stats(c_triton, ref_fp64_as_fp16),
        "triton_vs_fp64rounded_ulp": _fp16_ulp_error(c_triton, ref_fp64_as_fp16),
        "triton_exact_matches_vs_fp64rounded": exact_matches,
        "triton_exact_match_pct": float(100.0 * exact_matches / total),
        "total_elements": total,
    }

    # Where supported, also quantify how the two FP16 references differ from FP64.
    summary["fp16ref_vs_fp64rounded"] = _error_stats(ref_fp16, ref_fp64_as_fp16)
    if ref_fp16_nr is not None:
        summary["fp16ref_noreduce_vs_fp64rounded"] = _error_stats(
            ref_fp16_nr, ref_fp64_as_fp16)
        summary["triton_vs_fp16ref_noreduce"] = _error_stats(c_triton, ref_fp16_nr)

    with open(os.path.join(out_dir, "numerical_diagnostics.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary


def run_cross_config_equality(out_dir: str = DIAG_DIR_DEFAULT) -> dict:
    """torch.equal across the six configs on 1000x1500x700.

    Establishes whether the six configurations produce genuinely identical
    outputs (bitwise) for this workload, using the same deterministic inputs.
    Writes a JSON summary; does not touch correctness.csv.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Run in Colab on the Tesla T4.")
    _ensure_dir(out_dir)

    m, n, k = _RECT
    a, b = _make_inputs(m, k, n, seed=_RECT_SEED)

    configs = [
        MatmulConfig(64, 64, 32, 4, 2), MatmulConfig(64, 64, 32, 8, 2),
        MatmulConfig(128, 128, 32, 4, 2), MatmulConfig(128, 128, 32, 8, 2),
        MatmulConfig(128, 64, 32, 4, 2), MatmulConfig(128, 64, 32, 8, 2),
    ]
    outputs = []
    tags = []
    for cfg in configs:
        outputs.append(_launch_direct(a, b, cfg))
        tags.append(f"{cfg.block_m}x{cfg.block_n}x{cfg.block_k}_w{cfg.num_warps}")

    ref = outputs[0]
    equal_to_first = {tags[i]: bool(torch.equal(ref, outputs[i]))
                      for i in range(len(outputs))}
    all_identical = all(equal_to_first.values())

    summary = {
        "shape": {"m": m, "n": n, "k": k, "seed": _RECT_SEED},
        "reference_config": tags[0],
        "equal_to_reference": equal_to_first,
        "all_six_identical": all_identical,
    }
    with open(os.path.join(out_dir, "cross_config_equality.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary


def print_numerical_summary(num: dict, equal: dict) -> None:
    """Human-readable preview of the numerical diagnostics."""
    print("Numerical diagnostics (1000x1500x700)")
    print("  config                           :", num["config"])
    print("  reduced-precision toggle support :",
          num["reduced_precision_reduction_toggle_supported"])
    print("  Triton vs FP16 ref               :", num["vs_fp16_ref"])
    print("  Triton vs FP64->FP16             :", num["vs_fp64_ref_rounded_fp16"])
    print("  Triton vs FP64->FP16 (ULP)       :", num["triton_vs_fp64rounded_ulp"])
    print("  Triton exact matches vs FP64->FP16: %d / %d (%.4f%%)" % (
        num["triton_exact_matches_vs_fp64rounded"], num["total_elements"],
        num["triton_exact_match_pct"]))
    print("  FP16 ref vs FP64->FP16           :", num["fp16ref_vs_fp64rounded"])
    if "fp16ref_noreduce_vs_fp64rounded" in num:
        print("  FP16 ref (no-reduce) vs FP64->FP16:",
              num["fp16ref_noreduce_vs_fp64rounded"])
    print("\n  Cross-config torch.equal:")
    print("   reference:", equal["reference_config"])
    for tag, eq in equal["equal_to_reference"].items():
        print(f"    {tag:20s} equal={eq}")
    print("   all six identical:", equal["all_six_identical"])
