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


# ----------------------------------------------------------------------------
# Phase 3b: methodology review (repeatability, median-vs-p50, resources, clocks)
# ----------------------------------------------------------------------------
# These helpers are ADDITIVE. They do not change run_benchmarks or overwrite the
# first experimental run (results/benchmarks.csv). They write to separate files
# so the original measurements are preserved.

def gpu_state() -> dict:
    """Best-effort snapshot of GPU clock + temperature + memory.

    Used to check whether configurations are inadvertently measured under
    different GPU conditions (clock changes / thermal throttling). Every field
    degrades gracefully to None if the installed torch/driver does not expose
    it, so this never raises on the T4 runtime.
    """
    state = {
        "sm_clock_mhz": None,
        "mem_clock_mhz": None,
        "temperature_c": None,
        "power_w": None,
        "mem_allocated_mb": None,
        "mem_reserved_mb": None,
    }
    if not torch.cuda.is_available():
        return state
    try:
        state["mem_allocated_mb"] = torch.cuda.memory_allocated() / 1e6
        state["mem_reserved_mb"] = torch.cuda.memory_reserved() / 1e6
    except Exception:
        pass
    # clock_rate / temperature / power_draw are available on newer torch builds.
    for key, fn_name, scale in [
        ("sm_clock_mhz", "clock_rate", 1.0),
        ("temperature_c", "temperature", 1.0),
        ("power_w", "power_draw", 1e-3),
    ]:
        try:
            fn = getattr(torch.cuda, fn_name, None)
            if fn is not None:
                state[key] = fn(0) * scale
        except Exception:
            pass
    return state


def kernel_resources(m: int = 1024, n: int = 1024, k: int = 1024,
                     config: MatmulConfig = MatmulConfig(64, 64, 32, 4, 2),
                     seed: int = 0) -> dict:
    """Best-effort extraction of compiled-kernel resource metadata.

    Launches the kernel once (to trigger compilation) and inspects the Triton
    JIT cache for register count, shared-memory bytes, and spill count where the
    installed Triton version exposes them. Reports raw values ONLY -- it does
    NOT infer that the larger-tile slowdown is caused by spilling. Fields are
    None when unavailable.
    """
    info = {
        "config": (config.block_m, config.block_n, config.block_k,
                   config.num_warps, config.num_stages),
        "n_regs": None,
        "shared_bytes": None,
        "n_spills": None,
        "source": "unavailable",
    }
    if not torch.cuda.is_available():
        return info

    a, b = _make_inputs(m, k, n, seed)
    c = torch.empty((m, n), dtype=torch.float16, device="cuda")
    matmul(a, b, config, out=c)
    torch.cuda.synchronize()

    # Walk the kernel's compiled-cache entries and pull whatever attributes the
    # installed Triton exposes. The structure varies across versions, so this is
    # intentionally defensive.
    try:
        from kernels.matmul import _matmul_kernel  # the @triton.jit function
        cache = getattr(_matmul_kernel, "cache", None)
        compiled = None
        if cache:
            # cache is typically {device: {key: CompiledKernel}}.
            for device_cache in cache.values():
                for entry in device_cache.values():
                    compiled = entry
            if compiled is not None:
                for attr, dst in [("n_regs", "n_regs"),
                                   ("shared", "shared_bytes"),
                                   ("metadata", "metadata"),
                                   ("n_spills", "n_spills")]:
                    val = getattr(compiled, attr, None)
                    if attr == "metadata" and val is not None:
                        info["n_regs"] = getattr(val, "n_regs", info["n_regs"])
                        info["shared_bytes"] = getattr(val, "shared", info["shared_bytes"])
                        info["n_spills"] = getattr(val, "n_spills", info["n_spills"])
                    elif dst in info and val is not None:
                        info[dst] = val
                info["source"] = "triton_jit_cache"
    except Exception as exc:
        info["source"] = f"unavailable: {type(exc).__name__}: {exc}"
    return info


def repeat_compare(sizes=None, configs=None, repeats: int = 5, seed: int = 0):
    """Targeted, repeatable benchmark to confirm config-level speedups.

    Defaults to comparing 64x64x32 with 4 vs 8 warps on 1024^3 and 4096^3, over
    `repeats` independent runs. For each (size, config, repeat) it records BOTH
    the primary median (do_bench return_mode='median') and the quantile-run p50
    (do_bench quantiles=[...]) so the two can be compared directly, plus a GPU
    state snapshot to surface clock/thermal drift. Returns a list of dict rows.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Run this in Google Colab with a Tesla T4 GPU."
        )
    if sizes is None:
        sizes = [
            ShapeCase(1024, 1024, 1024, "1024^3", seed=0),
            ShapeCase(4096, 4096, 4096, "4096^3", seed=0),
        ]
    if configs is None:
        configs = [MatmulConfig(64, 64, 32, 4, 2), MatmulConfig(64, 64, 32, 8, 2)]

    rows = []
    for case in sizes:
        m, n, k = case.m, case.n, case.k
        for cfg in configs:
            for r in range(repeats):
                a, b = _make_inputs(m, k, n, case.seed)
                c = torch.empty((m, n), dtype=torch.float16, device="cuda")

                def run():
                    matmul(a, b, cfg, out=c)

                run()  # exclude compilation
                torch.cuda.synchronize()
                st_before = gpu_state()
                primary = bench_median(run)
                _, p50, _ = bench_quantiles(run)
                st_after = gpu_state()

                rows.append({
                    "label": case.label, "m": m, "n": n, "k": k,
                    "block_m": cfg.block_m, "block_n": cfg.block_n,
                    "block_k": cfg.block_k, "num_warps": cfg.num_warps,
                    "num_stages": cfg.num_stages, "repeat": r,
                    "primary_median_ms": primary,
                    "quantile_p50_ms": p50,
                    "median_minus_p50_ms": primary - p50,
                    "tflops_primary": tflops(m, n, k, primary),
                    "sm_clock_mhz_before": st_before["sm_clock_mhz"],
                    "sm_clock_mhz_after": st_after["sm_clock_mhz"],
                    "temperature_c_before": st_before["temperature_c"],
                    "temperature_c_after": st_after["temperature_c"],
                })
                torch.cuda.empty_cache()
    return rows


_REPEAT_FIELDS = [
    "label", "m", "n", "k", "block_m", "block_n", "block_k", "num_warps",
    "num_stages", "repeat", "primary_median_ms", "quantile_p50_ms",
    "median_minus_p50_ms", "tflops_primary",
    "sm_clock_mhz_before", "sm_clock_mhz_after",
    "temperature_c_before", "temperature_c_after",
]


def write_repeat_csv(rows, path="results/benchmarks_repeat.csv"):
    """Write the repeatability rows to a SEPARATE CSV (preserves run 1)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_REPEAT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in _REPEAT_FIELDS})
    return path


def summarize_repeats(rows) -> list:
    """Aggregate repeat rows into per (size, config) mean/min/max/std."""
    import statistics
    groups = {}
    for row in rows:
        key = (row["label"], row["num_warps"])
        groups.setdefault(key, []).append(row["primary_median_ms"])
    summary = []
    for (label, warps), vals in groups.items():
        summary.append({
            "label": label, "num_warps": warps, "repeats": len(vals),
            "mean_ms": statistics.fmean(vals),
            "min_ms": min(vals), "max_ms": max(vals),
            "std_ms": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        })
    return summary


def print_repeat_summary(rows) -> None:
    """Print per-run rows + an aggregated mean/min/max/std per (size, warps)."""
    print("Per-run (primary median vs quantile p50):")
    print("  label        warps repeat primary_ms  p50_ms     diff_ms    "
          "sm_clk_before sm_clk_after")
    for r in rows:
        print("  %-12s %-5d %-6d %-11.4f %-10.4f %-10.4f %-13s %-s" % (
            r["label"], r["num_warps"], r["repeat"],
            r["primary_median_ms"], r["quantile_p50_ms"], r["median_minus_p50_ms"],
            str(r["sm_clock_mhz_before"]), str(r["sm_clock_mhz_after"])))
    print("\nAggregated across repeats:")
    print("  label        warps repeats mean_ms    min_ms     max_ms     std_ms")
    for s in summarize_repeats(rows):
        print("  %-12s %-5d %-7d %-10.4f %-10.4f %-10.4f %-10.4f" % (
            s["label"], s["num_warps"], s["repeats"],
            s["mean_ms"], s["min_ms"], s["max_ms"], s["std_ms"]))


# ----------------------------------------------------------------------------
# Phase 5: plotting (reads measured records only)
# ----------------------------------------------------------------------------
def _read_benchmarks_csv(path="results/benchmarks.csv"):
    """Load benchmark rows from CSV for plotting (measured data only)."""
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def plot_latency(csv_path="results/benchmarks.csv",
                 out_path="results/latency_comparison.png",
                 log_y: bool = True):
    """Plot measured median latency (ms) per workload for the fastest custom
    config vs PyTorch. Reads only observed CSV rows; nothing is fabricated.

    `log_y=True` uses a logarithmic y-axis so the wide latency range (512^3 to
    4096^3) is legible. The original measurements are unchanged; this only
    changes the axis scale."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _read_benchmarks_csv(csv_path)
    labels = []
    for r in rows:
        if r["label"] not in labels:
            labels.append(r["label"])

    def best_custom(label):
        cand = [r for r in rows if r["label"] == label
                and r["impl"] == "triton_custom" and r["status"] == "ok"
                and r["latency_ms_primary_median"] not in ("", "nan")]
        if not cand:
            return None
        return min(float(r["latency_ms_primary_median"]) for r in cand)

    def torch_lat(label):
        cand = [r for r in rows if r["label"] == label
                and r["impl"] == "torch" and r["status"] == "ok"
                and r["latency_ms_primary_median"] not in ("", "nan")]
        if not cand:
            return None
        return min(float(r["latency_ms_primary_median"]) for r in cand)

    custom = [best_custom(l) for l in labels]
    torchv = [torch_lat(l) for l in labels]

    x = range(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([i - w / 2 for i in x], [c or 0 for c in custom], w,
           label="Triton (fastest config)")
    ax.bar([i + w / 2 for i in x], [t or 0 for t in torchv], w, label="PyTorch")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ylabel = "median latency (ms)  — lower is better"
    if log_y:
        ax.set_yscale("log")
        ylabel += "  [log scale]"
    ax.set_ylabel(ylabel)
    ax.set_title("Measured median latency: fastest Triton config vs PyTorch (T4)")
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_config_speedup(csv_path="results/benchmarks.csv",
                        out_path="results/config_speedup.png",
                        baseline_warps: int = 4, compare_warps: int = 8,
                        block_m: int = 64, block_n: int = 64):
    """Configuration-level speedup plot: for the 64x64 tile, how much faster the
    `compare_warps` config is than the `baseline_warps` config, per workload.

    This is the CUSTOM-vs-CUSTOM comparison (my 8-warp config relative to my
    4-warp baseline) and is DISTINCT from any comparison against PyTorch. Reads
    only measured rows; a workload is plotted only if both configs have an 'ok'
    median. Nothing is fabricated.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _read_benchmarks_csv(csv_path)
    labels = []
    for r in rows:
        if r["label"] not in labels:
            labels.append(r["label"])

    def median_for(label, warps):
        for r in rows:
            if (r["label"] == label and r["impl"] == "triton_custom"
                    and r["status"] == "ok"
                    and int(r["block_m"]) == block_m
                    and int(r["block_n"]) == block_n
                    and int(r["num_warps"]) == warps
                    and r["latency_ms_primary_median"] not in ("", "nan")):
                return float(r["latency_ms_primary_median"])
        return None

    plot_labels, speedups = [], []
    for lab in labels:
        base = median_for(lab, baseline_warps)
        comp = median_for(lab, compare_warps)
        if base and comp and comp > 0:
            plot_labels.append(lab)
            speedups.append(base / comp)   # >1 means compare_warps is faster

    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(plot_labels))
    ax.bar(list(x), speedups, 0.6, color="#4C72B0")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1,
               label="parity (no change)")
    for i, s in enumerate(speedups):
        ax.text(i, s, f"{s:.2f}x", ha="center", va="bottom")
    ax.set_xticks(list(x))
    ax.set_xticklabels(plot_labels, rotation=30, ha="right")
    ax.set_ylabel(f"speedup of {compare_warps} warps vs {baseline_warps} warps "
                  f"(custom {block_m}x{block_n})  — higher is better")
    ax.set_title(f"Config-level speedup: {block_m}x{block_n} "
                 f"{compare_warps}-warp vs {baseline_warps}-warp (T4)\n"
                 f"custom-vs-custom; NOT a comparison against PyTorch")
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_throughput(csv_path="results/benchmarks.csv",
                    out_path="results/throughput_comparison.png"):
    """Plot measured TFLOPS per workload for the fastest custom config vs
    PyTorch. Reads only observed CSV rows."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _read_benchmarks_csv(csv_path)
    labels = []
    for r in rows:
        if r["label"] not in labels:
            labels.append(r["label"])

    def best_custom_tflops(label):
        cand = [r for r in rows if r["label"] == label
                and r["impl"] == "triton_custom" and r["status"] == "ok"
                and r["tflops"] not in ("", "nan")]
        if not cand:
            return None
        return max(float(r["tflops"]) for r in cand)

    def torch_tflops(label):
        cand = [r for r in rows if r["label"] == label
                and r["impl"] == "torch" and r["status"] == "ok"
                and r["tflops"] not in ("", "nan")]
        if not cand:
            return None
        return max(float(r["tflops"]) for r in cand)

    custom = [best_custom_tflops(l) for l in labels]
    torchv = [torch_tflops(l) for l in labels]

    x = range(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([i - w / 2 for i in x], [c or 0 for c in custom], w,
           label="Triton (fastest config)")
    ax.bar([i + w / 2 for i in x], [t or 0 for t in torchv], w, label="PyTorch")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("effective TFLOPS  — higher is better")
    ax.set_title("Measured throughput: fastest Triton config vs PyTorch (T4)")
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
