# Triton GPU Kernel Optimization and Benchmarking

A learning-focused, reproducible study that implements a tiled FP16 matrix
multiplication kernel in Triton, validates it against `torch.matmul`, benchmarks
it against PyTorch, and runs manual configuration-level optimization experiments
on an NVIDIA Tesla T4. The goal is to understand, experimentally, the
engineering principles behind deep-learning kernel libraries such as NVIDIA
cuDNN and FlashInfer — grounded in observed evidence rather than assumption.

## Objective and motivation

Modern matmul libraries hide a lot of engineering behind a single call. This
project rebuilds a small slice of that stack from scratch — a tiled matmul
kernel — and measures it honestly against PyTorch on the same GPU, so the
tradeoffs (tile sizes, warps, precision, memory movement) become concrete and
measurable instead of abstract.

## How tiled GPU matrix multiplication works

For `A[M,K] @ B[K,N] = C[M,N]`, the output is partitioned into `BLOCK_M × BLOCK_N`
tiles. One Triton program computes one output tile:

1. **Program IDs → tile mapping.** A 1-D grid of `cdiv(M,BLOCK_M)·cdiv(N,BLOCK_N)`
   programs is launched; the flat `pid` is split row-major into `(pid_m, pid_n)`.
2. **Pointer arithmetic.** Per-lane offsets `offs_m/offs_n/offs_k` combine with the
   matrix strides to address the A and B sub-tiles.
3. **K-loop with FP32 accumulation.** The program walks K in `BLOCK_K` steps,
   loading masked FP16 sub-tiles and accumulating `tl.dot` products into an FP32
   accumulator.
4. **FP16 output.** After the full reduction, the accumulator is cast to FP16
   **once** and stored with a boundary mask.

Boundary masks on every load/store (`offs < M/N/k_remaining`, `other=0.0`) make
the kernel safe for rectangular and non-multiple-of-tile shapes.

## Kernel execution and memory access

- `kernels/matmul.py` holds the `@triton.jit` kernel and a `matmul()` host
  wrapper. The wrapper validates dtypes/shapes/device, validates any supplied
  preallocated `out` buffer, builds the grid, and launches with `num_warps` and
  `num_stages` passed explicitly (`num_stages` is fixed at 2 for all experiments
  and recorded, never hidden).
- Inputs are FP16, accumulation is FP32, output is FP16. No BF16/FP8/Ampere+
  features are used.

## GPU hardware and software environment

| Component | Value |
|-----------|-------|
| GPU | NVIDIA Tesla T4 |
| Compute capability | 7.5 (Turing) |
| PyTorch | 2.11.0+cu130 |
| CUDA | 13.0 |
| Triton | 3.6.0 |

These were confirmed in Google Colab and are treated as fixed. The project never
reinstalls or downgrades torch/triton.

## Correctness methodology

- Deterministic FP16 inputs (seeded per case) across square, rectangular, small
  non-square, and non-multiple-of-tile shapes, each run against all six
  tile/warp configurations (`num_stages=2`).
- Elementwise rule, evaluated in **FP32** (so the error calculation never uses
  FP16 arithmetic): `abs(c_triton − c_ref) ≤ atol + rtol·abs(c_ref)` with fixed
  `atol = 1e-2`, `rtol = 1e-2`. **Tolerances are never loosened to force a pass.**
- Two references: `ref_fp16 = torch.matmul(a, b)` (primary, pass/fail) and
  `ref_fp32 = torch.matmul(a.float(), b.float())` (diagnostic).

### Measured correctness result (preserved as observed)

- **24 / 30** shape×config combinations passed the **FP16-reference** criterion;
  **6 failed** — all six configs on the `1000×1500×700` case. **0** compilation
  or runtime errors.
- Investigation of the failing case (measured, not assumed): 79 violating
  elements out of 1,500,000 (0.0053%); **79 of 79** overlapped elements where
  PyTorch's own FP16 and FP32 references disagree; violation densities were
  near-identical at boundary (5.76e-05) and interior (5.24e-05).
- A targeted **FP32-reference** check of the full `1000×1500×700` output passed
  with **0 / 1,500,000** violations (max abs error 0.0442).

**Interpretation (evidence-based, not proven):** the FP16-reference failures are
most consistent with a precision artifact of PyTorch's FP16 `torch.matmul` on
near-zero-magnitude outputs — the custom kernel tracks the FP32 (true-math)
result, while the FP16 *reference* is the noisy operand in the relative-error
denominator. We do **not** claim this isolates every source of numerical error,
and we do **not** relabel the 24/30 FP16-reference result.

## Benchmark methodology

- Benchmarks the **identical** operation `C = A @ B` with **preallocated FP16
  output buffers on both sides**: `torch.matmul(a, b, out=c_torch)` and
  `matmul(a, b, config, out=c_triton)`.
- **Primary latency** via `triton.testing.do_bench(..., return_mode="median")`;
  **p20/p50/p80** recorded from a **separate** `do_bench(..., quantiles=[0.2,
  0.5, 0.8])` run. The quantile-run p50 is reported independently and is **not**
  assumed equal to the primary median.
- One-time Triton compilation is excluded via a throwaway launch before timing.
- `TFLOPS = (2·M·N·K) / (latency_seconds · 1e12)`; speedup from measured medians.
- Every size×config is attempted; `OutOfResources` / compile failures / OOM are
  recorded as skipped rows with the actual error message (none occurred in the
  first run).

## Experimental configurations

`num_stages = 2` for all six:

| BLOCK_M | BLOCK_N | BLOCK_K | num_warps | num_stages |
|--------:|--------:|--------:|----------:|-----------:|
| 64 | 64 | 32 | 4 | 2 |
| 64 | 64 | 32 | 8 | 2 |
| 128 | 128 | 32 | 4 | 2 |
| 128 | 128 | 32 | 8 | 2 |
| 128 | 64 | 32 | 4 | 2 |
| 128 | 64 | 32 | 8 | 2 |

Benchmark sizes: `512³`, `1024³`, `2048³`, `4096³`, `1000×1500×700`.

## Measured results (first experimental run)

Observed on the Tesla T4 (see `results/benchmarks.csv` for the full table):

- All 30 Triton configurations executed with **no resource or compilation
  errors**.
- The **64×64×32, 8-warps** configuration was the fastest custom config across
  all five workloads.
- `1024³`: 4→8 warps reduced custom latency from **1.4652 ms → 0.6836 ms**
  (~2.14×).
- `4096³`: 4→8 warps reduced latency from **105.468 ms → 51.476 ms** (~2.05×).
- **All custom configurations remained slower than PyTorch** on every workload.
- The **128×128** configurations showed unexpectedly severe regressions.

Plots: `results/latency_comparison.png`, `results/throughput_comparison.png`
(generated from the measured CSV only).

### Interpretation vs hypothesis (kept separate)

- **Measurement:** 8 warps beat 4 warps for the 64×64 tile on the large
  workloads; PyTorch (cuBLAS-backed) was faster than every custom config.
- **Hypothesis (not yet proven):** the 128×128 regression could relate to
  occupancy / resource pressure on Turing, but this is **not** attributed to
  register spilling without evidence. `kernel_resources()` reports register /
  shared-memory / spill metadata where Triton exposes it so this can be examined
  directly; `repeat_compare()` re-measures the 64×64 4-vs-8-warp result over
  multiple runs with GPU clock/temperature snapshots to check repeatability and
  rule out clock/thermal drift.

## Performance tradeoffs and limitations

- A straightforward tiled kernel does not match cuBLAS; PyTorch was faster
  throughout. This is expected and reported honestly — the value here is the
  measured comparison, not a claim of beating the vendor library.
- FP16 inputs with FP32 accumulation balance speed and accuracy; the FP16
  *reference* itself is a source of relative-error noise near zero.
- Results are specific to the Tesla T4 (CC 7.5) and the fixed `num_stages=2`.

## Reproduce in Google Colab

1. Open `notebooks/colab_runner.ipynb` in Colab with a **T4 GPU** runtime.
2. **Cell 1** — verify the environment matches the table above.
3. **Cell 2** — set `REPO_ROOT` (clone this repo or mount Drive).
4. **Cell 3** — 256×256 smoke test.
5. **Cell 4** — full correctness suite → `results/correctness.csv`.
6. **Cell 5 / Cell 6** — rectangular-case diagnostic and FP32-reference check.
7. **Cell 7** — benchmarks → `results/benchmarks.csv`.
8. **Cell 8** — methodology review (resources, repeat comparison) →
   `results/benchmarks_repeat.csv`.
9. **Cell 9** — plots → `results/latency_comparison.png`,
   `results/throughput_comparison.png`.

Save artifacts off Colab's ephemeral runtime by downloading `results/` or
committing to the repo. Only verified, observed results are committed — never
fabricated or empty data.

## Attribution and differences from the Triton tutorial

The kernel structure was informed by Triton's official matrix-multiplication
tutorial. This project differs in several deliberate ways:

- **Plain row-major 1-D program ordering** instead of the tutorial's grouped
  (super-grouping) ordering.
- **Manual configuration experiments** across six fixed configs instead of
  Triton `autotune`.
- **No fused activation / epilogue.**
- A **custom correctness harness** (two references, fixed FP32-evaluated
  tolerances) and a **custom benchmark harness** (preallocated-output, median +
  quantiles, honest skip reporting).
- A **Tesla T4-specific** environment (CC 7.5, FP16 in / FP32 accumulate / FP16
  out, `num_stages=2`).

## Repository layout

```
kernels/matmul.py             # Triton kernel + MatmulConfig + matmul() host wrapper
tests/test_matmul.py          # smoke test, correctness suite, diagnostics, FP32 check
benchmarks/benchmark.py       # benchmarking + methodology review + plotting
notebooks/colab_runner.ipynb  # thin runner: imports modules, runs each phase
results/                      # benchmarks.csv, correctness.csv, *.png (generated)
requirements.txt              # documented versions (minimal installs)
```
