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

- **Configuration-level:** **24 / 30** shape×config combinations passed the
  **FP16-reference** criterion; **6 failed** — all six configs on the
  `1000×1500×700` case. **0** compilation or runtime errors.
- **Shape-level:** of the **5 distinct shapes**, **4 / 5** passed on every
  config (`256³`, `512³`, `64×128×96`, `130×70×50`); **1 / 5** (`1000×1500×700`)
  failed the FP16-reference criterion on all six configs. The 24/30
  configuration-level count and the 4/5 shape-level count describe the same
  result at different granularities and are **not** interchangeable: the 6
  failures are 6 configs of one shape, not 6 independent shape failures.
- Investigation of the failing case (measured, not assumed): 79 violating
  elements out of 1,500,000 (0.0053%); **79 of 79** overlapped elements where
  PyTorch's own FP16 and FP32 references disagree; violation densities were
  near-identical at boundary (5.76e-05) and interior (5.24e-05).
- A targeted **FP32-reference** check of the full `1000×1500×700` output passed
  with **0 / 1,500,000** violations (max abs error 0.0442).

**Interpretation (evidence-based, not proven):** follow-up experiments found
that disabling PyTorch's `allow_fp16_reduced_precision_reduction` setting
eliminated the tolerance violations against the FP64-derived reference on the
tested `1000×1500×700` workload (0 violations with it disabled, vs the default
behavior that produced the mismatches). This is evidence that **PyTorch's
default reduced-precision reduction behavior contributes to the discrepancy**
between the custom kernel's output and the FP16 `torch.matmul` reference. We do
**not** claim to have established the exact internal cuBLAS arithmetic, and we do
**not** relabel the 24/30 FP16-reference result. See the numerical investigation
section below for the measured figures.

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

Observed on the Tesla T4 (all values read from `results/benchmarks.csv`,
first-run `latency_ms_primary_median`):

- All 30 Triton configurations executed with **no resource or compilation
  errors**.
- The **64×64×32, 8-warps** configuration was the fastest custom config across
  all five workloads (512³: 0.0942 ms; 1024³: 0.6976 ms; 2048³: 6.2935 ms;
  4096³: 51.882 ms; `1000×1500×700`: 0.6927 ms).
- `1024³`, 64×64 tile: 4→8 warps reduced custom median latency from
  **1.4655 ms → 0.6976 ms** (ratio **≈ 2.10×**).
- `4096³`, 64×64 tile: 4→8 warps reduced median latency from
  **104.436 ms → 51.882 ms** (ratio **≈ 2.01×**).
- **All custom configurations remained slower than PyTorch** on every workload.
  The best custom standing vs PyTorch was `speedup_vs_torch ≈ 0.195`
  (`1000×1500×700`, 64×64/8-warp), i.e. about **5× slower** than PyTorch; the
  512³ 64×64/4-warp case was `≈ 0.073` (about 14× slower).
- The **128×128** configurations showed severe regressions. On `4096³`,
  128×128/4-warp measured **2552.76 ms**; relative to the same-shape custom
  baselines that is **≈ 24.4× slower than 64×64/4-warp** (104.44 ms) and
  **≈ 49.2× slower than 64×64/8-warp** (51.88 ms). Static disassembly later
  linked this to heavy register spilling and local-memory traffic — see the
  hardware investigation section
  below (correlational, not a full causal measurement).

> **Reading `speedup_vs_torch`:** it is defined as
> `torch_median / triton_median` (PyTorch latency divided by Triton latency).
> A value **< 1 means the custom Triton kernel is slower** than PyTorch; **> 1**
> would mean faster. Every measured value is < 1.

Plots (generated from the measured CSV only):
- `results/latency_comparison.png` — median latency, fastest custom config vs
  PyTorch (log-scale y-axis so `512³`…`4096³` are all legible).
- `results/throughput_comparison.png` — effective TFLOPS, fastest custom config
  vs PyTorch.
- `results/config_speedup.png` — **custom-vs-custom** config-level speedup
  (64×64, 8-warp vs 4-warp); explicitly not a comparison against PyTorch.

### Repeatability (second run, separate file)

A targeted comparison of the 64×64×32 tile at 4 vs 8 warps on `1024³` and
`4096³` was repeated 5 times each (`results/benchmarks_repeat.csv`). The
8-vs-4-warp latency ratio was reproducible: on `4096³`, mean 4-warp
≈ **101.76 ms** vs mean 8-warp ≈ **50.70 ms**, a raw ratio of **≈ 2.01×**,
consistent with the first run. Within each config the run-to-run spread was
small (4-warp 101.66–101.87 ms; 8-warp 50.40–51.28 ms).

The repeat run records both the primary median and the quantile-run p50 per
repeat; small `median_minus_p50_ms` differences are expected because the two
come from separate `do_bench` invocations.

### GPU clock / thermal observation (corrects an earlier claim)

The repeat CSV shows the two configs **did not run at the same GPU clock**, and
the runs were **not interleaved** (all 4-warp repeats first, then all 8-warp
repeats). Observed SM clocks:

- `1024³`: 4-warp ≈ 1245–1260 MHz; 8-warp ≈ 1395–1410 MHz.
- `4096³`: 4-warp ≈ 900–945 MHz; 8-warp ≈ 1050–1140 MHz.
- Temperature rose monotonically across the session (72 °C → 77 °C).

An earlier version of this README claimed the warp-count speedup was independent
of clock/thermal changes. **That claim was incorrect** and is retracted: because
the 8-warp config ran at a higher clock and later in a warming session, the raw
~2.0× ratio **conflates** the warp-count effect with a clock-frequency
difference.

#### Approximate clock-normalized comparison (explicitly labeled, not controlled)

To estimate how much of the speedup survives after removing the clock
difference, we use a simple proxy: *work ∝ latency × clock* (a config running at
a higher clock does proportionally more cycles in the same wall-clock time).
Dividing out each config's own mean clock gives a clock-invariant estimate.

For `4096³`, 64×64 (means over 5 repeats; clock = mean of the before/after
readings):

| config | mean latency (ms) | mean SM clock (MHz) | latency × clock (proxy) |
|--------|------------------:|--------------------:|------------------------:|
| 4-warp | 101.76 | ≈ 933 | ≈ 94,942 |
| 8-warp | 50.70 | ≈ 1082 | ≈ 54,857 |

- **Raw wall-clock ratio:** ≈ **2.01×**.
- **Approximate clock-normalized ratio:** 94,942 / 54,857 ≈ **1.73×**.

So an estimated ~1.7× of the improvement plausibly comes from the warp-count
change itself, with the remainder attributable to the higher clock the 8-warp
run happened to receive.

**Method and limitations (important):** this is a back-of-envelope normalization,
**not** a controlled clock-independent measurement. (1) The clocks were sampled
*before* and *after* the timed region, not continuously *during* it, so the mean
is approximate. (2) The runs were not interleaved and temperature drifted, so
clock and thermal effects are confounded with ordering. (3) The `latency × clock`
proxy assumes latency scales inversely with clock, which holds only for
compute-bound, clock-sensitive work and ignores memory-bound components. A
properly controlled comparison would **lock the SM clock**
(`nvidia-smi --lock-gpu-clocks`) and **interleave** the configs; we have not done
that here, so the ~1.73× figure is an estimate to be treated as a hypothesis,
not a verified result.

### Quantile ties and limited samples (noted)

In `results/benchmarks.csv`, several of the **slow** configurations report
**identical p20 = p50 = p80** values (e.g. `2048³` and `4096³` 128×128 rows, and
some 128×64 rows). This indicates `do_bench` collected very few timed iterations
for those long-running kernels, so the quantiles collapse onto a single
sample rather than reflecting a real distribution. These quantiles should be
read as low-sample point estimates, not characterized spreads.

### Two distinct comparisons (kept separate)

- **Custom-vs-custom (optimization relative to my 4-warp baseline):** switching
  the 64×64 tile from 4 → 8 warps gave a raw **≈ 2.10×** (`1024³`,
  1.4655→0.6976 ms) and **≈ 2.01×** (`4096³`, 104.436→51.882 ms) lower median
  latency (first-run `benchmarks.csv`). This is an improvement **of my kernel
  over my own 4-warp baseline** and is plotted in `results/config_speedup.png`.
  Note the raw ratio includes a clock-frequency difference (see the
  clock-normalized estimate above, ≈ 1.73× for `4096³`).
- **Custom-vs-PyTorch (absolute standing):** even the fastest custom config
  (64×64, 8 warps) remained **slower than PyTorch** on every workload
  (`speedup_vs_torch < 1` for every row). The config-level improvement above
  does **not** change this; the two comparisons are reported separately and
  never conflated.

### Interpretation vs hypothesis (kept separate)

- **Measurement:** 8 warps beat 4 warps for the 64×64 tile on the large
  workloads; PyTorch (cuBLAS-backed) was faster than every custom config; the
  128×128 configs regressed severely.
- **Resource metadata:** metadata was not readable from the Triton JIT Python
  handle on this 3.6.0 / T4 build, but the static-disassembly path
  (`cuobjdump`) did yield register / spill / shared-memory counts and SASS
  instruction mixes. See the dedicated investigation section below for the
  actual numbers and their limitations.

## Hardware and numerical investigation (static evidence)

This section records a focused diagnostic pass on two configurations
(`benchmarks/diagnostics.py`, Colab Cells 10–11). Raw artifacts are under
`results/diagnostics/`. **These are documentation findings; they do not modify
the kernel or any original measurement.**

### Hardware evidence (static disassembly — not runtime counters)

Observed on the Tesla T4:

| metric | 64×64×32, 8 warps | 128×128×32, 4 warps |
|--------|------------------:|--------------------:|
| registers / thread (`n_regs`) | 188 | 64 |
| spills (`n_spills`) | 0 | 2,806 |
| shared memory (bytes) | 8,192 | 16,384 |
| SASS HMMA (Tensor Core FP16 MMA) | 0 | 0 |
| SASS IMMA | 0 | 0 |
| SASS FFMA (FP32 fused multiply-add) | 512 | 4,096 |
| SASS LDL / STL (local load / store) | 0 / 0 | 3,955 / 3,536 |

PTX and TTGIR showed **no** MMA instruction or MMA-layout matches for either
configuration.

**Important measurement caveat:** every count above is a **static disassembly
count** (from the compiled SASS/PTX/TTGIR), **not** a runtime
hardware-performance-counter measurement. A static count reflects how many such
instructions exist in the compiled program, not how many times they execute at
runtime, and it does not by itself quantify time spent. Register and spill
figures likewise come from the compiled binary, not from a profiler.

### Numerical evidence (FP64-anchored investigation, separate from the suite)

On the `1000×1500×700` workload (original input scheme, seed 2), comparing the
custom Triton output against an FP64 reference rounded to FP16:

- **0** tolerance violations (atol = rtol = 1e-2); max abs error **0.0625**;
  mean abs error **≈ 8.38e-6**.
- **1,496,774 / 1,500,000 (99.7849%)** Triton outputs **exactly** match the
  rounded FP64 reference (bitwise FP16).
- The **original PyTorch FP16 reference** vs rounded FP64: **80** tolerance
  violations, mean abs error **≈ 0.002640**.
- PyTorch FP16 with **`allow_fp16_reduced_precision_reduction=False`** vs rounded
  FP64: **0** violations, mean abs error **≈ 2.384e-5**.
- All six Triton configurations produced **bitwise-identical** results
  (`torch.equal` across the six configs).

**Relationship to the committed correctness result:** the original
FP16-reference suite result stands unchanged — **24 / 30** config-level passes
(**4 / 5** shapes), with the six `1000×1500×700` failures recorded in
`results/correctness.csv`. This FP64-based pass is a **separate investigation**,
not a re-scoring of that suite: it indicates the custom kernel closely tracks
the true (FP64) result and that the original FP16-reference failures line up
with PyTorch's default FP16 reduced-precision reduction (which, when disabled,
brings the PyTorch FP16 reference into agreement with FP64). The committed
pass/fail flags are not changed or reinterpreted.

### Conclusions (supported) and their limits

- **Supported (scoped):** the **two inspected configurations**, compiled with
  **Triton 3.6.0 for the NVIDIA Tesla T4 (sm_75)**, contain **no HMMA/IMMA
  Tensor Core matrix instructions** in their disassembly (SASS HMMA/IMMA = 0, no
  PTX/TTGIR MMA matches); both perform the multiply with ordinary **FP32 FFMA**
  instructions. This finding is specific to these two configs on this
  Triton/arch combination and is **not** generalized to other Triton versions,
  other Turing GPUs, or other possible Triton matmul implementations. The
  precise reason for this lowering behavior has not been isolated, and we do not
  speculate about it.
- **Supported:** the 128×128/4-warp kernel shows **2,806 spills** and heavy
  local-memory traffic (**LDL 3,955 / STL 3,536**), **consistent with** its
  severe measured slowdown.
- **Not claimed:** these static observations do **not** establish the complete
  quantitative cause of the gap versus PyTorch. We have not profiled runtime
  instruction issue, occupancy, or memory-stall cycles, and PyTorch's cuBLAS
  path was not disassembled. The spill evidence is correlational with the
  slowdown, not a measured time breakdown.

## Performance tradeoffs and limitations

- A straightforward tiled kernel does not match cuBLAS; PyTorch was faster
  throughout. This is expected and reported honestly — the value here is the
  measured comparison, not a claim of beating the vendor library.
- FP16 inputs with FP32 accumulation balance speed and accuracy; the FP16
  *reference* itself is a source of relative-error noise near zero.
- Results are specific to the Tesla T4 (CC 7.5) and the fixed `num_stages=2`.
- **512³ latency variability (noted, not explained):** the `512³` first-run rows
  show unusually wide quantile spread for some configs — e.g. 64×64/4-warp has
  p20 = 0.1512 ms but a primary median of 0.3010 ms, and 64×64/8-warp has
  primary median 0.0942 ms with p20 = p50 ≈ 0.094–0.096 ms. At this small size
  the kernel runtime is close to launch/measurement overhead, so the medians are
  noisier than at larger sizes. This is flagged as an observation; we do not yet
  have a diagnosed cause.

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
   `results/throughput_comparison.png`, `results/config_speedup.png`.
10. **Cell 10** — hardware diagnostics (read-only; direct `_matmul_kernel`
    launch) → `results/diagnostics/`.
11. **Cell 11** — numerical diagnostics (FP16 / FP16-no-reduce / FP64 references,
    cross-config `torch.equal`) → `results/diagnostics/`.

> Cells 10–11 are an investigation pass. They do not modify the kernel, the
> `matmul()` wrapper, or any committed result file. The diagnostic **code
> itself** does not automatically infer a conclusion — it only emits the
> compiled-kernel metadata and the PTX/TTGIR/SASS artifacts. The findings stated
> in the investigation section above come from the **subsequent analysis** of
> those generated artifacts, which supports them.

Save artifacts off Colab's ephemeral runtime by downloading `results/` or
committing to the repo. Only verified, observed results are committed — never
fabricated or empty data.

### Publishing the Colab-generated result files (without overwriting)

The committed repo intentionally contains **no** fabricated result files; the
measured CSVs/PNGs are produced in Colab and added separately. To add them to
GitHub without clobbering anything, from a clean local clone copy the downloaded
Colab artifacts into `results/`, then stage and commit only those files:

```
# after unzipping the Colab download into ./results
git add results/correctness.csv results/benchmarks.csv \
        results/benchmarks_repeat.csv results/latency_comparison.png \
        results/throughput_comparison.png results/config_speedup.png
git commit -m "Add measured T4 result files (observed Colab run)"
git push origin main
```

Because these are new result files (not tracked before), the commit only adds
them; existing code and docs are untouched.

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
