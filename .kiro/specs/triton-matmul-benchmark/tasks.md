# Implementation Plan: Triton GPU Kernel Optimization and Benchmarking

## Overview

This plan implements a tiled FP16 Triton matmul kernel, a correctness suite, a benchmarking framework, manual configuration experiments, and results/documentation. It follows the design's **phased workflow under a split execution model**: all code is authored locally in Kiro (no GPU), while every GPU-dependent task runs in Google Colab on a Tesla T4. Kiro cannot run or verify GPU execution — each GPU-dependent task therefore requires **developer-pasted Colab output** before any success, correctness, or performance claim is made.

**Stop-point rule:** Every phase ends with a Colab execution + stop-point sub-task. Work does not proceed to the next phase until the developer pastes the observed Colab output and it is interpreted. Each phase closes with the five-part report (files changed, implementation explanation, Colab execution instructions, what remains unverified, explicit stop point) per Requirement 6.4.

Language: Python with Triton (per the confirmed stack — torch==2.11.0+cu130, triton==3.6.0, CUDA 13.0, compute capability 7.5). Do NOT reinstall or downgrade torch/triton.

## Tasks

- [x] 1. Phase 0 — Repo skeleton, kernel, host wrapper, smoke test (implement now)

  > **Phase 0 status: COMPLETE (verified on NVIDIA Tesla T4, Colab).** The
  > 256x256 smoke test produced shape (256, 256), dtype torch.float16, config
  > (64, 64, 32, 4, 2), max_abs_error (FP32 eval) = 0.015625,
  > robust_rel_error = 1.16e-03, and the explicit elementwise check
  > (torch.testing.assert_close, atol=rtol=1e-2) returned **passed: True /
  > PASS**. GPU compilation, execution, and the 256x256 tolerance check are
  > all confirmed by observed Colab output.

  - [x] 1.1 Create repository skeleton and documented dependency stack
    - Create `kernels/matmul.py`, `tests/test_matmul.py` (stub), `benchmarks/benchmark.py` (stub), `notebooks/colab_runner.ipynb`, and the `results/` directory
    - Create `requirements.txt` documenting the confirmed stack: `torch==2.11.0+cu130`, `triton==3.6.0`, `matplotlib` (install only if missing); note that torch/triton are preinstalled on Colab and MUST NOT be reinstalled or downgraded
    - Create `README.md` stub to be finalized in Phase 5
    - Keep all kernel/test/benchmark logic in importable modules; the notebook only imports and calls (key boundary rule)
    - _Requirements: 5.1, 6.1, 6.2_

  - [x] 1.2 Implement MatmulConfig dataclass
    - Add `@dataclass(frozen=True) MatmulConfig` with `block_m`, `block_n`, `block_k`, `num_warps`, and `num_stages` fixed at 2 (recorded, not hidden)
    - Default `MatmulConfig(block_m=64, block_n=64, block_k=32, num_warps=4, num_stages=2)`
    - _Requirements: 1.5_

  - [x] 1.3 Implement the `_matmul_kernel` Triton kernel with teaching comments
    - Use `@triton.jit`; derive `(pid_m, pid_n)` from `tl.program_id(axis=0)` with row-major 1-D ordering
    - Explicit pointer arithmetic for A/B/C tiles via strides; `tl.arange` offsets for M/N/K
    - FP32 accumulator initialization; K-loop with `tl.dot`; masked `tl.load`/`tl.store` with `other=0.0`; FP16 cast on store
    - Add meaningful inline teaching comments covering: program IDs, grid-to-tile mapping, pointer offsets, memory access patterns, boundary masks, FP32 accumulator init, K-loop accumulation, and FP16 output conversion
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

  - [x] 1.4 Implement the `matmul()` host wrapper
    - Validate inputs: `a.dtype == b.dtype == float16`, `a.ndim == b.ndim == 2`, `a.shape[1] == b.shape[0]`, both on CUDA
    - Validate supplied `out`: shape `[M, N]`, FP16 dtype, same CUDA device as inputs, contiguous layout; allocate FP16 `C[M,N]` only when `out` is not supplied
    - Compute 1-D grid `cdiv(M, block_m) * cdiv(N, block_n)`; launch with strides, `num_warps`, and `num_stages` both passed EXPLICITLY
    - _Requirements: 1.1, 1.5, 1.6_

  - [x] 1.5 Add a 256x256 smoke test with conservative config
    - In `tests/test_matmul.py`, add a minimal smoke-test entry point that runs 256x256 with `MatmulConfig(64, 64, 32, 4, num_stages=2)`
    - Compare against `torch.matmul` by converting both outputs to FP32 and printing the observed max absolute error
    - Added an explicit elementwise pass/fail check via `torch.testing.assert_close(c_triton.float(), c_ref.float(), atol=1e-2, rtol=1e-2)`, reported as `passed`/`check_message` alongside the raw metrics
    - Verified on T4: passed=True / PASS
    - _Requirements: 1.1, 6.3_

  - [x] 1.6 Provide exact Colab cells and STOP for pasted output (Phase 0 stop point)
    - Notebook cells: (a) environment verification printing GPU name, compute capability, and torch/CUDA/triton versions; (b) repo setup setting `REPO_ROOT` via `git clone` OR upload/mount; (c) run the 256x256 smoke test and print shape, dtype, FP32 errors, and the explicit tolerance verdict
    - Fixed the `tests.test_matmul` import collision permanently: Cell 3 loads the test module directly from its file path via `importlib.util.spec_from_file_location` instead of `import tests.test_matmul`
    - Verified on T4; observed output pasted back and interpreted
    - _Requirements: 6.1, 6.3, 6.4_

- [x] 2. Phase 1 — Finalize and generalize the tiled kernel (after Phase 0 verified)
  - [x] 2.1 Generalize the kernel for all required shape families
    - Confirmed the existing kernel already handles square, rectangular, small non-square, and non-multiple-of-tile shapes with no out-of-bounds read/write: grid is `cdiv(M,BLOCK_M)*cdiv(N,BLOCK_N)`; masks cover `offs_m<M`, `offs_n<N`, `offs_k<k_remaining` on every load/store with `other=0.0`; single final FP16 cast after FP32 accumulation. No implementation change required
    - Shape generality is exercised through the Phase 2 correctness suite's shape cases (verified on GPU)
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6_

  - [x] 2.2 Add step-by-step kernel explanation
    - `kernels/matmul.py` carries a coherent step-by-step explanation (program IDs → grid mapping → pointer arithmetic → memory access → masks → FP32 accumulator → K-loop → FP16 conversion); the 4x4 / 2x2 hand trace was walked through with the developer
    - _Requirements: 1.4_

  - [x] 2.3 Provide Colab cells and STOP for pasted output (Phase 1 stop point)
    - Phase 1 shape generality folded into the Phase 2 correctness suite (Cell 4). Verified on T4: square/rectangular/small-non-square/non-multiple-of-tile shapes all exercised; no OOB behavior observed
    - _Requirements: 6.3, 6.4_

- [x] 3. Phase 2 — Correctness suite (after Phase 1 verified)
  - [x] 3.1 Implement input generation and references
    - `make_inputs(m, k, n, seed=0)`: `torch.manual_seed(seed)` per case; deterministic FP16 inputs in an FP16-safe magnitude band
    - `ref_fp16(a, b)`: primary reference `torch.matmul(a, b)` (pass/fail decision)
    - `ref_fp32(a, b)`: diagnostic reference `torch.matmul(a.float(), b.float())`
    - _Requirements: 2.1, 2.3, 2.4_

  - [x] 3.2 Implement error_metrics evaluated in FP32
    - `error_metrics(c_test, c_ref)`: cast both operands to FP32 first; return `(max_abs_error, robust_rel_error)` where rel error avoids division by near-zero reference magnitudes
    - _Requirements: 2.1, 2.3_

  - [x] 3.3 Implement DEFAULT_CASES, ShapeCase, and run_correctness
    - Define `ShapeCase` and `DEFAULT_CASES` covering `256^3`, `512^3`, `1000x1500x700`, a small non-square case (e.g. `64x128x96`), and a non-multiple-of-tile case (e.g. `130x70x50`)
    - `run_correctness(cases, config)`: run EVERY config, apply elementwise rule `abs(c_triton - c_ref) <= atol + rtol*abs(c_ref)` with fixed `atol=rtol=1e-2` in FP32 against `ref_fp16` (primary) and record diagnostics vs `ref_fp32`
    - Record all four metrics regardless of pass/fail; classify `passed`/`failed`; on failure preserve raw metrics and write a `note`; never loosen tolerances or discard cases
    - Validate every tile/warp config before its benchmark results are accepted
    - This realizes Correctness Properties P1–P4 (reference agreement, shape generality, accumulation precision, determinism) through the explicit required shape cases — no external property-generation library is used
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

  - [x] 3.4 Implement CorrectnessRecord and correctness.csv writer
    - Define `CorrectnessRecord` with all fields from the design (shape, config incl. `num_stages`, seed, FP16/FP32 errors, atol, rtol, passed, note)
    - Write `results/correctness.csv` with the design's exact schema
    - _Requirements: 2.3, 2.5, 5.1_

  - [x] 3.5 Provide Colab cells and STOP for pasted output (Phase 2 stop point)
    - Cell 4 verified on T4: 24/30 passed the FP16-reference criterion, 0 compile/runtime errors; the 6 failures (1000x1500x700) were investigated and the full output verified vs the FP32 reference (0/1,500,000 violations). The 24/30 FP16-reference result is preserved, not relabeled
    - (Observed Colab output pasted and interpreted)
    - _Requirements: 6.3, 6.4_

- [x] 4. Phase 3 — Benchmark framework (after Phase 2 verified)

  > **Phase 2 investigation outcome (observed on T4):** the 6 FP16-reference
  > failures on 1000x1500x700 were investigated. 79/79 Triton violations
  > overlap elements where PyTorch's own FP16 and FP32 references disagree;
  > violation densities are near-identical at boundary (5.76e-05) vs interior
  > (5.24e-05), not a masking pattern. Strong evidence (not proof) of an
  > FP16-reference precision artifact near zero-magnitude outputs, not a kernel
  > bug. A targeted FP32-reference verification (verify_against_fp32) is added
  > as the gate: if it passes, Phase 3 proceeds; the original 24/30
  > FP16-reference results are preserved, not relabeled.
  >
  > **Note (minor, in-scope deviation):** benchmarks.csv carries timing +
  > env metadata only; per-element FP16/FP32 error metrics live in
  > correctness.csv (already validated in Phase 2) to avoid re-running the
  > correctness comparison inside the timed benchmark path.

  - [x] 4.1 Implement TFLOPS and timing primitives
    - `tflops(m, n, k, latency_ms)`: `(2*m*n*k) / (latency_seconds * 1e12)`
    - `bench_median(fn)`: primary latency via `triton.testing.do_bench(fn, return_mode="median")`
    - `bench_quantiles(fn)`: separate `do_bench(fn, quantiles=[0.2, 0.5, 0.8])` returning (p20, p50, p80); quantile-run p50 recorded independently of the primary median
    - Realizes Correctness Property P5 (TFLOPS definition from the primary median)
    - _Requirements: 3.1, 3.2_

  - [x] 4.2 Implement environment_info and EnvInfo
    - `environment_info()`: collect GPU model, compute capability, torch/CUDA/triton versions, `cuda_available`
    - _Requirements: 3.3_

  - [x] 4.3 Implement run_benchmarks with preallocated output and robust error handling
    - Benchmark identical operation `C = A @ B` on BOTH sides with preallocated FP16 out: `torch.matmul(a, b, out=c_torch)` and `matmul(a, b, config, out=c_triton)`
    - Exclude one-time compilation via a throwaway launch before timing
    - Attempt every config; catch `triton.runtime.errors.OutOfResources`, Triton compile failures, and `torch.cuda.OutOfMemoryError`; record `skipped` with the ACTUAL error type + message in `skip_reason`; call `torch.cuda.empty_cache()` and continue
    - Compute TFLOPS and speedup from measured latencies only
    - Sizes: `512^3`, `1024^3`, `2048^3`, `4096^3`, `1000x1500x700`
    - Realizes Correctness Properties P6 (evidence integrity) and P7 (operation consistency)
    - _Requirements: 3.1, 3.2, 3.3_

  - [x] 4.4 Implement BenchmarkRecord and benchmarks.csv writer
    - Define `BenchmarkRecord` (primary median, p20/p50/p80, tflops, speedup, env metadata, status, skip_reason); per-element error metrics are kept in correctness.csv (see Phase 3 note)
    - `write_csv(records, path)` producing `results/benchmarks.csv`
    - _Requirements: 3.1, 3.2, 3.3, 5.1_

  - [x] 4.5 Provide Colab cells and STOP for pasted output (Phase 3 stop point)
    - Cells 6 and 7 verified on T4: FP32-reference gate passed (0 violations); all 30 size x config benchmarks executed with 0 skips/errors; results written to results/benchmarks.csv
    - _Requirements: 6.3, 6.4_

- [x] 5. Phase 4 — Manual optimization experiments (after Phase 3 verified)

  > **Phase 4 first-run results (observed on T4; preserved as run 1):** all 30
  > configs executed with 0 resource/compile errors. Fastest custom config was
  > 64x64x32 / 8 warps across all five workloads. 1024^3: 4->8 warps 1.4652 ->
  > 0.6836 ms (~2.14x). 4096^3: 105.468 -> 51.476 ms (~2.05x). All custom
  > configs remained SLOWER than PyTorch. 128x128 configs showed severe
  > regressions (cause NOT attributed to spilling without evidence).
  > A methodology-review pass (resources, GPU clock/thermal, repeatable 64x64
  > 4-vs-8-warp comparison, median-vs-p50) is added in benchmarks.py and
  > writes a SEPARATE results/benchmarks_repeat.csv so run 1 is never
  > overwritten.

  - [x] 5.1 Wire the six-config experiment matrix into the runner
    - `run_benchmarks` attempts the full config set across all sizes; skipped/failed configs would record the actual error type + message; support is never predetermined (first run: 0 errors)
    - _Requirements: 4.1, 4.2, 4.3_

  - [x] 5.2 Identify fastest correct config and compare implementations
    - Fastest correct config per workload identified from measured records (64x64x32/8-warps); initial-Triton vs best-Triton vs PyTorch compared from measured medians only (no invented speedups; custom remained slower than PyTorch)
    - _Requirements: 4.2, 4.4_

  - [x] 5.3 Provide Colab cells and STOP for pasted output (Phase 4 stop point)
    - Methodology review verified on T4: repeatability confirmed ~2.0x lower latency for 8 vs 4 warps on 4096^3 (second run, results/benchmarks_repeat.csv, run 1 preserved). GPU clock/thermal snapshots captured; repeatable ratio across runs indicates a real config effect, not drift. Kernel resource metadata was UNAVAILABLE on this Triton 3.6.0/T4 build, so the 128x128 regression is left unexplained (not attributed to spilling)
    - _Requirements: 6.3, 6.4_

- [x] 6. Phase 5 — Results, plots, and documentation (after Phase 4 verified)
  - [x] 6.1 Implement plotting helpers
    - `plot_latency(csv_path, out_path, log_y=True)` → `results/latency_comparison.png` (log-scale y-axis for legibility across 512^3..4096^3)
    - `plot_throughput(csv_path, out_path)` → `results/throughput_comparison.png`
    - `plot_config_speedup(...)` → `results/config_speedup.png` (custom 64x64 8-warp vs 4-warp; explicitly custom-vs-custom, NOT vs PyTorch)
    - All read ONLY the measured `results/benchmarks.csv`; nothing fabricated. Latency plot verified generated on T4
    - _Requirements: 5.1_

  - [x] 6.2 Finalize README.md
    - Include: objective, tiled matmul theory, memory access patterns, environment, correctness methodology, benchmark methodology (primary median vs quantile p50, compilation exclusion, `num_stages=2` as a documented experimental variable), experimental configs, measured results, tradeoffs/limitations, and Colab reproduction steps
    - Attribute the Triton official matmul tutorial and document differences: row-major 1-D ordering vs grouped ordering, manual configs vs autotuning, no fused activation/epilogue, custom correctness + benchmark harness, Tesla T4-specific (CC 7.5, FP16 in / FP32 accumulate / FP16 out)
    - Clearly separate measurements from interpretations and hypotheses
    - _Requirements: 5.2, 5.3_

  - [x] 6.3 Implement results-persistence step
    - README documents both options: download `results/` from Colab (zip + files.download) OR add the measured files to the repo. Publish instructions add only NEW result files via explicit `git add` of each artifact, so existing measured files are never overwritten
    - No auto-commit of unverified/empty/fabricated results; only observed Colab artifacts are published
    - _Requirements: 5.4_

  - [x] 6.4 Provide final Colab cells and STOP for pasted output (Phase 5 stop point)
    - Cell 9 generates latency (log-scale), throughput, and config-speedup PNGs from measured records and displays them; developer confirmed the latency plot was generated on T4
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 6.3, 6.4_

## Notes

- Every GPU-dependent task requires developer-pasted Colab output. Kiro has NO local GPU and cannot verify GPU compilation, correctness, or performance.
- Each phase HALTS at its stop-point sub-task until the developer pastes observed Colab evidence; the next phase does not proceed before that evidence is interpreted (Req 6.3, 6.4).
- Phase 0 is the immediately-actionable block: skeleton + kernel + host wrapper + smoke test + Colab cells.
- The confirmed stack (torch==2.11.0+cu130, triton==3.6.0, CUDA 13.0) is fixed; diagnose actual errors before proposing any dependency change — never auto-downgrade or reinstall.
- Correctness Properties P1–P7 from the design are realized through the explicit required shape cases and benchmark checks, not an external property-generation library (out of scope).
- `num_stages` is fixed at 2 and recorded everywhere (config, CSVs, README); it is not a hidden variable.
- No speedup or success is claimed without a measured record.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2"] },
    { "id": 2, "tasks": ["1.3"] },
    { "id": 3, "tasks": ["1.4"] },
    { "id": 4, "tasks": ["1.5"] },
    { "id": 5, "tasks": ["1.6"] },
    { "id": 6, "tasks": ["2.1"] },
    { "id": 7, "tasks": ["2.2"] },
    { "id": 8, "tasks": ["2.3"] },
    { "id": 9, "tasks": ["3.1", "3.2"] },
    { "id": 10, "tasks": ["3.3"] },
    { "id": 11, "tasks": ["3.4"] },
    { "id": 12, "tasks": ["3.5"] },
    { "id": 13, "tasks": ["4.1", "4.2"] },
    { "id": 14, "tasks": ["4.3"] },
    { "id": 15, "tasks": ["4.4"] },
    { "id": 16, "tasks": ["4.5"] },
    { "id": 17, "tasks": ["5.1"] },
    { "id": 18, "tasks": ["5.2"] },
    { "id": 19, "tasks": ["5.3"] },
    { "id": 20, "tasks": ["6.1"] },
    { "id": 21, "tasks": ["6.2", "6.3"] },
    { "id": 22, "tasks": ["6.4"] }
  ]
}
```
