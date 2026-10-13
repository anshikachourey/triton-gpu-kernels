# Requirements Document

## Introduction

This feature is a learning-focused, reproducible GPU kernel engineering study that implements a tiled FP16 matrix-multiplication kernel in Triton, validates its numerical correctness against `torch.matmul`, benchmarks it against PyTorch on the same GPU, and runs manual configuration-level optimization experiments. It operates under a split execution model: code is authored and edited locally in Kiro (no GPU), while all GPU execution happens in Google Colab on an NVIDIA Tesla T4. Every phase ends at an explicit stop point where observed Colab output must be pasted back before proceeding. No claim about compilation, correctness, benchmark numbers, or performance is made without developer-pasted evidence. These requirements are derived strictly from the approved design and are scoped to it; no new features are introduced.

## Glossary

- **Tile**: A fixed-size `BLOCK_M × BLOCK_N` region of the output matrix computed by one Triton program.
- **BLOCK_M / BLOCK_N / BLOCK_K**: Compile-time tile dimensions for the output rows, output columns, and the shared (K) reduction dimension.
- **num_warps**: Number of warps assigned to a Triton program; part of the experiment matrix (4 or 8).
- **num_stages**: Software-pipelining depth passed explicitly to the kernel launch; fixed at 2 for all experiments.
- **do_bench**: `triton.testing.do_bench`, used for warmed-up, synchronized, steady-state latency measurement.
- **TFLOPS**: Throughput in trillion floating-point operations per second, computed as `(2*M*N*K) / (latency_seconds * 1e12)`.
- **FP16 / FP32**: Half-precision (16-bit) and single-precision (32-bit) floating-point. Inputs/outputs are FP16; accumulation is FP32.
- **Turing / T4**: NVIDIA Tesla T4 GPU, compute capability 7.5 (Turing architecture).
- **Smoke test**: A minimal `256×256` run to confirm the kernel compiles and executes before fuller work.
- **OOM**: Out-of-memory condition (`torch.cuda.OutOfMemoryError`) during allocation or launch.
- **OutOfResources**: `triton.runtime.errors.OutOfResources`, raised when a config exceeds T4 resource limits.

## Requirements

### Requirement 1: Custom Tiled FP16 Triton Matmul Kernel

**User Story:** As a GPU kernel developer, I want a custom tiled FP16 Triton matmul kernel with a validated host wrapper, so that I can compute `C = A @ B` across arbitrary shapes with FP32 accumulation.

#### Acceptance Criteria

1. WHERE inputs are square or rectangular FP16 matrices with `A.shape[1] == B.shape[0]`, THE Kernel SHALL compute `C = A @ B` and return an FP16 result of shape `[M, N]`.
2. WHERE matrix dimensions are not multiples of the tile sizes, THE Kernel SHALL apply boundary masks on every load and store so that no out-of-bounds element is read or written.
3. THE Kernel SHALL accumulate partial products in FP32 and cast only the final tile to FP16 on store.
4. THE Kernel SHALL use `@triton.jit`, `tl.dot`, `tl.program_id` for tile assignment, explicit pointer arithmetic for A/B/C tiles, and masked `tl.load`/`tl.store` operations.
5. THE Kernel SHALL accept configurable `BLOCK_M`, `BLOCK_N`, `BLOCK_K`, and `num_warps`, with `num_stages` passed explicitly and fixed at 2, and SHALL NOT use autotuning.
6. WHEN a preallocated `out` tensor is supplied to the host wrapper, THE Host_Wrapper SHALL validate that `out` has shape `[M, N]`, FP16 dtype, the same CUDA device as the inputs, and a contiguous layout before launching.

### Requirement 2: Numerical Correctness Validation

**User Story:** As a kernel developer, I want numerical correctness validation against `torch.matmul`, so that I can trust the kernel before accepting any benchmark.

#### Acceptance Criteria

1. WHEN the kernel output is compared to `torch.matmul`, THE Correctness_Suite SHALL apply the elementwise rule `abs(c_triton - c_ref) <= atol + rtol*abs(c_ref)` with fixed `atol=1e-2` and `rtol=1e-2`, evaluated in FP32.
2. THE Correctness_Suite SHALL evaluate correctness across `256^3`, `512^3`, `1000×1500×700`, at least one small non-square shape, and at least one non-multiple-of-tile shape.
3. THE Correctness_Suite SHALL record the max absolute error and a robust relative error against both `ref_fp16` (primary, pass/fail decision) and `ref_fp32` (diagnostic), reflecting FP32 accumulation with FP16 output.
4. WHEN a case is run with a fixed seed and config, THE Correctness_Suite SHALL produce deterministic inputs and metrics within run-to-run GPU nondeterminism bounds.
5. IF a case violates the fixed-tolerance rule, THEN THE Correctness_Suite SHALL mark the case `failed`, preserve the raw error metrics, and SHALL NOT loosen tolerances or discard the case.
6. THE Correctness_Suite SHALL validate every tile/warp config before its benchmark results are accepted.

### Requirement 3: Reproducible Benchmarking Framework

**User Story:** As a performance engineer, I want a reproducible benchmarking framework, so that I can compare the Triton kernel and PyTorch fairly on the same GPU.

#### Acceptance Criteria

1. THE Benchmark_Framework SHALL measure primary latency via `do_bench(return_mode="median")` and additionally record p20/p50/p80 quantiles from a separate `do_bench(quantiles=[0.2, 0.5, 0.8])` invocation, treating the quantile-run p50 as distinct from the primary median.
2. THE Benchmark_Framework SHALL compute `TFLOPS = (2*M*N*K) / (latency_seconds * 1e12)` and speedup from measured latencies, and SHALL exclude one-time compilation from steady-state timing.
3. THE Benchmark_Framework SHALL benchmark the identical operation `C = A @ B` using preallocated FP16 output buffers on both the torch and triton paths, record GPU and software environment metadata, and cover sizes `512^3`, `1024^3`, `2048^3`, `4096^3`, and `1000×1500×700`.

### Requirement 4: Manual Optimization Experiments

**User Story:** As a kernel developer, I want manual configuration experiments, so that I can identify the fastest correct kernel configuration per workload from measured evidence.

#### Acceptance Criteria

1. THE Experiment_Runner SHALL attempt every `(BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)` config in the set `(64,64,32,4,2)`, `(64,64,32,8,2)`, `(128,128,32,4,2)`, `(128,128,32,8,2)`, `(128,64,32,4,2)`, `(128,64,32,8,2)`.
2. WHEN a config is attempted, THE Experiment_Runner SHALL validate correctness, then benchmark it, and SHALL identify the fastest correct config per workload.
3. IF a config is skipped or fails, THEN THE Experiment_Runner SHALL record the actual error type and message rather than a generic label.
4. THE Experiment_Runner SHALL compare the initial Triton implementation, the best measured Triton config, and PyTorch using only measured records, without inventing speedups.

### Requirement 5: Results and Documentation

**User Story:** As a reader, I want results artifacts and documentation, so that I can understand and reproduce the study.

#### Acceptance Criteria

1. THE Results_Module SHALL produce `results/benchmarks.csv`, `results/correctness.csv`, `results/latency_comparison.png`, and `results/throughput_comparison.png`.
2. THE Documentation SHALL include the objective, tiled matmul theory, memory access patterns, environment, correctness methodology, benchmark methodology, experimental configs, measured results, tradeoffs/limitations, and Colab reproduction steps.
3. THE Documentation SHALL attribute the Triton official matmul tutorial, document the differences from it, and separate measurements from interpretations and hypotheses.
4. THE Results_Module SHALL persist artifacts outside Colab's ephemeral runtime via download or git, and SHALL NOT auto-commit unverified, empty, or fabricated results.

### Requirement 6: Execution Environment and Evidence Integrity

**User Story:** As a developer under a split execution model, I want strict execution-environment and evidence-integrity rules, so that no unverified claim is ever made and the confirmed stack stays intact.

#### Acceptance Criteria

1. THE System SHALL run authoring locally in Kiro without a GPU and perform all GPU execution in Colab on the confirmed T4 stack (PyTorch 2.11.0+cu130, CUDA 13.0, Triton 3.6.0, compute capability 7.5), using FP16 inputs with FP32 accumulation and SHALL NOT use BF16, FP8, or Ampere+ features.
2. IF the matmul kernel fails to compile or run, THEN THE System SHALL diagnose the actual error before proposing any dependency change, and SHALL NOT auto-downgrade or reinstall PyTorch or Triton.
3. THE System SHALL NOT claim that code ran, benchmarks succeeded, or performance improved without developer-pasted Colab output, treating a prior vector-add success as no guarantee that the matmul kernel compiles.
4. WHEN a phase completes, THE System SHALL end with a five-part closing report (files changed, implementation explanation, Colab execution instructions, what remains unverified, explicit stop point) and halt until observed output is pasted.
