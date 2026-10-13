# Triton GPU Kernel Optimization and Benchmarking

A learning-focused, reproducible study that implements a tiled FP16 matrix
multiplication kernel in Triton, validates it against `torch.matmul`, benchmarks
it against PyTorch, and runs manual configuration-level optimization experiments
on an NVIDIA Tesla T4.

> This README is a **stub**. It is finalized in Phase 5 with the measured
> results, methodology, experimental configurations, tradeoffs, and Colab
> reproduction steps. Nothing in the results sections is filled in until it is
> backed by observed Colab output.

## Execution model

- Code is authored and edited locally in Kiro (no GPU).
- All GPU execution happens in **Google Colab** on a **Tesla T4** (compute
  capability 7.5, Turing).
- Confirmed stack: PyTorch 2.11.0+cu130, CUDA 13.0, Triton 3.6.0. These are
  preinstalled on the target Colab runtime and must not be reinstalled or
  downgraded.
- Numerics: FP16 inputs, FP32 accumulation, FP16 output. No BF16/FP8/Ampere+.

## Repository layout

```
kernels/matmul.py             # Triton kernel + MatmulConfig + matmul() host wrapper
tests/test_matmul.py          # smoke test now; full correctness suite in Phase 2
benchmarks/benchmark.py       # benchmarking framework (Phase 3)
notebooks/colab_runner.ipynb  # thin runner: imports modules, runs each phase
results/                      # benchmarks.csv, correctness.csv, *.png (generated)
requirements.txt              # documented versions (minimal installs)
```

## Status

Phase 0 (repo skeleton, kernel, host wrapper, 256x256 smoke test) is implemented
and awaiting GPU verification in Colab.
