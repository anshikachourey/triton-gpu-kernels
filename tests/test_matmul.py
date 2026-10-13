"""Correctness tests for the custom Triton matmul kernel.

Phase 0 provides ONLY the 256x256 smoke test below. The full correctness suite
(make_inputs, ref_fp16/ref_fp32, error_metrics, run_correctness, DEFAULT_CASES,
CorrectnessRecord, correctness.csv writer) is implemented in Phase 2 after the
smoke test is confirmed on the T4.

Everything here runs in Google Colab on the Tesla T4. Kiro has no local GPU and
makes no claim about results until the developer pastes the observed output.
"""

from __future__ import annotations

import torch

from kernels.matmul import MatmulConfig, matmul


def smoke_test_256(seed: int = 0) -> dict:
    """Minimal 256x256 smoke test with the conservative default config.

    Config: BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, num_warps=4, num_stages=2.

    Runs the custom kernel and compares against torch.matmul. The error is
    computed in FP32 (both outputs cast to float()) so the comparison itself
    never uses FP16 arithmetic. No pass/fail is forced before real GPU
    evidence exists — we just report the observed numbers.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Run this in Google Colab with a Tesla T4 GPU."
        )

    torch.manual_seed(seed)
    device = "cuda"

    # FP16 inputs in a modest, FP16-safe range.
    a = torch.randn((256, 256), dtype=torch.float16, device=device)
    b = torch.randn((256, 256), dtype=torch.float16, device=device)

    config = MatmulConfig(block_m=64, block_n=64, block_k=32, num_warps=4, num_stages=2)

    # Preallocated FP16 output buffer, exercising the `out=` path.
    c_triton = torch.empty((256, 256), dtype=torch.float16, device=device)
    matmul(a, b, config, out=c_triton)

    # Reference (FP16 primary).
    c_ref = torch.matmul(a, b)

    # Error metrics computed in FP32.
    diff = (c_triton.float() - c_ref.float()).abs()
    max_abs_error = diff.max().item()
    denom = c_ref.float().abs().clamp_min(1e-3)
    robust_rel_error = (diff / denom).max().item()

    result = {
        "shape": tuple(c_triton.shape),
        "dtype": str(c_triton.dtype),
        "config": (config.block_m, config.block_n, config.block_k,
                   config.num_warps, config.num_stages),
        "max_abs_error_fp32eval": max_abs_error,
        "robust_rel_error_fp32eval": robust_rel_error,
    }
    return result


if __name__ == "__main__":
    out = smoke_test_256()
    print("Smoke test 256x256")
    print(f"  output shape        : {out['shape']}")
    print(f"  output dtype        : {out['dtype']}")
    print(f"  config (BM,BN,BK,W,S): {out['config']}")
    print(f"  max abs error (FP32) : {out['max_abs_error_fp32eval']:.6e}")
    print(f"  robust rel error     : {out['robust_rel_error_fp32eval']:.6e}")
