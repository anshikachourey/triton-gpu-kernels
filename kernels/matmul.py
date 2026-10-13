"""Custom tiled FP16 matrix-multiplication kernel in Triton.

This module is the single source of truth for the kernel. It is imported and
run from Google Colab (Tesla T4, compute capability 7.5). Nothing here runs on
the local machine (Kiro has no GPU) — all GPU execution and verification
happens in Colab.

Numerics (hard constraints):
  - Inputs:  FP16  (A: [M, K], B: [K, N])
  - Accumulate: FP32 (inside the K-loop, via tl.dot)
  - Output: FP16  (C: [M, N], cast only once at the end)
  - No BF16, no FP8, no Ampere+ features. No autotuning in main experiments.

Config:
  - BLOCK_M / BLOCK_N / BLOCK_K and num_warps are configurable.
  - num_stages is fixed at 2 for ALL experiments and recorded, not hidden.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@dataclass(frozen=True)
class MatmulConfig:
    """Launch configuration for the tiled matmul kernel.

    block_m / block_n / block_k : compile-time tile dimensions.
    num_warps                   : warps per Triton program (experiment matrix: 4 or 8).
    num_stages                  : software-pipelining depth. FIXED at 2 for every
                                  experiment and recorded in all CSV/README output
                                  so it is never a hidden variable.
    """

    block_m: int = 64
    block_n: int = 64
    block_k: int = 32
    num_warps: int = 4
    num_stages: int = 2


@triton.jit
def _matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute one BLOCK_M x BLOCK_N tile of C = A @ B.

    Teaching walkthrough (step by step):

    1. PROGRAM IDs & GRID-TO-TILE MAPPING
       The host launches a 1-D grid of `cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N)`
       programs. Each program owns exactly one output tile. We linearize the
       flat program id into a 2-D tile coordinate (pid_m, pid_n) using plain
       row-major ordering: pid_m = pid // num_pid_n, pid_n = pid % num_pid_n.

    2. POINTER OFFSETS
       offs_m / offs_n / offs_k are per-lane index vectors for this tile along
       the M, N, and K dimensions. Combined with the matrix strides they give
       the exact memory addresses of the A and B sub-tiles.

    3. MEMORY ACCESS PATTERN
       a_ptrs addresses A[offs_m, offs_k] (shape [BLOCK_M, BLOCK_K]);
       b_ptrs addresses B[offs_k, offs_n] (shape [BLOCK_K, BLOCK_N]).
       Each K-loop iteration advances these pointers by BLOCK_K along K.

    4. BOUNDARY MASKS
       When M/N/K are not multiples of the tile sizes, some lanes point past
       the matrix. Masks force those out-of-bounds loads to 0.0 and prevent any
       out-of-bounds store, so arbitrary shapes are handled safely.

    5. FP32 ACCUMULATOR
       `acc` starts at zero in FP32. tl.dot multiplies the FP16 sub-tiles and
       accumulates the products into this FP32 buffer — accumulation precision
       is kept in FP32 even though the inputs are FP16.

    6. K-LOOP ACCUMULATION
       Loop invariant: after processing K-offset k0, `acc` holds the FP32
       partial sum of A[:, 0:k0+BLOCK_K] @ B[0:k0+BLOCK_K, :] for in-bounds
       lanes; masked lanes contribute 0.

    7. FP16 OUTPUT CONVERSION
       Only after the full reduction do we cast `acc` to FP16 and store it with
       a boundary mask. This single final cast is the only FP16 rounding of the
       result.
    """
    # ---- 1. Program id -> 2-D tile coordinate (row-major) -----------------
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n          # which tile-row of C this program owns
    pid_n = pid % num_pid_n           # which tile-col of C this program owns

    # ---- 2. Per-lane offsets for this tile --------------------------------
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # rows of C / rows of A
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # cols of C / cols of B
    offs_k = tl.arange(0, BLOCK_K)                     # shared reduction dim

    # ---- 2/3. Base pointers for the A and B sub-tiles ---------------------
    # A[offs_m, offs_k]: 2-D pointer block of shape [BLOCK_M, BLOCK_K].
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    # B[offs_k, offs_n]: 2-D pointer block of shape [BLOCK_K, BLOCK_N].
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # ---- 5. FP32 accumulator ----------------------------------------------
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # ---- 6. Reduction over K in BLOCK_K steps -----------------------------
    for k0 in range(0, K, BLOCK_K):
        k_remaining = K - k0
        # 4. Masks: valid rows/cols of this tile AND within the current K slice.
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)

        # Out-of-bounds lanes load 0.0 so they contribute nothing to the dot.
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # FP16 x FP16 -> FP32 accumulate.
        acc += tl.dot(a_tile, b_tile)

        # Advance along K for the next slice.
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # ---- 7. Single final FP16 cast + masked store -------------------------
    c = acc.to(tl.float16)
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(
    a: "torch.Tensor",
    b: "torch.Tensor",
    config: MatmulConfig = MatmulConfig(),
    out: "torch.Tensor | None" = None,
) -> "torch.Tensor":
    """Compute C = A @ B with the custom Triton tiled kernel.

    A: FP16 [M, K], B: FP16 [K, N]. Returns C: FP16 [M, N] with FP32
    accumulation internally. If `out` is provided it is validated and used as
    the output buffer (preallocated-output benchmarking); otherwise a fresh
    FP16 [M, N] tensor is allocated.

    `config.num_warps` and `config.num_stages` are passed explicitly to the
    kernel launch (num_stages is fixed at 2 for all experiments).
    """
    # ---- Input validation -------------------------------------------------
    if a.dtype != torch.float16 or b.dtype != torch.float16:
        raise TypeError("matmul requires FP16 inputs (a and b must be float16)")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("matmul requires 2-D inputs")
    if a.shape[1] != b.shape[0]:
        raise ValueError(
            f"inner dimensions must match: a.shape[1]={a.shape[1]} "
            f"!= b.shape[0]={b.shape[0]}"
        )
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("matmul requires CUDA tensors (run on the Tesla T4 in Colab)")

    M, K = a.shape
    _, N = b.shape

    # ---- Output buffer: validate a supplied `out`, else allocate ----------
    if out is not None:
        if tuple(out.shape) != (M, N):
            raise ValueError(f"out must have shape {(M, N)}, got {tuple(out.shape)}")
        if out.dtype != torch.float16:
            raise TypeError("out must be FP16")
        if not out.is_cuda or out.device != a.device:
            raise ValueError("out must be on the same CUDA device as the inputs")
        if not out.is_contiguous():
            raise ValueError("out must use a supported (contiguous) memory layout")
        c = out
    else:
        c = torch.empty((M, N), dtype=torch.float16, device=a.device)

    # ---- 1-D grid: one program per (BLOCK_M x BLOCK_N) output tile --------
    grid = (triton.cdiv(M, config.block_m) * triton.cdiv(N, config.block_n),)

    _matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=config.block_m,
        BLOCK_N=config.block_n,
        BLOCK_K=config.block_k,
        num_warps=config.num_warps,     # explicit
        num_stages=config.num_stages,   # explicit, fixed at 2
    )
    return c
