// wmma_matmul_shared.cu
//
// OPTIMIZED (shared-memory tiled) CUDA C++ FP16 matmul using NVIDIA WMMA
// Tensor Core operations. This is a SEPARATE second kernel; the naive baseline
// in wmma_matmul.cu is left untouched.
//
// Optimization: cooperative shared-memory tiling with reuse across warps.
//   * One thread block computes a 64x64 output tile of C.
//   * Block = 16 warps (4x4), each warp owns one 16x16 output sub-tile.
//   * Per K-step (BK=16) the block cooperatively stages A's 64x16 slice and
//     B's 16x64 slice into shared memory ONCE, then all 16 warps reuse those
//     shared tiles (the 4 warps in a block-row share As rows; the 4 warps in a
//     block-column share Bs columns). This removes the baseline's redundant
//     global re-reads of A and B.
//   * FP16 inputs, FP32 accumulation, FP16 output.
//   * Deliberately simple: no async copy, no double buffering, no pipelining,
//     no autotuning.
//
// Build (Tesla T4, Turing):
//   nvcc -arch=sm_75 -Xcompiler -fPIC -shared wmma_matmul_shared.cu \
//        -o libwmma_matmul_shared.so
//
// Row-major contiguous: A[M,K] (lda=K), B[K,N] (ldb=N), C[M,N] (ldc=N).

#include <mma.h>
#include <cuda_fp16.h>
#include <cstdio>

using namespace nvcuda;

static const int WMMA_M = 16;
static const int WMMA_N = 16;
static const int WMMA_K = 16;

// Block (CTA) output tile and K-step.
static const int BM = 64;   // block tile rows of C
static const int BN = 64;   // block tile cols of C
static const int BK = 16;   // K-slice staged per step (one WMMA K)

// Warps per block along each dimension (BM/WMMA_M = 4, BN/WMMA_N = 4).
static const int WARPS_M = BM / WMMA_M;   // 4
static const int WARPS_N = BN / WMMA_N;   // 4

// Shared-memory padding (elements) to reduce bank conflicts.
static const int PAD = 8;

// Error codes (shared launcher). Mirror the baseline, plus a block-align code.
#define WMMA_OK                 0
#define WMMA_ERR_NULLPTR       -1
#define WMMA_ERR_DIM           -2
#define WMMA_ERR_ALIGN         -3   // not a multiple of 16 (WMMA tile)
#define WMMA_ERR_BLOCK_ALIGN   -4   // not a multiple of the 64 block tile

// ---------------------------------------------------------------------------
// Shared-memory tiled WMMA kernel. One block -> 64x64 output tile.
// ---------------------------------------------------------------------------
__global__ void wmma_matmul_shared_kernel(const half* __restrict__ A,
                                          const half* __restrict__ B,
                                          half* __restrict__ C,
                                          int M, int N, int K) {
    const int lda = K, ldb = N, ldc = N;

    // --- Block-to-tile mapping ----------------------------------------------
    // This block owns C[blockRow0 : blockRow0+BM, blockCol0 : blockCol0+BN].
    const int blockRow0 = blockIdx.y * BM;
    const int blockCol0 = blockIdx.x * BN;

    // --- Warp-to-subtile mapping within the block --------------------------
    // 16 warps laid out 4 (N) x 4 (M). warpId in [0,16).
    const int threadId = threadIdx.y * blockDim.x + threadIdx.x;
    const int warpId   = threadId / warpSize;      // 0..15
    const int warpRow  = warpId / WARPS_N;         // 0..3  (which 16-row band)
    const int warpCol  = warpId % WARPS_N;         // 0..3  (which 16-col band)

    // This warp's 16x16 output sub-tile origin (within C).
    const int subRow0 = blockRow0 + warpRow * WMMA_M;
    const int subCol0 = blockCol0 + warpCol * WMMA_N;

    // --- Shared-memory staging tiles (padded) -------------------------------
    // As: [BM][BK] slice of A for this K-step; Bs: [BK][BN] slice of B.
    __shared__ half As[BM][BK + PAD];
    __shared__ half Bs[BK][BN + PAD];

    // FP32 accumulator for this warp's sub-tile.
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    const int nThreads = blockDim.x * blockDim.y;   // 128 * 4 = 512

    // --- Main loop over K in BK-wide slices ---------------------------------
    for (int k0 = 0; k0 < K; k0 += BK) {
        // Cooperatively load A's BM x BK slice into As.
        // Element (r, c) of As maps to A[blockRow0 + r, k0 + c].
        for (int idx = threadId; idx < BM * BK; idx += nThreads) {
            int r = idx / BK;
            int c = idx % BK;
            As[r][c] = A[(blockRow0 + r) * lda + (k0 + c)];
        }
        // Cooperatively load B's BK x BN slice into Bs.
        // Element (r, c) of Bs maps to B[k0 + r, blockCol0 + c].
        for (int idx = threadId; idx < BK * BN; idx += nThreads) {
            int r = idx / BN;
            int c = idx % BN;
            Bs[r][c] = B[(k0 + r) * ldb + (blockCol0 + c)];
        }

        // Make the staged tiles visible to all warps before any fragment load.
        __syncthreads();

        // Each warp loads its fragments FROM SHARED MEMORY and accumulates.
        // As sub-tile for this warp: rows [warpRow*16 .. +16), all BK cols.
        // Bs sub-tile for this warp: all BK rows, cols [warpCol*16 .. +16).
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;

        const half* as_ptr = &As[warpRow * WMMA_M][0];   // leading dim BK+PAD
        const half* bs_ptr = &Bs[0][warpCol * WMMA_N];   // leading dim BN+PAD
        wmma::load_matrix_sync(a_frag, as_ptr, BK + PAD);
        wmma::load_matrix_sync(b_frag, bs_ptr, BN + PAD);

        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

        // Guard the shared tiles before the next iteration overwrites them.
        __syncthreads();
    }

    // --- FP32 -> FP16 and store this warp's sub-tile ------------------------
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;
    for (int i = 0; i < acc_frag.num_elements; ++i) {
        c_frag.x[i] = __float2half(acc_frag.x[i]);
    }
    half* c_tile = C + subRow0 * ldc + subCol0;
    wmma::store_matrix_sync(c_tile, c_frag, ldc, wmma::mem_row_major);
}

// ---------------------------------------------------------------------------
// C-compatible launcher. Validates preconditions; returns an error code
// instead of launching on unsupported shapes. Device pointers in; caller
// handles synchronization/timing.
// ---------------------------------------------------------------------------
extern "C" int wmma_matmul_shared_launch(const half* A, const half* B, half* C,
                                         int M, int N, int K) {
    if (A == nullptr || B == nullptr || C == nullptr) return WMMA_ERR_NULLPTR;
    if (M <= 0 || N <= 0 || K <= 0)                   return WMMA_ERR_DIM;
    if ((M % WMMA_M) != 0 || (N % WMMA_N) != 0 || (K % WMMA_K) != 0)
        return WMMA_ERR_ALIGN;
    // Stricter requirement for the cooperative 64x64 block tile: M, N, K must
    // be multiples of the block tile sizes so no partial block is launched.
    if ((M % BM) != 0 || (N % BN) != 0 || (K % BK) != 0)
        return WMMA_ERR_BLOCK_ALIGN;

    // 512 threads/block (16 warps): blockDim.x=128 (4 warps), blockDim.y=4.
    dim3 block(128, 4, 1);
    dim3 grid(N / BN, M / BM, 1);

    wmma_matmul_shared_kernel<<<grid, block>>>(A, B, C, M, N, K);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "wmma_matmul_shared_kernel launch failed: %s\n",
                cudaGetErrorString(err));
        return (int)err > 0 ? -(int)err : (int)err;
    }
    return WMMA_OK;
}
