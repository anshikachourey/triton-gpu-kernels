// wmma_matmul.cu
//
// Minimal, NAIVE CUDA C++ FP16 matrix multiply using NVIDIA WMMA Tensor Core
// operations. This is an EXTENSION to the Triton project, not a replacement,
// and it is deliberately NOT optimized.
//
// Design (naive on purpose):
//   * One warp computes exactly one 16x16 output tile of C.
//   * The A and B fragments are loaded straight from GLOBAL memory each K-step
//     (no shared-memory staging, no double buffering, no pipelining).
//   * FP16 inputs, FP32 accumulation, FP16 output (to match the Triton and
//     torch.matmul paths in this project).
//   * Square, tile-aligned workloads only to start (M, N, K multiples of 16).
//
// Build (Tesla T4, Turing):
//   nvcc -arch=sm_75 -Xcompiler -fPIC -shared wmma_matmul.cu -o libwmma_matmul.so
//
// The library exposes a single extern "C" entry point so Python can call it via
// ctypes using tensor.data_ptr(); the SAME torch tensors can then feed the
// CUDA, Triton, and PyTorch paths.
//
// WMMA tile shape: 16 x 16 x 16 (M x N x K), the supported FP16 shape on sm_75.

#include <mma.h>
#include <cuda_fp16.h>
#include <cstdio>

using namespace nvcuda;

// WMMA fragment tile dimensions (supported FP16 shape on Turing).
static const int WMMA_M = 16;
static const int WMMA_N = 16;
static const int WMMA_K = 16;

// Error codes returned by the host wrapper (checked in Python).
//   0  = launched OK
//  -1  = null pointer
//  -2  = non-positive dimension
//  -3  = dimension not a multiple of the 16x16x16 WMMA tile
#define WMMA_OK            0
#define WMMA_ERR_NULLPTR  -1
#define WMMA_ERR_DIM      -2
#define WMMA_ERR_ALIGN    -3

// ---------------------------------------------------------------------------
// Kernel: one warp -> one 16x16 output tile of C = A @ B.
//
// A is row-major [M, K], B is row-major [K, N], C is row-major [M, N], all
// contiguous. lda = K, ldb = N, ldc = N for row-major contiguous storage.
// ---------------------------------------------------------------------------
__global__ void wmma_matmul_kernel(const half* __restrict__ A,
                                   const half* __restrict__ B,
                                   half* __restrict__ C,
                                   int M, int N, int K) {
    // --- Warp-to-tile mapping ------------------------------------------------
    // Each warp (32 threads) owns one 16x16 output tile. We lay the grid/block
    // out so that one warp maps to one (tileRow, tileCol).
    //   * threadIdx.x spans warps along N within the block: warpId = threadIdx.x / 32
    //   * threadIdx.y spans warps along M within the block.
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y);          // tile row index
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize; // tile col index

    // Top-left element of this warp's output tile in C.
    int tileRow = warpM * WMMA_M;   // row in [0, M)
    int tileCol = warpN * WMMA_N;   // col in [0, N)

    // Guard: skip warps whose tile is fully out of range. (For tile-aligned
    // square inputs this only trims the padded block edge.)
    if (tileRow >= M || tileCol >= N) return;

    // --- Fragment declarations ----------------------------------------------
    // a_frag: a 16x16 FP16 tile of A, row-major.
    // b_frag: a 16x16 FP16 tile of B, row-major.
    // acc_frag: the 16x16 FP32 accumulator for this output tile.
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

    // Start the accumulator at zero.
    wmma::fill_fragment(acc_frag, 0.0f);

    // --- Accumulate over the K dimension in 16-wide steps -------------------
    // Row-major leading dimensions.
    int lda = K;   // stride between rows of A
    int ldb = N;   // stride between rows of B
    int ldc = N;   // stride between rows of C

    for (int k0 = 0; k0 < K; k0 += WMMA_K) {
        // A tile starts at row=tileRow, col=k0  ->  A[tileRow*lda + k0]
        const half* a_tile = A + tileRow * lda + k0;
        // B tile starts at row=k0, col=tileCol  ->  B[k0*ldb + tileCol]
        const half* b_tile = B + k0 * ldb + tileCol;

        // Load the two 16x16 FP16 fragments straight from global memory.
        wmma::load_matrix_sync(a_frag, a_tile, lda);
        wmma::load_matrix_sync(b_frag, b_tile, ldb);

        // Tensor Core multiply-accumulate: acc += a_frag * b_frag (FP32 accum).
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    // --- Convert FP32 accumulator -> FP16 and store -------------------------
    // store_matrix_sync needs a fragment whose element type matches the output
    // pointer (half). We copy the FP32 accumulator element-wise into a half
    // accumulator fragment, then store that. (num_elements / x[] are the
    // documented per-thread fragment storage.)
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;
    for (int i = 0; i < acc_frag.num_elements; ++i) {
        c_frag.x[i] = __float2half(acc_frag.x[i]);
    }

    half* c_tile = C + tileRow * ldc + tileCol;
    wmma::store_matrix_sync(c_tile, c_frag, ldc, wmma::mem_row_major);
}

// ---------------------------------------------------------------------------
// Host wrapper: validate preconditions, configure the launch, run the kernel.
// Returns an error code INSTEAD of launching on unsupported shapes.
//
// Pointers are DEVICE pointers (from torch tensor.data_ptr()). The caller is
// responsible for synchronization/timing around this call.
// ---------------------------------------------------------------------------
extern "C" int wmma_matmul_launch(const half* A, const half* B, half* C,
                                  int M, int N, int K) {
    // Precondition checks (return codes, no launch on bad input).
    if (A == nullptr || B == nullptr || C == nullptr) return WMMA_ERR_NULLPTR;
    if (M <= 0 || N <= 0 || K <= 0)                   return WMMA_ERR_DIM;
    if ((M % WMMA_M) != 0 || (N % WMMA_N) != 0 || (K % WMMA_K) != 0)
        return WMMA_ERR_ALIGN;

    // One warp per 16x16 output tile. Block holds 4x4 = 16 warps:
    //   blockDim.x = 128 (= 4 warps along N), blockDim.y = 4 (warps along M).
    dim3 block(128, 4, 1);
    int warpsPerBlockN = block.x / 32;   // 4
    int warpsPerBlockM = block.y;        // 4
    dim3 grid((N / WMMA_N + warpsPerBlockN - 1) / warpsPerBlockN,
              (M / WMMA_M + warpsPerBlockM - 1) / warpsPerBlockM,
              1);

    wmma_matmul_kernel<<<grid, block>>>(A, B, C, M, N, K);

    // Report a launch error if the configuration was rejected by the driver.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "wmma_matmul_kernel launch failed: %s\n",
                cudaGetErrorString(err));
        return (int)err > 0 ? -(int)err : (int)err;  // negative = error
    }
    return WMMA_OK;
}
