/**
 * gemv_kernel.cu — CUTLASS GEMV kernel wrapper (isolated from PyTorch headers)
 *
 * Provides two APIs:
 *   launch_gemv_sorted   — optimized: sorted I/O, constant memory work table, no scatter/gather
 *   launch_gemv_adaptive — legacy: strided batching with scatter/gather (for correctness comparison)
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>

#include "cutlass/numeric_types.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/array.h"
#include "cutlass/gemm/device/gemv.h"
#include "cutlass/gemm/kernel/gemv.h"

using CutlassElementA  = cutlass::int4b_t;
using CutlassElementB  = cutlass::float_e4m3_t;
using CutlassElementC  = __nv_bfloat16;
using CutlassElementSF = cutlass::half_t;
using CutlassAcc       = float;

static constexpr int kEPA = 128 / cutlass::sizeof_bits<CutlassElementB>::value;
static constexpr int kTC  = 128;
static constexpr int kTPR = 8;
static constexpr int kSKS = 1;
static constexpr int kSFV = 128;

using CutlassLayoutA = cutlass::layout::RowMajor;
using CutlassEpilogue = cutlass::epilogue::thread::LinearCombination<
    CutlassElementC, 4, CutlassAcc, CutlassAcc>;

// Two kernel specializations: strided I/O (false) and sorted I/O (true)
using CutlassGemvKernelStrided = cutlass::gemm::kernel::Gemv<
    CutlassElementA, CutlassLayoutA, CutlassElementB, CutlassElementC,
    CutlassAcc, CutlassEpilogue,
    kEPA, kTC, kTPR, kSKS, CutlassElementSF, kSFV, /*kSortedIO=*/false>;

using CutlassGemvKernelSorted = cutlass::gemm::kernel::Gemv<
    CutlassElementA, CutlassLayoutA, CutlassElementB, CutlassElementC,
    CutlassAcc, CutlassEpilogue,
    kEPA, kTC, kTPR, kSKS, CutlassElementSF, kSFV, /*kSortedIO=*/true>;

using CutlassGemvDeviceStrided = cutlass::gemm::device::Gemv<CutlassGemvKernelStrided>;
using CutlassGemvDeviceSorted  = cutlass::gemm::device::Gemv<CutlassGemvKernelSorted>;
using ExpertWork = cutlass::gemm::kernel::ExpertWork;

// ============================================================================
// Static device buffers (allocated once, reused across calls)
// ============================================================================
static ExpertWork *d_work_table_static = nullptr;
static int32_t    *d_metadata_buf      = nullptr;  // device
static int32_t    *h_metadata_pinned   = nullptr;  // host pinned
static cudaStream_t build_stream       = nullptr;
static cudaEvent_t  build_done_event   = nullptr;

static void ensure_work_table_buf() {
    if (d_work_table_static == nullptr) {
        cudaMalloc(&d_work_table_static, 256 * sizeof(ExpertWork));
    }
    if (d_metadata_buf == nullptr) {
        cudaMalloc(&d_metadata_buf, 4 * sizeof(int32_t));
    }
    if (h_metadata_pinned == nullptr) {
        cudaHostAlloc(&h_metadata_pinned, 4 * sizeof(int32_t), cudaHostAllocDefault);
    }
    if (build_stream == nullptr) {
        cudaStreamCreate(&build_stream);
    }
    if (build_done_event == nullptr) {
        cudaEventCreate(&build_done_event);
    }
}

// ============================================================================
// GPU-side work table builder (replaces Python build_work_table)
// ============================================================================
__global__ void build_work_table_kernel(
    const int32_t* __restrict__ n_per_expert,
    ExpertWork* __restrict__ work_table,
    int32_t* __restrict__ metadata,
    int E, int M_tiles, float threshold)
{
    int max_n = 0;
    int num_active = 0;
    int cta_offset = 0;

    for (int i = 0; i < E; i++) {
        int n = n_per_expert[i];
        if (n > max_n) max_n = n;
        if (n > 0) {
            int nt = (n + 7) / 8;
            work_table[num_active].expert_id  = i;
            work_table[num_active].n_tiles    = nt;
            work_table[num_active].cta_start  = cta_offset;
            num_active++;
            cta_offset += M_tiles * nt;
        }
    }

    int orig_ctas = M_tiles * ((max_n + 7) / 8) * E;
    metadata[0] = num_active;
    metadata[1] = cta_offset;  // total_flat_ctas
    metadata[2] = max_n;
    metadata[3] = (orig_ctas > 0 && cta_offset < (int)(orig_ctas * threshold)) ? 1 : 0;
}


// ============================================================================
// Scatter/Gather kernels (kept for legacy API)
// ============================================================================

__global__ void scatter_to_strided_kernel(
    const void *sorted, void *strided,
    const int32_t *expert_offsets, const int32_t *n_per_expert,
    int E, int K, int max_N)
{
    int expert = blockIdx.y;
    int n = n_per_expert[expert];
    int tok = blockIdx.x;
    if (tok >= n) return;

    int src_off = expert_offsets[expert] + tok;
    int dst_off = expert * max_N + tok;

    const uint4 *src = (const uint4*)((const char*)sorted + (int64_t)src_off * K);
    uint4 *dst = (uint4*)((char*)strided + (int64_t)dst_off * K);
    for (int i = threadIdx.x; i < K / 16; i += blockDim.x)
        dst[i] = src[i];
}

__global__ void gather_from_strided_kernel(
    const void *strided, void *sorted,
    const int32_t *expert_offsets, const int32_t *n_per_expert,
    int E, int M, int max_N)
{
    int expert = blockIdx.y;
    int n = n_per_expert[expert];
    int tok = blockIdx.x;
    if (tok >= n) return;

    int dst_row = expert_offsets[expert] + tok;
    const __nv_bfloat16 *src_base = (const __nv_bfloat16*)strided
        + (int64_t)expert * M * max_N + (int64_t)tok * M;
    __nv_bfloat16 *dst_base = (__nv_bfloat16*)sorted + (int64_t)dst_row * M;

    const uint2 *src = (const uint2*)src_base;
    uint2 *dst = (uint2*)dst_base;
    for (int i = threadIdx.x; i < M / 4; i += blockDim.x)
        dst[i] = src[i];
}


// ============================================================================
// C-linkage API
// ============================================================================

extern "C" {

// ---- Upload work table (call once per routing change) ----
void upload_work_table(void *work_table_host, int num_active, cudaStream_t stream)
{
    ensure_work_table_buf();
    if (num_active > 0) {
        cudaMemcpyAsync(d_work_table_static, work_table_host,
                        num_active * sizeof(ExpertWork),
                        cudaMemcpyHostToDevice, stream);
    }
}

// ---- GPU-side work table build (replaces Python build_work_table + upload) ----
// Uses a dedicated stream + pinned memory so D2H is truly async (no implicit sync).
void build_work_table_cuda(
    int32_t *n_per_expert,   // device [E]
    int E, int M_tiles, float threshold,
    int32_t *metadata_out,   // host [4]: {num_active, total_flat_ctas, max_N, use_flat}
    cudaStream_t main_stream)
{
    ensure_work_table_buf();

    // Launch build + D2H on dedicated stream with pinned memory
    build_work_table_kernel<<<1, 1, 0, build_stream>>>(
        n_per_expert, d_work_table_static, d_metadata_buf,
        E, M_tiles, threshold);
    cudaMemcpyAsync(h_metadata_pinned, d_metadata_buf, 4 * sizeof(int32_t),
                    cudaMemcpyDeviceToHost, build_stream);
    cudaStreamSynchronize(build_stream);

    // Copy pinned → caller's buffer
    metadata_out[0] = h_metadata_pinned[0];
    metadata_out[1] = h_metadata_pinned[1];
    metadata_out[2] = h_metadata_pinned[2];
    metadata_out[3] = h_metadata_pinned[3];

    // Main stream must wait for work table to be ready before GEMV kernel reads it
    cudaEventRecord(build_done_event, build_stream);
    cudaStreamWaitEvent(main_stream, build_done_event, 0);
}

// ---- Optimized: sorted I/O, no scatter/gather, no sync ----
// Dual path: total_flat_ctas > 0 → 1D flat grid (skew)
//            total_flat_ctas = 0 → 3D grid (uniform), needs max_N
void launch_gemv_sorted(
    void *d_sorted_out,      // [total, M] bf16 output
    void *w_interleaved,     // [E, M, K/2] int8
    void *act_sorted,        // [total, K] fp8 input
    void *scales,            // [E, M, padded_sfs] fp16
    int32_t *expert_offsets, // [E+1] device ptr
    int32_t *n_per_expert,   // [E] device ptr
    float alpha,
    int M, int K, int E,
    int max_N,               // needed for 3D grid (when total_flat_ctas=0)
    int num_active,
    int M_tiles,
    int total_flat_ctas,     // 0 = 3D grid, >0 = 1D flat
    cudaStream_t stream)
{
    // Build args — sorted I/O mode (expert_offsets always set)
    CutlassGemvDeviceSorted::Arguments args{
        (int32_t)M, n_per_expert, (int32_t)K, (int32_t)max_N,
        (int32_t)E,
        {alpha, 0.0f},
        {(CutlassElementA*)w_interleaved, K},
        act_sorted,       // B = sorted activations directly
        d_sorted_out,     // C = output (unused with beta=0)
        d_sorted_out,     // D = sorted output directly
        (int64_t)M * K,   // batch_stride_A (weights, still batched)
        0,                // batch_stride_B (unused in sorted mode)
        0,                // batch_stride_C (unused)
        0,                // batch_stride_D (unused)
        (CutlassElementSF*)scales,
    };

    // Set sorted I/O
    args.expert_offsets = expert_offsets;

    // Set flat mode fields (only used when total_flat_ctas > 0)
    if (total_flat_ctas > 0) {
        args.work_table = d_work_table_static;
        args.num_active = num_active;
        args.M_tiles = M_tiles;
        args.total_flat_ctas = total_flat_ctas;
    }

    // Launch — sorted specialization, no scatter/gather
    CutlassGemvDeviceSorted gemv;
    gemv.initialize(args);
    gemv.run(stream);
}


// ---- Fused: C++ host build work table + async H2D + launch GEMV ----
// Zero sync. CPU builds work table (~2us), async H2D (~1us), then launches GEMV.
// h_n_per_expert is a HOST pointer (caller is responsible for D2H if needed).
void build_and_launch_gemv_sorted(
    const int32_t *h_n_per_expert, // host [E] — tokens per expert
    void *d_sorted_out,
    void *w_interleaved,
    void *act_sorted,
    void *scales,
    int32_t *expert_offsets,       // device [E+1]
    int32_t *d_n_per_expert,       // device [E] (for GEMV kernel)
    float alpha,
    int M, int K, int E,
    float threshold,
    cudaStream_t stream)
{
    ensure_work_table_buf();

    int block_y = kTC / kTPR;
    int M_tiles = (M / 4 + block_y - 1) / block_y;

    // 1. CPU build work table (~2us for E=256)
    ExpertWork h_table[256];
    int max_n = 0, num_active = 0, cta_offset = 0;
    for (int i = 0; i < E; i++) {
        int n = h_n_per_expert[i];
        if (n > max_n) max_n = n;
        if (n > 0) {
            int nt = (n + 7) / 8;
            h_table[num_active].expert_id  = i;
            h_table[num_active].n_tiles    = nt;
            h_table[num_active].cta_start  = cta_offset;
            num_active++;
            cta_offset += M_tiles * nt;
        }
    }
    int total_flat_ctas = cta_offset;
    int orig_ctas = M_tiles * ((max_n + 7) / 8) * E;
    bool use_flat = (orig_ctas > 0 && total_flat_ctas < (int)(orig_ctas * threshold));

    // 2. Async H2D work table (~1us for ~1KB) — no sync needed
    if (use_flat && num_active > 0) {
        cudaMemcpyAsync(d_work_table_static, h_table,
                        num_active * sizeof(ExpertWork),
                        cudaMemcpyHostToDevice, stream);
    }

    // 3. Launch GEMV immediately on same stream (ordered after H2D)
    CutlassGemvDeviceSorted::Arguments args{
        (int32_t)M, d_n_per_expert, (int32_t)K, (int32_t)max_n,
        (int32_t)E,
        {alpha, 0.0f},
        {(CutlassElementA*)w_interleaved, K},
        act_sorted, d_sorted_out, d_sorted_out,
        (int64_t)M * K, 0, 0, 0,
        (CutlassElementSF*)scales,
    };
    args.expert_offsets = expert_offsets;

    if (use_flat) {
        args.work_table = d_work_table_static;
        args.num_active = num_active;
        args.M_tiles = M_tiles;
        args.total_flat_ctas = total_flat_ctas;
    }

    CutlassGemvDeviceSorted gemv;
    gemv.initialize(args);
    gemv.run(stream);
}

// ---- Strided I/O: scatter + 3D kernel + gather, no D2H sync ----
// Best for uniform routing (no expert_offsets overhead in kernel)
void launch_gemv_strided(
    void *d_sorted_out,      // [total, M] bf16 output
    void *w_interleaved,     // [E, M, K/2] int8
    void *act_sorted,        // [total, K] fp8 input
    void *scales,            // [E, M, padded_sfs] fp16
    int32_t *expert_offsets, // [E+1] device ptr
    int32_t *n_per_expert,   // [E] device ptr
    float alpha,
    int M, int K, int max_N, int E,
    void *B_strided_buf,     // [E, max_N, K] fp8 pre-allocated
    void *D_strided_buf,     // [E, M, max_N] bf16 pre-allocated
    cudaStream_t stream)
{
    // --- Scatter: sorted → strided ---
    {
        dim3 grid(max_N, E);
        int bk = K / 16;
        dim3 block(bk > 256 ? 256 : (bk > 0 ? bk : 1));
        scatter_to_strided_kernel<<<grid, block, 0, stream>>>(
            act_sorted, B_strided_buf,
            expert_offsets, n_per_expert, E, K, max_N);
    }

    // --- 3D GEMV kernel (strided specialization, no expert_offsets code) ---
    CutlassGemvDeviceStrided::Arguments args{
        (int32_t)M, n_per_expert, (int32_t)K, (int32_t)max_N,
        (int32_t)E,
        {alpha, 0.0f},
        {(CutlassElementA*)w_interleaved, K},
        B_strided_buf, D_strided_buf, D_strided_buf,
        (int64_t)M * K, (int64_t)max_N * K,
        (int64_t)M * max_N, (int64_t)M * max_N,
        (CutlassElementSF*)scales,
    };
    // total_flat_ctas = 0 → 3D grid (default)

    CutlassGemvDeviceStrided gemv;
    gemv.initialize(args);
    gemv.run(stream);

    // --- Gather: strided → sorted ---
    {
        dim3 grid(max_N, E);
        int bm = M / 4;
        dim3 block(bm > 256 ? 256 : (bm > 0 ? bm : 1));
        gather_from_strided_kernel<<<grid, block, 0, stream>>>(
            D_strided_buf, d_sorted_out,
            expert_offsets, n_per_expert, E, M, max_N);
    }
}


// ---- Legacy: strided batching with scatter/gather ----
void launch_gemv_adaptive(
    void *d_sorted_out,    // [total, M] bf16
    void *w_interleaved,   // [E, M, K/2] int8
    void *act_sorted,      // [total, K] fp8
    void *scales,          // [E, M, padded_sfs] fp16
    int32_t *expert_offsets, // [E+1]
    int32_t *n_per_expert,   // [E]
    float alpha,
    int M, int K, int max_N, int E,
    void *B_strided_buf,   // [E, max_N, K] fp8 temp
    void *D_strided_buf,   // [E, M, max_N] bf16 temp
    cudaStream_t stream)
{
    int block_y = kTC / kTPR;
    int M_tiles = (M / 4 + block_y - 1) / block_y;

    // --- Scatter ---
    {
        dim3 grid(max_N, E);
        int bk = K / 16;
        dim3 block(bk > 256 ? 256 : (bk > 0 ? bk : 1));
        scatter_to_strided_kernel<<<grid, block, 0, stream>>>(
            act_sorted, B_strided_buf,
            expert_offsets, n_per_expert, E, K, max_N);
    }

    // --- Compute dispatch decision ---
    int32_t h_npe[4096];
    cudaMemcpyAsync(h_npe, n_per_expert, E * sizeof(int32_t),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    ExpertWork h_table[4096];
    int num_active = 0;
    int cta_offset = 0;
    for (int e = 0; e < E; e++) {
        if (h_npe[e] > 0) {
            h_table[num_active].expert_id = e;
            h_table[num_active].n_tiles = (h_npe[e] + 7) / 8;
            h_table[num_active].cta_start = cta_offset;
            cta_offset += M_tiles * h_table[num_active].n_tiles;
            num_active++;
        }
    }
    int flat_ctas = cta_offset;
    int orig_ctas = M_tiles * ((max_N + 7) / 8) * E;
    bool use_flat = (flat_ctas < (int)(orig_ctas * 0.7));

    // --- Build args ---
    CutlassGemvDeviceStrided::Arguments args{
        (int32_t)M, n_per_expert, (int32_t)K, (int32_t)max_N,
        (int32_t)E,
        {alpha, 0.0f},
        {(CutlassElementA*)w_interleaved, K},
        B_strided_buf, D_strided_buf, D_strided_buf,
        (int64_t)M * K, (int64_t)max_N * K,
        (int64_t)M * max_N, (int64_t)M * max_N,
        (CutlassElementSF*)scales,
    };

    if (use_flat && num_active > 0) {
        ensure_work_table_buf();
        cudaMemcpyAsync(d_work_table_static, h_table,
                        num_active * sizeof(ExpertWork),
                        cudaMemcpyHostToDevice, stream);
        args.work_table = d_work_table_static;
        args.num_active = num_active;
        args.M_tiles = M_tiles;
        args.total_flat_ctas = flat_ctas;
    }

    // --- Launch ---
    CutlassGemvDeviceStrided gemv;
    gemv.initialize(args);
    gemv.run(stream);

    // --- Gather ---
    {
        dim3 grid(max_N, E);
        int bm = M / 4;
        dim3 block(bm > 256 ? 256 : (bm > 0 ? bm : 1));
        gather_from_strided_kernel<<<grid, block, 0, stream>>>(
            D_strided_buf, d_sorted_out,
            expert_offsets, n_per_expert, E, M, max_N);
    }
}


void launch_preprocess_weights(
    void *w_interleaved, void *s_padded,
    void *w_orig, void *s_orig,
    int E, int M, int K, cudaStream_t stream)
{
    CutlassGemvKernelStrided::matrix_A_interleave(
        (CutlassElementA*)w_interleaved, (CutlassElementA*)w_orig,
        (CutlassElementSF*)s_padded, (CutlassElementSF*)s_orig,
        E, M, K, stream);
    cudaStreamSynchronize(stream);
}

}  // extern "C"
