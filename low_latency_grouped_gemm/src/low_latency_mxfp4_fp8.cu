// SPDX-License-Identifier: Apache-2.0
#include "low_latency_grouped_gemm/include/low_latency_mxfp4_fp8.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "low_latency_grouped_gemm/include/low_latency_grouped_gemm.h"
#include "low_latency_grouped_gemm/include/low_latency_mxfp4_fp8_kernel.cuh"

namespace mga {
namespace {

using Kernel =
    low_latency_mxfp4_fp8_detail::LowLatencyMxfp4Fp8Kernel;
using Params = Kernel::Params;

[[noreturn]] void abort_mxfp4(const char* msg) {
    std::fprintf(stderr,
                 "[low_latency_mxfp4_fp8] invalid launch options: %s\n",
                 msg);
    std::abort();
}

void check_cuda(cudaError_t error, const char* msg) {
    if (error != cudaSuccess) {
        std::fprintf(stderr,
                     "[low_latency_mxfp4_fp8] %s: %s\n",
                     msg,
                     cudaGetErrorString(error));
        std::abort();
    }
}

void validate_shape(int G, int N_orig, int K) {
    if (G < 0) abort_mxfp4("G must be non-negative");
    if (N_orig <= 0 || N_orig % 64 != 0) {
        abort_mxfp4("N_orig must be a positive multiple of 64");
    }
    if (!Kernel::can_implement(K)) {
        abort_mxfp4("K must be a positive multiple of 64");
    }
}

__constant__ uint8_t kHummingRewriteLut[256 * 16];

uint32_t float_bits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

float float_from_bits(uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

uint8_t quantize_e2m1_like_humming(double value) {
    const float rounded_input = static_cast<float>(value);
    const uint32_t bits = float_bits(rounded_input);
    constexpr uint32_t kMask = 0x81c00000U;
    const uint32_t rz_bits = bits & kMask;
    const uint32_t ru_bits = (bits + 0x00200000U) & kMask;
    const double rz = static_cast<double>(float_from_bits(rz_bits));
    const double ru = static_cast<double>(float_from_bits(ru_bits));
    const uint32_t rounded =
        std::fabs(value - rz) >= std::fabs(value - ru) ? ru_bits : rz_bits;
    return static_cast<uint8_t>(
        ((rounded & 0x80000000U) >> 28U) |
        ((rounded & 0x01c00000U) >> 22U));
}

const std::array<uint8_t, 256 * 16>& humming_rewrite_lut() {
    static const std::array<uint8_t, 256 * 16> lut = [] {
        std::array<uint8_t, 256 * 16> result{};
        for (uint32_t delta = 0; delta < 256; ++delta) {
            const uint32_t scale_bits =
                0x3f800000U - (delta << 23U);
            const double scale =
                static_cast<double>(float_from_bits(scale_bits));
            for (uint32_t code = 0; code < 16; ++code) {
                uint8_t normalized =
                    static_cast<uint8_t>(code == 8 ? 0 : code);
                if (delta != 0) {
                    const uint32_t value_bits =
                        ((normalized & 0x8U) << 28U) |
                        ((normalized & 0x7U) << 22U);
                    const double value =
                        static_cast<double>(float_from_bits(value_bits)) *
                        scale;
                    normalized = quantize_e2m1_like_humming(value);
                }
                result[delta * 16 + code] = normalized;
            }
        }
        return result;
    }();
    return lut;
}

__global__ void preprocess_scales_kernel(
    const uint8_t* raw,
    uint8_t* exp_offsets,
    uint8_t* delta_offsets,
    float* residual,
    size_t scales_per_expert) {
    const int g = blockIdx.x;
    int local_min = 255;
    int local_max = 0;
    const uint8_t* raw_g =
        raw + static_cast<size_t>(g) * scales_per_expert;

    for (size_t i = threadIdx.x; i < scales_per_expert; i += blockDim.x) {
        const int value = raw_g[i];
        local_min = value < local_min ? value : local_min;
        local_max = value > local_max ? value : local_max;
    }

    __shared__ int mins[256];
    __shared__ int maxs[256];
    __shared__ int floor_exp;
    mins[threadIdx.x] = local_min;
    maxs[threadIdx.x] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            mins[threadIdx.x] =
                mins[threadIdx.x] < mins[threadIdx.x + stride]
                    ? mins[threadIdx.x]
                    : mins[threadIdx.x + stride];
            maxs[threadIdx.x] =
                maxs[threadIdx.x] > maxs[threadIdx.x + stride]
                    ? maxs[threadIdx.x]
                    : maxs[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const int range = min(maxs[0] - mins[0], 11);
        floor_exp = maxs[0] - range;
        residual[g] = exp2f(static_cast<float>(floor_exp - 128));
    }
    __syncthreads();

    uint8_t* exp_g =
        exp_offsets + static_cast<size_t>(g) * scales_per_expert;
    uint8_t* delta_g =
        delta_offsets + static_cast<size_t>(g) * scales_per_expert;
    for (size_t i = threadIdx.x; i < scales_per_expert; i += blockDim.x) {
        const int value = raw_g[i];
        const int clamped = value > floor_exp ? value : floor_exp;
        delta_g[i] = static_cast<uint8_t>(clamped - value);
        exp_g[i] = static_cast<uint8_t>((clamped - floor_exp + 1) & 0xf);
    }
}

__global__ void rewrite_payload_kernel(
    const uint8_t* raw_weight,
    const uint8_t* delta_offsets,
    uint8_t* processed_weight,
    size_t bytes_per_expert,
    size_t scales_per_expert) {
    const int g = blockIdx.y;
    const size_t expert_base = static_cast<size_t>(g) * bytes_per_expert;
    for (size_t byte_idx =
             static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         byte_idx < bytes_per_expert;
         byte_idx += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const uint8_t packed = raw_weight[expert_base + byte_idx];
        const uint8_t delta =
            delta_offsets[static_cast<size_t>(g) * scales_per_expert +
                          byte_idx / 16];
        const uint8_t lo =
            kHummingRewriteLut[static_cast<int>(delta) * 16 +
                               (packed & 0xf)];
        const uint8_t hi =
            kHummingRewriteLut[static_cast<int>(delta) * 16 +
                               (packed >> 4)];
        processed_weight[expert_base + byte_idx] =
            static_cast<uint8_t>(lo | (hi << 4));
    }
}

__device__ __forceinline__ uint32_t
preprocess_fp4x8_signs_for_fp8(uint32_t fp4x8) {
    const uint32_t em = fp4x8 & 0x77777777U;
    const uint32_t signs =
        ((fp4x8 & 0x00000008U) << 4U) |
        ((fp4x8 & 0x00000080U) << 8U) |
        ((fp4x8 & 0x00000800U) << 12U) |
        ((fp4x8 & 0x00008000U) << 16U) |
        ((fp4x8 & 0x00080000U) >> 16U) |
        ((fp4x8 & 0x00800000U) >> 12U) |
        ((fp4x8 & 0x08000000U) >> 8U) |
        ((fp4x8 & 0x80000000U) >> 4U);
    return em | signs;
}

__global__ void interleave_kernel(
    const uint8_t* processed_weight,
    const uint8_t* exp_offsets_logical,
    uint8_t* w_interleaved,
    uint8_t* exp_offsets_interleaved,
    int G,
    int N,
    int K) {
    const int tid = threadIdx.x;
    for (int g = blockIdx.y; g < G; g += gridDim.y) {
        for (int row_tile = blockIdx.x;
             row_tile < N / 64;
             row_tile += gridDim.x) {
            const int t0 = tid & 3;
            const int t1 = (tid >> 2) & 7;
            const int t2 = tid >> 5;
            const int row0 = row_tile * 64 + t1 + t2 * 16;
            const int row1 = row0 + 8;
            const size_t weight_expert_base =
                static_cast<size_t>(g) * N * K / 2;
            const size_t scale_expert_base =
                static_cast<size_t>(g) * N * K / 32;
            const int k32_count = K / 32;

            for (int k32_idx = 0; k32_idx < k32_count; ++k32_idx) {
                const int k_base = k32_idx * 32 + t0 * 4;
                const uint8_t* row0_ptr =
                    processed_weight + weight_expert_base +
                    static_cast<size_t>(row0) * K / 2;
                const uint8_t* row1_ptr =
                    processed_weight + weight_expert_base +
                    static_cast<size_t>(row1) * K / 2;
                const uint32_t row0_logical =
                    static_cast<uint32_t>(
                        *reinterpret_cast<const uint16_t*>(
                            row0_ptr + k_base / 2)) |
                    (static_cast<uint32_t>(
                         *reinterpret_cast<const uint16_t*>(
                             row0_ptr + k_base / 2 + 8))
                     << 16U);
                const uint32_t row1_logical =
                    static_cast<uint32_t>(
                        *reinterpret_cast<const uint16_t*>(
                            row1_ptr + k_base / 2)) |
                    (static_cast<uint32_t>(
                         *reinterpret_cast<const uint16_t*>(
                             row1_ptr + k_base / 2 + 8))
                     << 16U);
                const uint64_t physical =
                    static_cast<uint64_t>(preprocess_fp4x8_signs_for_fp8(
                        row0_logical)) |
                    (static_cast<uint64_t>(preprocess_fp4x8_signs_for_fp8(
                         row1_logical))
                     << 32U);
                const int tile64_count = N / 64;
                const int full_pair_count = tile64_count / 2;
                size_t dst = 0;
                if (row_tile < full_pair_count * 2) {
                    const int pair = row_tile / 2;
                    const int part = row_tile & 1;
                    dst = weight_expert_base +
                          (static_cast<size_t>(pair) * k32_count +
                           k32_idx) * 128 * 16 +
                          static_cast<size_t>(tid) * 16 + part * 8;
                } else {
                    dst = weight_expert_base +
                          static_cast<size_t>(full_pair_count) *
                              k32_count * 128 * 16 +
                          (static_cast<size_t>(k32_idx) * 128 + tid) * 8;
                }
                *reinterpret_cast<uint64_t*>(w_interleaved + dst) = physical;

                if (tid < 32) {
                    const int scale_group = tid;
                    const int scale_row0_local =
                        (scale_group & 7) + (scale_group >> 3) * 16;
                    const int scale_row1_local = scale_row0_local + 8;
                    const size_t logical_idx0 =
                        scale_expert_base +
                        static_cast<size_t>(row_tile * 64 +
                                            scale_row0_local) *
                            k32_count +
                        k32_idx;
                    const size_t logical_idx1 =
                        scale_expert_base +
                        static_cast<size_t>(row_tile * 64 +
                                            scale_row1_local) *
                            k32_count +
                        k32_idx;
                    size_t interleaved_idx = 0;
                    if (row_tile < full_pair_count * 2) {
                        const int pair = row_tile / 2;
                        const int part = row_tile & 1;
                        interleaved_idx =
                            scale_expert_base +
                            (static_cast<size_t>(pair) * k32_count +
                             k32_idx) * 128 +
                            static_cast<size_t>(scale_group) * 4 +
                            part * 2;
                    } else {
                        interleaved_idx =
                            scale_expert_base +
                            static_cast<size_t>(full_pair_count) *
                                k32_count * 128 +
                            static_cast<size_t>(k32_idx) * 64 +
                            static_cast<size_t>(scale_group) * 2;
                    }
                    exp_offsets_interleaved[interleaved_idx] =
                        exp_offsets_logical[logical_idx0];
                    exp_offsets_interleaved[interleaved_idx + 1] =
                        exp_offsets_logical[logical_idx1];
                }
            }
        }
    }
}

__global__ void prepare_deepep_layout_kernel(
    const int32_t* token_counts, int G, int capacity,
    int32_t* padded_offsets, int32_t* compact_offsets,
    int32_t* tile_experts, int32_t* tile_n, int32_t* num_token_tiles) {
    extern __shared__ int32_t scratch[];
    int32_t* compact = scratch;
    int32_t* tile_offsets = compact + G + 1;
    if (threadIdx.x == 0) {
        compact[0] = 0;
        tile_offsets[0] = 0;
        padded_offsets[0] = 0;
        compact_offsets[0] = 0;
        for (int expert = 0; expert < G; ++expert) {
            const int count = token_counts[expert];
            compact[expert + 1] = compact[expert] + count;
            tile_offsets[expert + 1] =
                tile_offsets[expert] + (count + 7) / 8;
            padded_offsets[expert + 1] = (expert + 1) * capacity;
            compact_offsets[expert + 1] = compact[expert + 1];
        }
        *num_token_tiles = tile_offsets[G];
    }
    __syncthreads();
    for (int expert = threadIdx.x; expert < G; expert += blockDim.x) {
        const int begin = tile_offsets[expert];
        const int end = tile_offsets[expert + 1];
        for (int tile = begin; tile < end; ++tile) {
            tile_experts[tile] = expert;
            tile_n[tile] = tile - begin;
        }
    }
}

__global__ void requantize_deepep_compact_kernel(
    const __nv_fp8_e4m3* input, const float* input_group_scales,
    const int32_t* token_counts, const int32_t* compact_offsets,
    int G, int capacity, int hidden_size, int64_t scale_stride_expert,
    int64_t scale_stride_token, int64_t scale_stride_group,
    __nv_fp8_e4m3* output, float* output_scales) {
    const int padded_row = blockIdx.x;
    const int expert = padded_row / capacity;
    const int local_row = padded_row - expert * capacity;
    if (expert >= G || local_row >= token_counts[expert]) return;

    const int groups = hidden_size / 128;
    const int64_t scale_base =
        static_cast<int64_t>(expert) * scale_stride_expert +
        static_cast<int64_t>(local_row) * scale_stride_token;
    float local_max_scale = 0.0f;
    for (int group = threadIdx.x; group < groups; group += blockDim.x) {
        local_max_scale = fmaxf(
            local_max_scale,
            input_group_scales[scale_base + group * scale_stride_group]);
    }
    const unsigned mask = __activemask();
    for (int delta = 16; delta > 0; delta >>= 1) {
        local_max_scale = fmaxf(
            local_max_scale,
            __shfl_down_sync(mask, local_max_scale, delta));
    }
    __shared__ float warp_maxima[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) warp_maxima[warp] = local_max_scale;
    __syncthreads();
    if (warp == 0) {
        float block_max = lane < 8 ? warp_maxima[lane] : 0.0f;
        for (int delta = 16; delta > 0; delta >>= 1) {
            block_max = fmaxf(
                block_max, __shfl_down_sync(0xffffffffu, block_max, delta));
        }
        if (lane == 0) warp_maxima[0] = block_max;
    }
    __syncthreads();

    const float token_scale = warp_maxima[0];
    const int compact_row = compact_offsets[expert] + local_row;
    if (threadIdx.x == 0) output_scales[compact_row] = token_scale;
    const size_t input_base = static_cast<size_t>(padded_row) * hidden_size;
    const size_t output_base = static_cast<size_t>(compact_row) * hidden_size;
    for (int column = threadIdx.x; column < hidden_size; column += blockDim.x) {
        const int group = column / 128;
        const float group_scale =
            input_group_scales[scale_base + group * scale_stride_group];
        const float ratio = token_scale == 0.0f ? 0.0f : group_scale / token_scale;
        const float value = static_cast<float>(input[input_base + column]) * ratio;
        output[output_base + column] = __nv_fp8_e4m3(
            fmaxf(fminf(value, 448.0f), -448.0f));
    }
}

__device__ __forceinline__ float situ_activate(
    float gate, float up, float beta, float linear_beta) {
    const float gate_out =
        beta * tanhf(gate / beta) / (1.0f + expf(-gate));
    const float up_out = linear_beta * tanhf(up / linear_beta);
    return gate_out * up_out;
}

__global__ void situ_quant_compact_kernel(
    const __nv_bfloat16* gate_up, const int32_t* compact_offsets, int G,
    int hidden_size, float beta, float linear_beta,
    __nv_fp8_e4m3* output, float* output_scales) {
    const int token = blockIdx.x;
    if (token >= compact_offsets[G]) return;
    constexpr int kValuesPerThread = 8;
    float values[kValuesPerThread];
    float local_max = 0.0f;
    const size_t input_base =
        static_cast<size_t>(token) * hidden_size * 2;
    const int column_base = threadIdx.x * kValuesPerThread;
#pragma unroll
    for (int i = 0; i < kValuesPerThread; ++i) {
        const int column = column_base + i;
        float value = 0.0f;
        if (column < hidden_size) {
            const float gate = __bfloat162float(gate_up[input_base + column]);
            const float up =
                __bfloat162float(gate_up[input_base + hidden_size + column]);
            value = __bfloat162float(__float2bfloat16(
                situ_activate(gate, up, beta, linear_beta)));
        }
        values[i] = value;
        local_max = fmaxf(local_max, fabsf(value));
    }

    const unsigned mask = __activemask();
    for (int delta = 16; delta > 0; delta >>= 1) {
        local_max = fmaxf(
            local_max, __shfl_down_sync(mask, local_max, delta));
    }
    __shared__ float warp_maxima[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) warp_maxima[warp] = local_max;
    __syncthreads();
    if (warp == 0) {
        const int warp_count = (blockDim.x + 31) / 32;
        float block_max = lane < warp_count ? warp_maxima[lane] : 0.0f;
        for (int delta = 16; delta > 0; delta >>= 1) {
            block_max = fmaxf(
                block_max, __shfl_down_sync(0xffffffffu, block_max, delta));
        }
        if (lane == 0) warp_maxima[0] = block_max;
    }
    __syncthreads();
    const float scale = warp_maxima[0] / 448.0f;
    const float inv_scale = scale == 0.0f ? 0.0f : 1.0f / scale;
    if (threadIdx.x == 0) output_scales[token] = scale;
    const size_t output_base = static_cast<size_t>(token) * hidden_size;
#pragma unroll
    for (int i = 0; i < kValuesPerThread; ++i) {
        const int column = column_base + i;
        if (column < hidden_size) {
            output[output_base + column] =
                __nv_fp8_e4m3(fmaxf(
                    fminf(values[i] * inv_scale, 448.0f), -448.0f));
        }
    }
}

__global__ void combine_token_scales_kernel(
    const float* activation_dequant,
    const float* residual,
    const int32_t* expert_offsets,
    float* token_scales) {
    const int g = blockIdx.x;
    const int begin = expert_offsets[g];
    const int end = expert_offsets[g + 1];
    const float expert_scale = residual[g] * 64.0f;
    for (int token = begin + threadIdx.x;
         token < end;
         token += blockDim.x) {
        token_scales[token] =
            activation_dequant[token] * expert_scale;
    }
}

Params make_params(const LowLatencyMxfp4Fp8LaunchOpts& opts) {
    Params params{};
    params.M = opts.N_orig;
    params.N = opts.token_counts;
    params.K = opts.K;
    params.batch_count = opts.G;
    params.ptr_A = opts.w_interleaved;
    params.ptr_B =
        reinterpret_cast<const low_latency_mxfp4_fp8_detail::ElementB*>(
            opts.acts);
    params.ptr_exp_offsets = opts.exp_offsets_interleaved;
    params.ptr_token_scales = opts.token_scales;
    params.ptr_D = opts.outs;
    params.input_offsets = opts.expert_offsets;
    params.output_offsets = opts.output_expert_offsets
        ? opts.output_expert_offsets
        : opts.expert_offsets;
    params.tile_experts = opts.tile_experts;
    params.tile_n = opts.tile_n;
    params.num_token_tiles = opts.num_token_tiles;
    params.num_token_tiles_device = opts.num_token_tiles_device;
    return params;
}

}  // namespace

void launch_low_latency_mxfp4_fp8_preprocess_scales(
    const uint8_t* raw_e8m0_scales,
    uint8_t* exp_offsets_logical,
    uint8_t* delta_offsets,
    float* expert_residual,
    int G,
    int N_orig,
    int K,
    cudaStream_t stream) {
    validate_shape(G, N_orig, K);
    if (G == 0) return;
    if (!raw_e8m0_scales || !exp_offsets_logical ||
        !delta_offsets || !expert_residual) {
        abort_mxfp4("null scale-preprocess tensor");
    }
    const size_t scales_per_expert =
        static_cast<size_t>(N_orig) * K / 32;
    preprocess_scales_kernel<<<G, 256, 0, stream>>>(
        raw_e8m0_scales,
        exp_offsets_logical,
        delta_offsets,
        expert_residual,
        scales_per_expert);
}

void launch_low_latency_mxfp4_fp8_rewrite_payload(
    const uint8_t* raw_weight,
    const uint8_t* delta_offsets,
    uint8_t* processed_weight,
    int G,
    int N_orig,
    int K,
    cudaStream_t stream) {
    validate_shape(G, N_orig, K);
    if (G == 0) return;
    if (!raw_weight || !delta_offsets || !processed_weight) {
        abort_mxfp4("null payload-rewrite tensor");
    }
    const auto& lut = humming_rewrite_lut();
    check_cuda(cudaMemcpyToSymbolAsync(kHummingRewriteLut,
                                       lut.data(),
                                       lut.size(),
                                       0,
                                       cudaMemcpyHostToDevice,
                                       stream),
               "failed to upload Humming rewrite LUT");
    const size_t bytes_per_expert =
        static_cast<size_t>(N_orig) * K / 2;
    const size_t scales_per_expert =
        static_cast<size_t>(N_orig) * K / 32;
    constexpr int kThreads = 256;
    const int blocks =
        static_cast<int>(
            std::min<size_t>((bytes_per_expert + kThreads - 1) / kThreads,
                             4096));
    rewrite_payload_kernel<<<dim3(blocks, G), kThreads, 0, stream>>>(
        raw_weight,
        delta_offsets,
        processed_weight,
        bytes_per_expert,
        scales_per_expert);
}

void launch_low_latency_mxfp4_fp8_interleave(
    const uint8_t* processed_weight,
    const uint8_t* exp_offsets_logical,
    uint8_t* w_interleaved,
    uint8_t* exp_offsets_interleaved,
    int G,
    int N_orig,
    int K,
    cudaStream_t stream) {
    validate_shape(G, N_orig, K);
    if (G == 0) return;
    if (!processed_weight || !exp_offsets_logical ||
        !w_interleaved || !exp_offsets_interleaved) {
        abort_mxfp4("null interleave tensor");
    }
    const dim3 block(128, 1, 1);
    const dim3 grid(std::min(N_orig / 64, 1024), std::min(G, 1024), 1);
    interleave_kernel<<<grid, block, 0, stream>>>(
        processed_weight,
        exp_offsets_logical,
        w_interleaved,
        exp_offsets_interleaved,
        G,
        N_orig,
        K);
}

void launch_low_latency_mxfp4_fp8_preprocess_weight(
    const uint8_t* raw_weight,
    const uint8_t* raw_e8m0_scales,
    uint8_t* delta_workspace,
    uint8_t* processed_workspace,
    uint8_t* w_interleaved,
    uint8_t* exp_offsets_logical,
    uint8_t* exp_offsets_interleaved,
    float* expert_residual,
    int G,
    int N_orig,
    int K,
    cudaStream_t stream) {
    launch_low_latency_mxfp4_fp8_preprocess_scales(
        raw_e8m0_scales,
        exp_offsets_logical,
        delta_workspace,
        expert_residual,
        G,
        N_orig,
        K,
        stream);
    launch_low_latency_mxfp4_fp8_rewrite_payload(
        raw_weight,
        delta_workspace,
        processed_workspace,
        G,
        N_orig,
        K,
        stream);
    launch_low_latency_mxfp4_fp8_interleave(
        processed_workspace,
        exp_offsets_logical,
        w_interleaved,
        exp_offsets_interleaved,
        G,
        N_orig,
        K,
        stream);
}

void launch_low_latency_mxfp4_fp8_prepare_deepep_layout(
    const int32_t* token_counts, int G, int capacity,
    int tile_schedule_capacity, int32_t* padded_offsets,
    int32_t* compact_offsets, int32_t* tile_experts, int32_t* tile_n,
    int32_t* num_token_tiles, cudaStream_t stream) {
    if (G < 0 || capacity < 0) abort_mxfp4("invalid DeepEP layout shape");
    if (G == 0) {
        if (num_token_tiles) check_cuda(cudaMemsetAsync(
            num_token_tiles, 0, sizeof(int32_t), stream),
            "failed to reset empty tile count");
        return;
    }
    if (!token_counts || !padded_offsets || !compact_offsets ||
        !tile_experts || !tile_n || !num_token_tiles) {
        abort_mxfp4("null DeepEP layout tensor");
    }
    const int required_capacity = G * ((capacity + 7) / 8);
    if (tile_schedule_capacity < required_capacity) {
        abort_mxfp4("DeepEP tile schedule capacity is too small");
    }
    constexpr int kThreads = 256;
    const size_t shared_bytes =
        static_cast<size_t>(2 * (G + 1)) * sizeof(int32_t);
    prepare_deepep_layout_kernel<<<1, kThreads, shared_bytes, stream>>>(
        token_counts, G, capacity, padded_offsets, compact_offsets,
        tile_experts, tile_n, num_token_tiles);
}

void launch_low_latency_mxfp4_fp8_requantize_deepep_compact(
    const __nv_fp8_e4m3* input, const float* input_group_scales,
    const int32_t* token_counts, const int32_t* compact_offsets,
    int G, int capacity, int hidden_size, int64_t scale_stride_expert,
    int64_t scale_stride_token, int64_t scale_stride_group,
    __nv_fp8_e4m3* output, float* output_scales, cudaStream_t stream) {
    if (G < 0 || capacity < 0 || hidden_size <= 0 || hidden_size % 128 != 0) {
        abort_mxfp4("invalid DeepEP FP8 requantize shape");
    }
    if (G == 0 || capacity == 0) return;
    if (!input || !input_group_scales || !token_counts || !compact_offsets ||
        !output || !output_scales) {
        abort_mxfp4("null DeepEP FP8 requantize tensor");
    }
    constexpr int kThreads = 256;
    requantize_deepep_compact_kernel<<<G * capacity, kThreads, 0, stream>>>(
        input, input_group_scales, token_counts, compact_offsets, G, capacity,
        hidden_size, scale_stride_expert, scale_stride_token,
        scale_stride_group, output, output_scales);
}

void launch_low_latency_mxfp4_fp8_situ_quant_compact(
    const __nv_bfloat16* gate_up, const int32_t* compact_offsets, int G,
    int max_tokens, int hidden_size, float beta, float linear_beta,
    __nv_fp8_e4m3* output, float* output_scales, cudaStream_t stream) {
    if (G < 0 || max_tokens < 0 || hidden_size <= 0 ||
        hidden_size % 8 != 0 || hidden_size > 8192) {
        abort_mxfp4("invalid compact SiTU shape");
    }
    if (max_tokens == 0) return;
    if (!gate_up || !compact_offsets || !output || !output_scales) {
        abort_mxfp4("null compact SiTU tensor");
    }
    if (beta == 0.0f || linear_beta == 0.0f) {
        abort_mxfp4("SiTU beta values must be non-zero");
    }
    const int threads = (hidden_size + 7) / 8;
    situ_quant_compact_kernel<<<max_tokens, threads, 0, stream>>>(
        gate_up, compact_offsets, G, hidden_size, beta, linear_beta,
        output, output_scales);
}

void launch_low_latency_mxfp4_fp8_combine_token_scales(
    const float* activation_dequant,
    const float* expert_residual,
    const int32_t* expert_offsets,
    float* token_scales,
    int G,
    cudaStream_t stream) {
    if (G < 0) abort_mxfp4("G must be non-negative");
    if (G == 0) return;
    if (!activation_dequant || !expert_residual ||
        !expert_offsets || !token_scales) {
        abort_mxfp4("null token-scale tensor");
    }
    combine_token_scales_kernel<<<G, 256, 0, stream>>>(
        activation_dequant,
        expert_residual,
        expert_offsets,
        token_scales);
}

void launch_low_latency_mxfp4_fp8(
    const LowLatencyMxfp4Fp8LaunchOpts& opts) {
    validate_shape(opts.G, opts.N_orig, opts.K);
    if (opts.G == 0) return;
    if (!opts.acts || !opts.w_interleaved ||
        !opts.exp_offsets_interleaved || !opts.token_scales ||
        !opts.token_counts || !opts.expert_offsets || !opts.outs) {
        abort_mxfp4("null GEMM tensor");
    }

    const bool host_scheduled = opts.num_token_tiles > 0;
    const bool device_scheduled = opts.num_token_tiles_device != nullptr;
    if (!host_scheduled && !device_scheduled && opts.max_M_g <= 0) {
        abort_mxfp4("max_M_g must be positive for rectangular launch");
    }
    if (host_scheduled && device_scheduled) {
        abort_mxfp4("host and device schedules are mutually exclusive");
    }
    if ((host_scheduled || device_scheduled) &&
        (!opts.tile_experts || !opts.tile_n)) {
        abort_mxfp4("compact schedule requires tile arrays");
    }

    const Params params = make_params(opts);
    const dim3 block(
        low_latency_mxfp4_fp8_detail::kThreadsPerRow,
        low_latency_mxfp4_fp8_detail::kThreadCount /
            low_latency_mxfp4_fp8_detail::kThreadsPerRow,
        1);
    const int row_tiles = (opts.N_orig + 127) / 128;

    if (device_scheduled) {
        if (opts.persistent_ctas <= 0) {
            abort_mxfp4("device schedule requires persistent_ctas");
        }
        if (opts.build_device_schedule) {
            if (opts.tile_schedule_capacity <= 0) {
                abort_mxfp4("device schedule capacity is too small");
            }
            launch_low_latency_grouped_gemm_build_tile_schedule(
                opts.token_counts,
                opts.G,
                opts.tile_schedule_capacity,
                opts.tile_experts,
                opts.tile_n,
                opts.num_token_tiles_device,
                opts.stream);
        }
        if (opts.N_orig == 2560 && opts.K == 4096) {
            low_latency_mxfp4_fp8_detail::
                low_latency_mxfp4_fp8_device_schedule_kernel<
                    2560, 4096><<<
                    opts.persistent_ctas, block, 0, opts.stream>>>(
                    params, row_tiles);
        } else if (opts.N_orig == 4096 && opts.K == 1280) {
            low_latency_mxfp4_fp8_detail::
                low_latency_mxfp4_fp8_device_schedule_kernel<
                    4096, 1280><<<
                    opts.persistent_ctas, block, 0, opts.stream>>>(
                    params, row_tiles);
        } else if (opts.N_orig == 1024 && opts.K == 4096) {
            low_latency_mxfp4_fp8_detail::
                low_latency_mxfp4_fp8_device_schedule_kernel<
                    1024, 4096><<<
                    opts.persistent_ctas, block, 0, opts.stream>>>(
                    params, row_tiles);
        } else if (opts.N_orig == 4096 && opts.K == 512) {
            low_latency_mxfp4_fp8_detail::
                low_latency_mxfp4_fp8_device_schedule_kernel<
                    4096, 512><<<
                    opts.persistent_ctas, block, 0, opts.stream>>>(
                    params, row_tiles);
        } else {
            low_latency_mxfp4_fp8_detail::
                low_latency_mxfp4_fp8_device_schedule_kernel<><<<
                    opts.persistent_ctas, block, 0, opts.stream>>>(
                    params, row_tiles);
        }
        return;
    }

    const dim3 grid(
        row_tiles,
        host_scheduled ? 1 : (opts.max_M_g + 7) / 8,
        host_scheduled ? opts.num_token_tiles : opts.G);
    if (opts.N_orig == 2560 && opts.K == 4096) {
        low_latency_mxfp4_fp8_detail::
            low_latency_mxfp4_fp8_kernel<2560, 4096><<<
                grid, block, 0, opts.stream>>>(params);
    } else if (opts.N_orig == 4096 && opts.K == 1280) {
        low_latency_mxfp4_fp8_detail::
            low_latency_mxfp4_fp8_kernel<4096, 1280><<<
                grid, block, 0, opts.stream>>>(params);
    } else if (opts.N_orig == 1024 && opts.K == 4096) {
        low_latency_mxfp4_fp8_detail::
            low_latency_mxfp4_fp8_kernel<1024, 4096><<<
                grid, block, 0, opts.stream>>>(params);
    } else if (opts.N_orig == 4096 && opts.K == 512) {
        low_latency_mxfp4_fp8_detail::
            low_latency_mxfp4_fp8_kernel<4096, 512><<<
                grid, block, 0, opts.stream>>>(params);
    } else {
        low_latency_mxfp4_fp8_detail::low_latency_mxfp4_fp8_kernel<><<<
            grid, block, 0, opts.stream>>>(params);
    }
}

}  // namespace mga
