// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include "low_latency_grouped_gemm/include/low_latency_mxfp4_fp8.h"

namespace {
cudaStream_t current_stream(const torch::Tensor& tensor) {
  return c10::cuda::getCurrentCUDAStream(tensor.get_device()).stream();
}
void check_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}
__global__ void counts_from_offsets_kernel(
    const int32_t* offsets, int32_t* counts, int experts) {
  for (int expert = blockIdx.x * blockDim.x + threadIdx.x;
       expert < experts; expert += blockDim.x * gridDim.x) {
    counts[expert] = offsets[expert + 1] - offsets[expert];
  }
}

std::vector<torch::Tensor> preprocess_weight(
    torch::Tensor raw_weight, torch::Tensor raw_e8m0_scales) {
  check_cuda_contiguous(raw_weight, "raw_weight");
  check_cuda_contiguous(raw_e8m0_scales, "raw_e8m0_scales");
  TORCH_CHECK(raw_e8m0_scales.dim() == 3,
              "raw_e8m0_scales must have shape [experts, N, K/32]");
  TORCH_CHECK(raw_e8m0_scales.element_size() == 1,
              "raw_e8m0_scales must use one-byte E8M0 storage");
  const int experts = raw_e8m0_scales.size(0);
  const int n = raw_e8m0_scales.size(1);
  const int k = raw_e8m0_scales.size(2) * 32;
  const int64_t weight_bytes =
      mga::low_latency_mxfp4_fp8_weight_bytes(experts, n, k);
  const int64_t scale_bytes =
      mga::low_latency_mxfp4_fp8_scale_bytes(experts, n, k);
  TORCH_CHECK(raw_weight.numel() * raw_weight.element_size() == weight_bytes,
              "raw_weight byte size does not match scale-derived shape");
  TORCH_CHECK(raw_e8m0_scales.numel() == scale_bytes,
              "raw_e8m0_scales size does not match derived shape");

  c10::cuda::CUDAGuard device_guard(raw_weight.device());
  const auto byte_options = raw_weight.options().dtype(torch::kUInt8);
  const auto float_options = raw_weight.options().dtype(torch::kFloat32);
  auto delta = torch::empty({scale_bytes}, byte_options);
  auto processed = torch::empty({weight_bytes}, byte_options);
  auto interleaved = torch::empty({weight_bytes}, byte_options);
  auto logical_offsets = torch::empty({scale_bytes}, byte_options);
  auto interleaved_offsets = torch::empty({scale_bytes}, byte_options);
  auto residual = torch::empty({experts}, float_options);
  mga::launch_low_latency_mxfp4_fp8_preprocess_weight(
      reinterpret_cast<const uint8_t*>(raw_weight.data_ptr()),
      reinterpret_cast<const uint8_t*>(raw_e8m0_scales.data_ptr()),
      delta.data_ptr<uint8_t>(), processed.data_ptr<uint8_t>(),
      interleaved.data_ptr<uint8_t>(), logical_offsets.data_ptr<uint8_t>(),
      interleaved_offsets.data_ptr<uint8_t>(), residual.data_ptr<float>(),
      experts, n, k, current_stream(raw_weight));
  return {interleaved, interleaved_offsets, residual};
}

torch::Tensor grouped_gemm_out_impl(
    torch::Tensor acts, torch::Tensor activation_scales,
    torch::Tensor interleaved_weight, torch::Tensor interleaved_exp_offsets,
    torch::Tensor expert_residual, torch::Tensor expert_offsets,
    torch::Tensor token_counts, torch::Tensor token_scales,
    torch::Tensor tile_experts, torch::Tensor tile_n,
    torch::Tensor num_tiles, torch::Tensor output,
    int64_t n, int64_t k, int64_t persistent_ctas,
    bool token_counts_are_precomputed) {
  for (const auto& item : {
           std::pair<const torch::Tensor*, const char*>{&acts, "acts"},
           {&activation_scales, "activation_scales"},
           {&interleaved_weight, "interleaved_weight"},
           {&interleaved_exp_offsets, "interleaved_exp_offsets"},
           {&expert_residual, "expert_residual"},
           {&expert_offsets, "expert_offsets"},
           {&token_counts, "token_counts"},
           {&token_scales, "token_scales"},
           {&tile_experts, "tile_experts"}, {&tile_n, "tile_n"},
           {&num_tiles, "num_tiles"}, {&output, "output"}}) {
    check_cuda_contiguous(*item.first, item.second);
  }
  TORCH_CHECK(acts.dim() == 2 && acts.size(1) == k,
              "acts must have shape [routed_tokens, K]");
  TORCH_CHECK(acts.element_size() == 1, "acts must use one-byte FP8 storage");
  TORCH_CHECK(activation_scales.scalar_type() == torch::kFloat32,
              "activation_scales must be float32");
  TORCH_CHECK(expert_residual.scalar_type() == torch::kFloat32,
              "expert_residual must be float32");
  TORCH_CHECK(expert_offsets.scalar_type() == torch::kInt32,
              "expert_offsets must be int32");
  TORCH_CHECK(token_counts.scalar_type() == torch::kInt32,
              "token_counts must be int32");
  TORCH_CHECK(output.scalar_type() == torch::kBFloat16,
              "output must be bfloat16");
  TORCH_CHECK(persistent_ctas > 0, "persistent_ctas must be positive");

  const int experts = expert_residual.numel();
  const int64_t routed_tokens = acts.size(0);
  TORCH_CHECK(expert_offsets.numel() == experts + 1,
              "expert_offsets must have experts + 1 entries");
  TORCH_CHECK(token_counts.numel() == experts,
              "token_counts must have one entry per expert");
  TORCH_CHECK(activation_scales.numel() >= routed_tokens,
              "activation_scales capacity is too small");
  TORCH_CHECK(token_scales.numel() >= routed_tokens,
              "token_scales capacity is too small");
  TORCH_CHECK(tile_experts.numel() >= routed_tokens &&
                  tile_n.numel() >= routed_tokens,
              "tile schedule capacity is too small");
  TORCH_CHECK(num_tiles.numel() >= 1, "num_tiles must have one entry");
  TORCH_CHECK(output.dim() == 2 && output.size(0) >= routed_tokens &&
                  output.size(1) == n,
              "output capacity or shape is invalid");

  c10::cuda::CUDAGuard device_guard(acts.device());
  const cudaStream_t stream = current_stream(acts);
  if (!token_counts_are_precomputed) {
    counts_from_offsets_kernel<<<1, 256, 0, stream>>>(
        expert_offsets.data_ptr<int32_t>(), token_counts.data_ptr<int32_t>(),
        experts);
  }
  mga::launch_low_latency_mxfp4_fp8_combine_token_scales(
      activation_scales.data_ptr<float>(), expert_residual.data_ptr<float>(),
      expert_offsets.data_ptr<int32_t>(), token_scales.data_ptr<float>(),
      experts, stream);

  mga::LowLatencyMxfp4Fp8LaunchOpts launch{};
  launch.G = experts;
  launch.N_orig = static_cast<int>(n);
  launch.K = static_cast<int>(k);
  launch.max_M_g = 0;
  launch.acts = reinterpret_cast<const __nv_fp8_e4m3*>(acts.data_ptr());
  launch.w_interleaved = interleaved_weight.data_ptr<uint8_t>();
  launch.exp_offsets_interleaved = interleaved_exp_offsets.data_ptr<uint8_t>();
  launch.token_scales = token_scales.data_ptr<float>();
  launch.token_counts = token_counts.data_ptr<int32_t>();
  launch.expert_offsets = expert_offsets.data_ptr<int32_t>();
  launch.tile_experts = tile_experts.data_ptr<int32_t>();
  launch.tile_n = tile_n.data_ptr<int32_t>();
  launch.num_token_tiles_device = num_tiles.data_ptr<int32_t>();
  launch.tile_schedule_capacity = tile_experts.numel();
  launch.persistent_ctas = static_cast<int>(persistent_ctas);
  launch.build_device_schedule = true;
  launch.outs =
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>());
  launch.stream = stream;
  mga::launch_low_latency_mxfp4_fp8(launch);
  return output.narrow(0, 0, routed_tokens);
}
}  // namespace

torch::Tensor grouped_gemm_out(
    torch::Tensor acts, torch::Tensor activation_scales,
    torch::Tensor interleaved_weight, torch::Tensor interleaved_exp_offsets,
    torch::Tensor expert_residual, torch::Tensor expert_offsets,
    torch::Tensor token_counts, torch::Tensor token_scales,
    torch::Tensor tile_experts, torch::Tensor tile_n,
    torch::Tensor num_tiles, torch::Tensor output,
    int64_t n, int64_t k, int64_t persistent_ctas) {
  return grouped_gemm_out_impl(
      acts, activation_scales, interleaved_weight, interleaved_exp_offsets,
      expert_residual, expert_offsets, token_counts, token_scales,
      tile_experts, tile_n, num_tiles, output, n, k, persistent_ctas, false);
}

torch::Tensor grouped_gemm_out_precomputed_counts(
    torch::Tensor acts, torch::Tensor activation_scales,
    torch::Tensor interleaved_weight, torch::Tensor interleaved_exp_offsets,
    torch::Tensor expert_residual, torch::Tensor expert_offsets,
    torch::Tensor token_counts, torch::Tensor token_scales,
    torch::Tensor tile_experts, torch::Tensor tile_n,
    torch::Tensor num_tiles, torch::Tensor output,
    int64_t n, int64_t k, int64_t persistent_ctas) {
  return grouped_gemm_out_impl(
      acts, activation_scales, interleaved_weight, interleaved_exp_offsets,
      expert_residual, expert_offsets, token_counts, token_scales,
      tile_experts, tile_n, num_tiles, output, n, k, persistent_ctas, true);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("preprocess_weight", &preprocess_weight,
             "Preprocess raw MXFP4 weights for LowLatencyGroupedGEMM");
  module.def("grouped_gemm_out", &grouped_gemm_out,
             "Run LowLatencyGroupedGEMM into caller-owned graph-safe buffers");
  module.def("grouped_gemm_out_precomputed_counts",
             &grouped_gemm_out_precomputed_counts,
             "Run LowLatencyGroupedGEMM with caller-provided expert counts");
}
