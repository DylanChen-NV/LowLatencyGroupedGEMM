# W4A8 MoE GEMV Decode Kernel

Optimized CUTLASS GEMV kernel for W4A8 MoE decode on Hopper GPUs (sm_90a). Adds adaptive dispatch with 1D flat grid and sorted I/O to the base GEMV kernel from the [LowLatencyGroupedGEMM](https://github.com/StudyingShao/LowLatencyGroupedGEMM) project.

## Build

```bash
# Clone with CUTLASS submodule
git clone --recursive -b feat/adaptive-gemv-w4a8 \
    https://github.com/DylanChen-NV/LowLatencyGroupedGEMM.git
cd LowLatencyGroupedGEMM

# Build shared library
nvcc -shared -o libgemv_kernel.so gemv_kernel.cu \
  -O3 -std=c++17 \
  -gencode arch=compute_90a,code=sm_90a \
  -Icutlass/include -Icutlass/tools/util/include \
  --expt-extended-lambda --expt-relaxed-constexpr \
  -Xcompiler -fPIC
```

## C API

All functions use `extern "C"` linkage. Load via `dlopen` / `ctypes`.

### `launch_preprocess_weights`

One-time weight preprocessing (interleave INT4 weights + pad scales).

```c
void launch_preprocess_weights(
    void *w_interleaved,   // [E, M, K/2] int8 output
    void *s_padded,        // [E, M, padded_sfs] fp16 output
    void *w_orig,          // [E, M, K/2] int8 input
    void *s_orig,          // [E, M, K/SFV] fp16 input
    int E, int M, int K,
    cudaStream_t stream);
```

### `build_and_launch_gemv_sorted` (recommended)

Fused work table build + GEMV launch. Zero `cudaStreamSynchronize`. Automatically selects 1D flat grid (skewed routing) or 3D grid (uniform routing) based on CTA savings threshold.

```c
void build_and_launch_gemv_sorted(
    const int32_t *h_n_per_expert, // HOST [E] — tokens per expert
    void *d_sorted_out,            // [total, M] bf16 output
    void *w_interleaved,           // [E, M, K/2] int8 (preprocessed)
    void *act_sorted,              // [total, K] fp8 input (sorted by expert)
    void *scales,                  // [E, M, padded_sfs] fp16 (preprocessed)
    int32_t *expert_offsets,       // [E+1] device — prefix sum of tokens per expert
    int32_t *d_n_per_expert,       // [E] device — tokens per expert
    float alpha,                   // scaling factor
    int M, int K, int E,
    float threshold,               // dispatch threshold (0.7 = use flat when >30% CTA savings)
    cudaStream_t stream);
```

**Data layout**: inputs/outputs are in expert-sorted order (contiguous tokens per expert). Use `expert_offsets` prefix sum for addressing. No scatter/gather buffers needed.

### `launch_gemv_strided`

Strided I/O path with scatter/gather. Requires pre-allocated strided buffers. Best for uniform routing where 3D grid has no CTA waste.

```c
void launch_gemv_strided(
    void *d_sorted_out,            // [total, M] bf16 output
    void *w_interleaved,           // [E, M, K/2] int8
    void *act_sorted,              // [total, K] fp8 input
    void *scales,                  // [E, M, padded_sfs] fp16
    int32_t *expert_offsets,       // [E+1] device
    int32_t *n_per_expert,         // [E] device
    float alpha,
    int M, int K, int max_N, int E,
    void *B_strided_buf,           // [E, max_N, K] fp8 scratch
    void *D_strided_buf,           // [E, M, max_N] bf16 scratch
    cudaStream_t stream);
```

## Dispatch Logic

`build_and_launch_gemv_sorted` decides the dispatch path per call:

```
flat_ctas  = sum of (M_tiles * n_tiles) for active experts
orig_ctas  = M_tiles * ceil(max_N/8) * E

if flat_ctas < orig_ctas * threshold:
    → sorted I/O + 1D flat grid    (skew: 99% fewer CTAs)
else:
    → sorted I/O + 3D grid         (uniform: no binary search overhead)
```
