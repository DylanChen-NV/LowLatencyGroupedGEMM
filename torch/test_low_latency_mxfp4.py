from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

import low_latency_mxfp4 as low_latency


def main():
    experts, n, k, routed_tokens = 8, 2560, 4096, 24
    device = torch.device("cuda")
    torch.manual_seed(7)

    raw_weight = torch.randint(
        0, 256, (experts, n, k // 2), dtype=torch.uint8, device=device
    )
    raw_scale = torch.randint(
        120, 133, (experts, n, k // 32), dtype=torch.uint8, device=device
    )
    weight, scale, residual = low_latency.preprocess_weight(raw_weight, raw_scale)

    source = torch.randn(routed_tokens, k, dtype=torch.bfloat16, device=device)
    activation_scale = (
        source.abs().amax(1, keepdim=True).float() / 448.0
    ).clamp_min(1.0e-12)
    acts = (source.float() / activation_scale).to(torch.float8_e4m3fn)

    counts = torch.full((experts,), 3, dtype=torch.int32, device=device)
    offsets = torch.cat(
        (
            torch.zeros(1, dtype=torch.int32, device=device),
            counts.cumsum(0).to(torch.int32),
        )
    )
    token_counts = torch.empty_like(counts)
    token_scales = torch.empty(routed_tokens, dtype=torch.float32, device=device)
    tile_experts = torch.empty(routed_tokens, dtype=torch.int32, device=device)
    tile_n = torch.empty_like(tile_experts)
    num_tiles = torch.empty(1, dtype=torch.int32, device=device)
    output = torch.empty(
        routed_tokens, n, dtype=torch.bfloat16, device=device
    )

    def run():
        return low_latency.grouped_gemm_out(
            acts,
            activation_scale,
            weight,
            scale,
            residual,
            offsets,
            token_counts,
            token_scales,
            tile_experts,
            tile_n,
            num_tiles,
            output,
            n,
            k,
            528,
        )

    result = run()
    torch.cuda.synchronize()
    assert result.shape == (routed_tokens, n)
    assert torch.isfinite(result).all()

    # Verify independent input/output expert bases: read a padded carrier and
    # write the same valid rows into compact expert-major order.
    capacity = 4
    padded_tokens = experts * capacity
    padded_acts = torch.zeros(
        padded_tokens, k, dtype=torch.float8_e4m3fn, device=device
    )
    padded_scales = torch.ones(
        padded_tokens, 1, dtype=torch.float32, device=device
    )
    input_offsets = torch.empty(
        experts + 1, dtype=torch.int32, device=device
    )
    compact_offsets = torch.empty_like(input_offsets)
    low_latency.prepare_deepep_layout_out(
        counts, capacity, input_offsets, compact_offsets, tile_experts,
        tile_n, num_tiles
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(
        input_offsets,
        torch.arange(experts + 1, dtype=torch.int32, device=device) * capacity,
    )
    torch.testing.assert_close(compact_offsets, offsets)
    for expert in range(experts):
        compact_slice = slice(expert * 3, expert * 3 + 3)
        padded_slice = slice(expert * capacity, expert * capacity + 3)
        padded_acts[padded_slice].copy_(acts[compact_slice])
        padded_scales[padded_slice].copy_(activation_scale[compact_slice])
    dual_token_scales = torch.empty(
        padded_tokens, dtype=torch.float32, device=device
    )
    dual_output = torch.full(
        (padded_tokens, n), torch.nan, dtype=torch.bfloat16, device=device
    )
    dual_result = low_latency.grouped_gemm_out_dual_offsets(
        padded_acts, padded_scales, weight, scale, residual, input_offsets,
        compact_offsets, counts, dual_token_scales, tile_experts, tile_n, num_tiles,
        dual_output, n, k, 528
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(
        dual_result[:routed_tokens], result, rtol=0, atol=0
    )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.isfinite(output).all()
    print("PASS: dual offsets, finite output, and CUDA Graph replay")


if __name__ == "__main__":
    main()
