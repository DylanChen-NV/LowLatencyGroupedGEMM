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
    tile_experts = torch.empty(
        padded_tokens, dtype=torch.int32, device=device
    )
    tile_n = torch.empty_like(tile_experts)
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

    activation_hidden = 256
    gate_up = torch.randn(
        padded_tokens, activation_hidden * 2,
        dtype=torch.bfloat16, device=device
    )
    q2 = torch.zeros(
        padded_tokens, activation_hidden,
        dtype=torch.float8_e4m3fn, device=device
    )
    s2 = torch.full(
        (padded_tokens, 1), torch.nan, dtype=torch.float32, device=device
    )
    low_latency.situ_quant_compact_out(
        gate_up, compact_offsets, q2, s2, 4.0, 25.0
    )
    torch.cuda.synchronize()
    gate, up = gate_up[:routed_tokens].float().chunk(2, dim=1)
    reference_activation = (
        4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)
        * 25.0 * torch.tanh(up / 25.0)
    )
    dequant_activation = (
        q2[:routed_tokens].float() * s2[:routed_tokens].float()
    )
    torch.testing.assert_close(
        dequant_activation, reference_activation, rtol=0.05, atol=0.05
    )
    assert torch.count_nonzero(q2[routed_tokens:]) == 0
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

    intermediate = n // 2
    raw_w2 = torch.randint(
        0, 256, (experts, k, intermediate // 2),
        dtype=torch.uint8, device=device
    )
    raw_s2 = torch.randint(
        120, 133, (experts, k, intermediate // 32),
        dtype=torch.uint8, device=device
    )
    w2, w2_offsets, w2_residual = low_latency.preprocess_weight(
        raw_w2, raw_s2
    )
    fc1_token_scales = torch.empty(
        padded_tokens, dtype=torch.float32, device=device
    )
    gate_up_pipeline = torch.full(
        (padded_tokens, n), torch.nan, dtype=torch.bfloat16, device=device
    )
    q2_pipeline = torch.zeros(
        (padded_tokens, intermediate),
        dtype=torch.float8_e4m3fn, device=device
    )
    q2_pipeline_scales = torch.full(
        (padded_tokens, 1), torch.nan, dtype=torch.float32, device=device
    )
    fc2_token_scales = torch.empty_like(fc1_token_scales)
    pipeline_output = torch.full(
        (padded_tokens, k), torch.nan, dtype=torch.bfloat16, device=device
    )
    low_latency.deepep_moe_out(
        padded_acts, padded_scales, weight, scale, residual, w2, w2_offsets,
        w2_residual, counts, input_offsets, compact_offsets, tile_experts,
        tile_n, num_tiles, fc1_token_scales, gate_up_pipeline, q2_pipeline,
        q2_pipeline_scales, fc2_token_scales, pipeline_output, capacity, k,
        intermediate, 528, 4.0, 25.0
    )
    torch.cuda.synchronize()
    valid = torch.cat([
        torch.arange(expert * capacity, expert * capacity + 3, device=device)
        for expert in range(experts)
    ])
    padding = torch.cat([
        torch.arange(expert * capacity + 3, (expert + 1) * capacity, device=device)
        for expert in range(experts)
    ])
    assert torch.isfinite(pipeline_output[valid]).all()
    assert torch.isnan(pipeline_output[padding]).all()
    pipeline_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(pipeline_graph):
        low_latency.deepep_moe_out(
            padded_acts, padded_scales, weight, scale, residual, w2, w2_offsets,
            w2_residual, counts, input_offsets, compact_offsets, tile_experts,
            tile_n, num_tiles, fc1_token_scales, gate_up_pipeline, q2_pipeline,
            q2_pipeline_scales, fc2_token_scales, pipeline_output, capacity, k,
            intermediate, 528, 4.0, 25.0
        )
    pipeline_graph.replay()
    torch.cuda.synchronize()
    assert torch.isfinite(pipeline_output[valid]).all()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.isfinite(output).all()
    print("PASS: dual offsets, finite output, and CUDA Graph replay")


if __name__ == "__main__":
    main()
