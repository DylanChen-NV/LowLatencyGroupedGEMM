from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import low_latency_mxfp4 as llop


def make_weight(experts, n, k, device):
    raw_w = torch.randint(
        0, 256, (experts, n, k // 2), dtype=torch.uint8, device=device
    )
    raw_s = torch.randint(
        120, 133, (experts, n, k // 32), dtype=torch.uint8, device=device
    )
    return llop.preprocess_weight(raw_w, raw_s)


def main():
    device = torch.device("cuda")
    torch.manual_seed(11)
    experts, capacity, hidden, intermediate = 4, 4, 512, 128
    rows = experts * capacity
    counts = torch.tensor([3, 1, 2, 4], dtype=torch.int32, device=device)
    valid_rows = int(counts.sum().item())
    groups = hidden // 128

    source = torch.randn(
        (experts, capacity, hidden), dtype=torch.bfloat16, device=device
    )
    grouped = source.float().view(experts, capacity, groups, 128)
    group_scales = (grouped.abs().amax(dim=-1) / 448.0).clamp_min(1.0e-12)
    recv_q = (grouped / group_scales.unsqueeze(-1)).to(
        torch.float8_e4m3fn
    ).view(experts, capacity, hidden)
    # DeepEP documents the last two scale dimensions as column-major.
    recv_scales = group_scales.permute(0, 2, 1).contiguous().permute(0, 2, 1)
    assert recv_scales.shape == (experts, capacity, groups)
    assert not recv_scales.is_contiguous()

    padded_offsets = torch.empty(experts + 1, dtype=torch.int32, device=device)
    compact_offsets = torch.empty_like(padded_offsets)
    tile_experts = torch.empty(rows, dtype=torch.int32, device=device)
    tile_n = torch.empty_like(tile_experts)
    num_tiles = torch.empty(1, dtype=torch.int32, device=device)
    llop.prepare_deepep_layout_out(
        counts, capacity, padded_offsets, compact_offsets,
        tile_experts, tile_n, num_tiles,
    )

    compact_q = torch.zeros(
        (rows, hidden), dtype=torch.float8_e4m3fn, device=device
    )
    compact_scales = torch.full(
        (rows, 1), torch.nan, dtype=torch.float32, device=device
    )
    llop.requantize_deepep_fp8_compact_out(
        recv_q, recv_scales, counts, compact_offsets,
        compact_q, compact_scales,
    )
    torch.cuda.synchronize()

    dequant = (
        recv_q.view(experts, capacity, groups, 128).float()
        * group_scales.unsqueeze(-1)
    ).view(experts, capacity, hidden)
    expected_scales = group_scales.amax(dim=-1, keepdim=True)
    expected_q = (dequant / expected_scales).to(torch.float8_e4m3fn)
    expected_q_compact = torch.cat(
        [expected_q[e, : int(counts[e].item())] for e in range(experts)]
    )
    expected_s_compact = torch.cat(
        [expected_scales[e, : int(counts[e].item())] for e in range(experts)]
    )
    torch.testing.assert_close(
        compact_q[:valid_rows].float(), expected_q_compact.float(), rtol=0, atol=0
    )
    torch.testing.assert_close(
        compact_scales[:valid_rows], expected_s_compact, rtol=0, atol=0
    )

    w13, w13_off, w13_res = make_weight(experts, 2 * intermediate, hidden, device)
    w2, w2_off, w2_res = make_weight(experts, hidden, intermediate, device)

    def workspace():
        return {
            "padded": torch.empty(experts + 1, dtype=torch.int32, device=device),
            "compact": torch.empty(experts + 1, dtype=torch.int32, device=device),
            "te": torch.empty(rows, dtype=torch.int32, device=device),
            "tn": torch.empty(rows, dtype=torch.int32, device=device),
            "nt": torch.empty(1, dtype=torch.int32, device=device),
            "q1": torch.empty((rows, hidden), dtype=torch.float8_e4m3fn, device=device),
            "q1s": torch.empty((rows, 1), dtype=torch.float32, device=device),
            "fc1ts": torch.empty(rows, dtype=torch.float32, device=device),
            "gate_up": torch.empty((rows, 2 * intermediate), dtype=torch.bfloat16, device=device),
            "q2": torch.empty((rows, intermediate), dtype=torch.float8_e4m3fn, device=device),
            "q2s": torch.empty((rows, 1), dtype=torch.float32, device=device),
            "fc2ts": torch.empty(rows, dtype=torch.float32, device=device),
            "out": torch.full((rows, hidden), torch.nan, dtype=torch.bfloat16, device=device),
        }

    f1 = workspace()
    b1 = workspace()

    def run_f1():
        llop.deepep_fp8_moe_out(
            recv_q, recv_scales, w13, w13_off, w13_res,
            w2, w2_off, w2_res, counts, f1["padded"], f1["compact"],
            f1["te"], f1["tn"], f1["nt"], f1["q1"], f1["q1s"],
            f1["fc1ts"], f1["gate_up"], f1["q2"], f1["q2s"],
            f1["fc2ts"], f1["out"], capacity, hidden, intermediate,
            528, 4.0, 25.0,
        )

    padded_q = expected_q.reshape(rows, hidden).contiguous()
    padded_s = expected_scales.reshape(rows, 1).contiguous()
    llop.deepep_moe_out(
        padded_q, padded_s, w13, w13_off, w13_res,
        w2, w2_off, w2_res, counts, b1["padded"], b1["compact"],
        b1["te"], b1["tn"], b1["nt"], b1["fc1ts"], b1["gate_up"],
        b1["q2"], b1["q2s"], b1["fc2ts"], b1["out"],
        capacity, hidden, intermediate, 528, 4.0, 25.0,
    )
    run_f1()
    torch.cuda.synchronize()

    valid = torch.cat([
        torch.arange(e * capacity, e * capacity + int(counts[e].item()), device=device)
        for e in range(experts)
    ])
    torch.testing.assert_close(f1["out"][valid], b1["out"][valid], rtol=0, atol=0)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_f1()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.isfinite(f1["out"][valid]).all()
    print("PASS: DeepEP group128 FP8 compact, B1 equivalence, and CUDA Graph")


if __name__ == "__main__":
    main()
