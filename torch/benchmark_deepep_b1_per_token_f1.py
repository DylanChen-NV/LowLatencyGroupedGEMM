import json
import os
import statistics

import torch

import low_latency_mxfp4 as llop
from sglang.kernels.ops.quantization import sgl_per_token_quant_fp8


E = 28
H = 3584
I = 3072
CAP = 8
ROWS = E * CAP
PERSISTENT_CTAS = 528
BETA = 4.0
LINEAR_BETA = 25.0
WARMUP = 20
ITERS = 200
REPETITIONS = 3


torch.manual_seed(42)
device = torch.device("cuda")
assert torch.cuda.get_device_capability(0) == (9, 0)

# 64 routed rows/rank approximates BS128 * topk16 / EP32. The remaining
# expert capacity is DeepEP padding and is intentionally skipped by F1.
counts = torch.tensor([3] * 8 + [2] * 20, dtype=torch.int32, device=device)
assert int(counts.sum()) == 64
valid = torch.cat(
    [
        torch.arange(e * CAP, e * CAP + int(counts[e].item()), device=device)
        for e in range(E)
    ]
)

hidden = torch.randn((E, CAP, H), dtype=torch.bfloat16, device=device)
recv_q_flat = torch.empty((ROWS, H), dtype=torch.float8_e4m3fn, device=device)
token_scales_flat = torch.empty((ROWS, 1), dtype=torch.float32, device=device)
sgl_per_token_quant_fp8(
    hidden.reshape(ROWS, H), recv_q_flat, token_scales_flat
)
recv_q = recv_q_flat.view(E, CAP, H)
token_scales = token_scales_flat.view(E, CAP)
# New F1 repeats the exact B1 per-token scale in DeepEP's legacy group-128
# carrier, whose last two dimensions are column-major.
recv_group_scales = (
    token_scales.unsqueeze(-1)
    .expand(E, CAP, H // 128)
    .permute(0, 2, 1)
    .contiguous()
    .permute(0, 2, 1)
)
assert not recv_group_scales.is_contiguous()


def make_weight(n, k):
    raw_w = torch.randint(
        0, 256, (E, n, k // 2), dtype=torch.uint8, device=device
    )
    raw_s = torch.randint(
        120, 133, (E, n, k // 32), dtype=torch.uint8, device=device
    )
    return llop.preprocess_weight(raw_w, raw_s)


w13, w13_off, w13_res = make_weight(2 * I, H)
w2, w2_off, w2_res = make_weight(H, I)


def workspace():
    return {
        "padded": torch.empty((E + 1,), dtype=torch.int32, device=device),
        "compact": torch.empty((E + 1,), dtype=torch.int32, device=device),
        "te": torch.empty((ROWS,), dtype=torch.int32, device=device),
        "tn": torch.empty((ROWS,), dtype=torch.int32, device=device),
        "nt": torch.empty((1,), dtype=torch.int32, device=device),
        "q1": torch.empty((ROWS, H), dtype=torch.float8_e4m3fn, device=device),
        "q1s": torch.empty((ROWS, 1), dtype=torch.float32, device=device),
        "fc1ts": torch.empty((ROWS,), dtype=torch.float32, device=device),
        "gate_up": torch.empty((ROWS, 2 * I), dtype=torch.bfloat16, device=device),
        "q2": torch.empty((ROWS, I), dtype=torch.float8_e4m3fn, device=device),
        "q2s": torch.empty((ROWS, 1), dtype=torch.float32, device=device),
        "fc2ts": torch.empty((ROWS,), dtype=torch.float32, device=device),
        "out": torch.full(
            (ROWS, H), torch.nan, dtype=torch.bfloat16, device=device
        ),
    }


b1 = workspace()
f1 = workspace()
hidden_flat = hidden.reshape(ROWS, H)


def b1_quant_all_padded():
    sgl_per_token_quant_fp8(hidden_flat, b1["q1"], b1["q1s"])


def f1_compact_valid_rows():
    llop.compact_deepep_per_token_fp8_out(
        recv_q,
        recv_group_scales,
        counts,
        f1["compact"],
        f1["q1"],
        f1["q1s"],
    )


def b1_full_post_dispatch():
    b1_quant_all_padded()
    llop.deepep_moe_out(
        b1["q1"],
        b1["q1s"],
        w13,
        w13_off,
        w13_res,
        w2,
        w2_off,
        w2_res,
        counts,
        b1["padded"],
        b1["compact"],
        b1["te"],
        b1["tn"],
        b1["nt"],
        b1["fc1ts"],
        b1["gate_up"],
        b1["q2"],
        b1["q2s"],
        b1["fc2ts"],
        b1["out"],
        CAP,
        H,
        I,
        PERSISTENT_CTAS,
        BETA,
        LINEAR_BETA,
    )


def f1_full_post_dispatch():
    llop.deepep_per_token_fp8_moe_out(
        recv_q,
        recv_group_scales,
        w13,
        w13_off,
        w13_res,
        w2,
        w2_off,
        w2_res,
        counts,
        f1["padded"],
        f1["compact"],
        f1["te"],
        f1["tn"],
        f1["nt"],
        f1["q1"],
        f1["q1s"],
        f1["fc1ts"],
        f1["gate_up"],
        f1["q2"],
        f1["q2s"],
        f1["fc2ts"],
        f1["out"],
        CAP,
        H,
        I,
        PERSISTENT_CTAS,
        BETA,
        LINEAR_BETA,
    )


llop.prepare_deepep_layout_out(
    counts,
    CAP,
    f1["padded"],
    f1["compact"],
    f1["te"],
    f1["tn"],
    f1["nt"],
)
for _ in range(5):
    b1_full_post_dispatch()
    f1_full_post_dispatch()
torch.cuda.synchronize()

a = b1["out"][valid].float()
b = f1["out"][valid].float()
diff = a - b
correctness = {
    "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
    "relative_l2": float(
        torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(a)
    ),
    "cosine": float(
        torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)
    ),
    "max_abs": float(diff.abs().max()),
}


def error_metrics(reference, actual):
    delta = reference - actual
    return {
        "relative_l2": float(
            torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(reference)
        ),
        "cosine": float(
            torch.nn.functional.cosine_similarity(
                reference.flatten(), actual.flatten(), dim=0
            )
        ),
        "max_abs": float(delta.abs().max()),
    }


hidden_valid = hidden_flat[valid].float()
b1_input_dequant = b1["q1"][valid].float() * b1["q1s"][valid]
communication_input_dequant = (
    recv_q.float() * token_scales.unsqueeze(-1)
).view(ROWS, H)[valid]
valid_rows = int(counts.sum())
f1_input_dequant = (
    f1["q1"][:valid_rows].float() * f1["q1s"][:valid_rows]
)
input_quantization = {
    "b1_per_token_vs_bf16": error_metrics(hidden_valid, b1_input_dequant),
    "communication_per_token_vs_bf16": error_metrics(
        hidden_valid, communication_input_dequant
    ),
    "f1_compact_vs_b1": error_metrics(
        b1_input_dequant, f1_input_dequant
    ),
}


def capture(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    graph.replay()
    torch.cuda.synchronize()
    return graph


def time_graph(graph):
    samples = []
    for _ in range(REPETITIONS):
        for _ in range(WARMUP):
            graph.replay()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(ITERS):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / ITERS)
    return {
        "samples": samples,
        "mean": statistics.mean(samples),
        "stdev": statistics.stdev(samples),
    }


graphs = {
    "b1_quant_all_padded": capture(b1_quant_all_padded),
    "f1_compact_valid_rows": capture(f1_compact_valid_rows),
    "b1_full_post_dispatch": capture(b1_full_post_dispatch),
    "f1_full_post_dispatch": capture(f1_full_post_dispatch),
}
if os.environ.get("K3_PROFILE_ONLY") == "1":
    profile_replays = int(os.environ.get("K3_PROFILE_REPLAYS", "10"))
    for name, graph in graphs.items():
        torch.cuda.nvtx.range_push(name)
        for _ in range(profile_replays):
            graph.replay()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
    print(
        json.dumps(
            {"profile_replays": profile_replays, "ranges": list(graphs)}
        )
    )
    raise SystemExit(0)
latency_us = {name: time_graph(graph) for name, graph in graphs.items()}
b1_full = latency_us["b1_full_post_dispatch"]["mean"]
f1_full = latency_us["f1_full_post_dispatch"]["mean"]

result = {
    "scope": (
        "Post-dispatch MoE only. DeepEP BF16/FP8 communication itself is not "
        "included; B1 begins with padded BF16 and new F1 begins with the exact "
        "B1 per-token FP8 bytes plus repeated per-token scales."
    ),
    "gpu": torch.cuda.get_device_name(0),
    "shape": {
        "experts": E,
        "hidden": H,
        "intermediate": I,
        "capacity": CAP,
        "valid_rows": int(counts.sum()),
        "padded_rows": ROWS,
        "persistent_ctas": PERSISTENT_CTAS,
    },
    "correctness": correctness,
    "input_quantization": input_quantization,
    "latency_us": latency_us,
    "speedup": {"b1_over_f1_full_post_dispatch": b1_full / f1_full},
    "reduction_percent": {"full_post_dispatch": 100.0 * (b1_full - f1_full) / b1_full},
    "iterations_per_repetition": ITERS,
    "repetitions": REPETITIONS,
}
print(json.dumps(result, indent=2, sort_keys=True))
