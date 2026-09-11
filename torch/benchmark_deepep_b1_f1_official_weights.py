import json
import statistics

import torch
from safetensors import safe_open

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
grouped = hidden.float().view(E, CAP, H // 128, 128)
group_scales_contiguous = (grouped.abs().amax(dim=-1) / 448.0).clamp_min(1.0e-12)
recv_q = (grouped / group_scales_contiguous.unsqueeze(-1)).to(
    torch.float8_e4m3fn
).view(E, CAP, H)
# Match DeepEP's documented column-major layout for the last two dimensions.
recv_group_scales = (
    group_scales_contiguous.permute(0, 2, 1).contiguous().permute(0, 2, 1)
)
assert not recv_group_scales.is_contiguous()


MODEL = (
    "/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/ziqingc/"
    "05_claude_ws/models/Kimi-K3-official-f831ab6"
)
SHARD = f"{MODEL}/model-00013-of-000096.safetensors"
LAYER = 12
PREFIX = f"language_model.model.layers.{LAYER}.block_sparse_moe.experts"


def load_expert_tensor(handle, expert, projection, suffix):
    key = f"{PREFIX}.{expert}.{projection}.{suffix}"
    return handle.get_tensor(key)


with safe_open(SHARD, framework="pt", device="cpu") as handle:
    raw_w1 = torch.stack(
        [load_expert_tensor(handle, e, "w1", "weight_packed") for e in range(E)]
    ).to(device)
    raw_s1 = torch.stack(
        [load_expert_tensor(handle, e, "w1", "weight_scale") for e in range(E)]
    ).to(device)
    raw_w3 = torch.stack(
        [load_expert_tensor(handle, e, "w3", "weight_packed") for e in range(E)]
    ).to(device)
    raw_s3 = torch.stack(
        [load_expert_tensor(handle, e, "w3", "weight_scale") for e in range(E)]
    ).to(device)
    raw_w2 = torch.stack(
        [load_expert_tensor(handle, e, "w2", "weight_packed") for e in range(E)]
    ).to(device)
    raw_s2 = torch.stack(
        [load_expert_tensor(handle, e, "w2", "weight_scale") for e in range(E)]
    ).to(device)

raw_w13 = torch.cat((raw_w1, raw_w3), dim=1).contiguous()
raw_s13 = torch.cat((raw_s1, raw_s3), dim=1).contiguous()
w13, w13_off, w13_res = llop.preprocess_weight(raw_w13, raw_s13)
w2, w2_off, w2_res = llop.preprocess_weight(raw_w2, raw_s2)
del raw_w1, raw_s1, raw_w3, raw_s3, raw_w13, raw_s13, raw_w2, raw_s2


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


def f1_requant_valid_rows():
    llop.requantize_deepep_fp8_compact_out(
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
    llop.deepep_fp8_moe_out(
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
group_dequant_padded = (
    recv_q.view(E, CAP, H // 128, 128).float()
    * group_scales_contiguous.unsqueeze(-1)
).view(ROWS, H)
group_dequant_valid = group_dequant_padded[valid]
valid_rows = int(counts.sum())
f1_input_dequant = (
    f1["q1"][:valid_rows].float() * f1["q1s"][:valid_rows]
)
input_quantization = {
    "b1_per_token_vs_bf16": error_metrics(hidden_valid, b1_input_dequant),
    "deepep_group128_vs_bf16": error_metrics(hidden_valid, group_dequant_valid),
    "f1_requant_vs_bf16": error_metrics(hidden_valid, f1_input_dequant),
    "f1_requant_vs_deepep_group128": error_metrics(
        group_dequant_valid, f1_input_dequant
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
    "f1_requant_valid_rows": capture(f1_requant_valid_rows),
    "b1_full_post_dispatch": capture(b1_full_post_dispatch),
    "f1_full_post_dispatch": capture(f1_full_post_dispatch),
}
latency_us = {name: time_graph(graph) for name, graph in graphs.items()}
b1_full = latency_us["b1_full_post_dispatch"]["mean"]
f1_full = latency_us["f1_full_post_dispatch"]["mean"]

result = {
    "scope": (
        "Post-dispatch MoE only. DeepEP BF16/FP8 communication itself is not "
        "included; B1 begins with padded BF16 and F1 begins with DeepEP "
        "group-128 FP8."
    ),
    "gpu": torch.cuda.get_device_name(0),
    "weight_source": {
        "checkpoint": MODEL,
        "layer": LAYER,
        "experts": [0, E],
    },
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
