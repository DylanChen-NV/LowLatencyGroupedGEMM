import json
import math
import time
import torch

import low_latency_mxfp4 as llop
from sglang.kernels.ops.quantization import sgl_per_token_quant_fp8
from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import situ_and_mul

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

torch.manual_seed(42)
device = torch.device('cuda')
assert torch.cuda.get_device_capability(0) == (9, 0)

# 64 routed rows/rank approximates BS128 * topk16 / EP32.
counts = torch.tensor([3] * 8 + [2] * 20, dtype=torch.int32, device=device)
assert int(counts.sum()) == 64
padded_offsets = torch.arange(E + 1, dtype=torch.int32, device=device) * CAP
valid = torch.cat([
    torch.arange(e * CAP, e * CAP + int(counts[e].item()), device=device)
    for e in range(E)
])

hidden = torch.randn((ROWS, H), dtype=torch.bfloat16, device=device)
q1 = torch.empty((ROWS, H), dtype=torch.float8_e4m3fn, device=device)
q1_scales = torch.empty((ROWS, 1), dtype=torch.float32, device=device)

def make_weight(n, k):
    raw_w = torch.randint(0, 256, (E, n, k // 2), dtype=torch.uint8, device=device)
    raw_s = torch.randint(120, 133, (E, n, k // 32), dtype=torch.uint8, device=device)
    return llop.preprocess_weight(raw_w, raw_s)

w13, w13_off, w13_res = make_weight(2 * I, H)
w2, w2_off, w2_res = make_weight(H, I)

# B0 strided workspaces.
b0_counts = torch.empty((E,), dtype=torch.int32, device=device)
b0_te = torch.empty((ROWS,), dtype=torch.int32, device=device)
b0_tn = torch.empty_like(b0_te)
b0_nt = torch.empty((1,), dtype=torch.int32, device=device)
b0_fc1_ts = torch.empty((ROWS,), dtype=torch.float32, device=device)
b0_gate_up = torch.empty((ROWS, 2 * I), dtype=torch.bfloat16, device=device)
b0_act = torch.empty((ROWS, I), dtype=torch.bfloat16, device=device)
b0_q2 = torch.empty((ROWS, I), dtype=torch.float8_e4m3fn, device=device)
b0_q2s = torch.empty((ROWS, 1), dtype=torch.float32, device=device)
b0_fc2_ts = torch.empty((ROWS,), dtype=torch.float32, device=device)
b0_out = torch.full((ROWS, H), torch.nan, dtype=torch.bfloat16, device=device)

# B1 compact-internal workspaces.
b1_padded = torch.empty((E + 1,), dtype=torch.int32, device=device)
b1_compact = torch.empty_like(b1_padded)
b1_te = torch.empty((ROWS,), dtype=torch.int32, device=device)
b1_tn = torch.empty_like(b1_te)
b1_nt = torch.empty((1,), dtype=torch.int32, device=device)
b1_fc1_ts = torch.empty((ROWS,), dtype=torch.float32, device=device)
b1_gate_up = torch.empty((ROWS, 2 * I), dtype=torch.bfloat16, device=device)
b1_q2 = torch.empty((ROWS, I), dtype=torch.float8_e4m3fn, device=device)
b1_q2s = torch.empty((ROWS, 1), dtype=torch.float32, device=device)
b1_fc2_ts = torch.empty((ROWS,), dtype=torch.float32, device=device)
b1_out = torch.full((ROWS, H), torch.nan, dtype=torch.bfloat16, device=device)

def quant1():
    sgl_per_token_quant_fp8(hidden, q1, q1_scales)

def b0_inner():
    llop.grouped_gemm_out_precomputed_counts(
        q1, q1_scales, w13, w13_off, w13_res, padded_offsets,
        counts, b0_fc1_ts, b0_te, b0_tn, b0_nt, b0_gate_up,
        2 * I, H, PERSISTENT_CTAS,
    )
    situ_and_mul(b0_act, b0_gate_up, BETA, LINEAR_BETA)
    sgl_per_token_quant_fp8(b0_act, b0_q2, b0_q2s)
    llop.grouped_gemm_out_precomputed_schedule(
        b0_q2, b0_q2s, w2, w2_off, w2_res, padded_offsets,
        counts, b0_fc2_ts, b0_te, b0_tn, b0_nt, b0_out,
        H, I, PERSISTENT_CTAS,
    )

def b1_inner():
    llop.deepep_moe_out(
        q1, q1_scales, w13, w13_off, w13_res, w2, w2_off, w2_res,
        counts, b1_padded, b1_compact, b1_te, b1_tn, b1_nt,
        b1_fc1_ts, b1_gate_up, b1_q2, b1_q2s, b1_fc2_ts, b1_out,
        CAP, H, I, PERSISTENT_CTAS, BETA, LINEAR_BETA,
    )

def b0_full():
    quant1()
    b0_inner()

def b1_full():
    quant1()
    b1_inner()

for _ in range(5):
    b0_full()
    b1_full()
torch.cuda.synchronize()

# Check only valid padded rows; both paths intentionally leave padding undefined.
b0_inner()
b1_inner()
torch.cuda.synchronize()
a = b0_out[valid].float()
b = b1_out[valid].float()
diff = a - b
rel_l2 = float(torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(a))
cosine = float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))
max_abs = float(diff.abs().max())
finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())


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
    return start.elapsed_time(end) * 1000.0 / ITERS

graphs = {
    'b0_inner': capture(b0_inner),
    'b1_inner': capture(b1_inner),
    'b0_full': capture(b0_full),
    'b1_full': capture(b1_full),
}
lat = {name: time_graph(graph) for name, graph in graphs.items()}
result = {
    'gpu': torch.cuda.get_device_name(0),
    'shape': {'experts': E, 'hidden': H, 'intermediate': I, 'capacity': CAP,
              'valid_rows': int(counts.sum()), 'padded_rows': ROWS,
              'persistent_ctas': PERSISTENT_CTAS},
    'correctness': {'finite': finite, 'relative_l2': rel_l2,
                    'cosine': cosine, 'max_abs': max_abs},
    'latency_us': lat,
    'speedup': {
        'inner_b0_over_b1': lat['b0_inner'] / lat['b1_inner'],
        'full_b0_over_b1': lat['b0_full'] / lat['b1_full'],
    },
    'iterations': ITERS,
}
print(json.dumps(result, indent=2, sort_keys=True))
