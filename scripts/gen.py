#!/usr/bin/env python3
"""Generate a dense BF16 Qwen3 manifest from checkpoint geometry and built cubins.

Single-GPU, greedy decoding, chunked prefill and batched decode. KV pages contain
16 slots, each [K/V, kv_heads, head_dim]. No model knowledge enters kern.
"""

import argparse
import copy
import hashlib
import json
import math
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def geometry(config, context):
    if config.get("model_type") != "qwen3":
        raise ValueError("expected dense Qwen3 (model_type=qwen3)")
    for name in ("attention_bias", "use_sliding_window", "rope_scaling", "quantization_config"):
        if config.get(name):
            raise ValueError(f"unsupported checkpoint option: {name}")
    if config.get("hidden_act") != "silu":
        raise ValueError("expected hidden_act=silu")
    fields = [
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "intermediate_size",
        "vocab_size",
        "max_position_embeddings",
    ]
    for name in fields:
        if type(config.get(name)) is not int or config[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if config["head_dim"] not in (64, 128, 256):
        raise ValueError("head_dim must be 64, 128 or 256")
    if config["num_attention_heads"] % config["num_key_value_heads"]:
        raise ValueError("query heads must be divisible by KV heads")
    if not 0 < context <= config["max_position_embeddings"] or context % 16:
        raise ValueError("context must be a positive multiple of 16 within the checkpoint's limit")
    for name in ("rms_norm_eps", "rope_theta"):
        value = config.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    return config


def manifest(config, modules, *, context=4096, tokens=256, seqs=32):
    c = geometry(config, context)
    if tokens < seqs or seqs < 1 or tokens > context:
        raise ValueError("expected 1 <= max-seqs <= max-tokens <= context")
    h, layers = c["hidden_size"], c["num_hidden_layers"]
    heads, kvheads, d = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
    qdim, kvdim, f, vocab = heads * d, kvheads * d, c["intermediate_size"], c["vocab_size"]
    b, s, i = lambda x: {"buf": x}, lambda x: {"state": x}, lambda x: {"i32": x}
    rows = {"var": "tokens"}
    groups = {"var": "seqs"}
    buffers = {}

    def buffer(name, dtype, shape, kind="workspace", **kwargs):
        buffers[name] = dict(dtype=dtype, shape=shape, kind=kind, **kwargs)

    def weight(name, shape, tensor=None):
        buffer(name, "bf16", shape, "weight", bind=[{"tensor": tensor or name}])
        return b(name)

    buffer("token_ids", "i64", ["tokens"], "input", fill="token", domain={"min": 0, "max": vocab - 1})
    buffer("positions", "i64", ["tokens"], "input", fill="position", domain={"min": 0, "max": context - 1})
    buffer("slots", "i64", ["tokens"], "input", fill="slot", domain={"index_into": "kv.0"})
    buffer("pages", "i32", ["seqs", context // 16], "input", domain={"index_into": "kv.0", "stride": 16})
    buffer("lengths", "i32", ["seqs"], "input", fill="seq_len", domain={"min": 1, "max": context})
    buffer(
        "offsets",
        "i32",
        [seqs + 1],
        "input",
        fill="cu_seqlens",
        domain={"min": 0, "max": "tokens", "monotone": True},
    )
    buffer("next_token", "i64", ["seqs"], "output", fill="tokens", domain={"min": 0, "max": vocab - 1})
    buffer("logits", "bf16", ["seqs", vocab])
    for name, width in (
        ("h", h),
        ("x", h),
        ("q", qdim),
        ("k", kvdim),
        ("v", kvdim),
        ("attention", qdim),
        ("gate", f),
        ("up", f),
        ("act", f),
    ):
        buffer(name, "bf16", ["tokens", width])

    ops = {
        "gemm": {
            "params": ["in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32", "i32"],
            "impl": {"launches": [{"entry": "extern:cublaslt_bf16_tn"}]},
        }
    }

    def op(name, entry, params, grid, *, extra=(), module="qwen3", block=256):
        ops[name] = {
            "params": params,
            "impl": {
                "launches": [
                    {
                        "module": module,
                        "entry": entry,
                        "params": params + [next(iter(x)) for x in extra],
                        "grid": grid,
                        "block": [block, 1, 1],
                        "args": [{"param": n} for n in range(len(params))] + list(extra),
                    }
                ]
            },
        }

    norm = ["out buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>"]
    op("norm", "rms_norm", norm, ["tokens", 1, 1], extra=[i(h), {"f32": c["rms_norm_eps"]}])
    # Q/K normalize in place; one interface argument is forwarded to both ABI pointers.
    for name, count in (("q_norm", heads), ("k_norm", kvheads)):
        op(name, "rms_norm", ["inout buffer<bf16>", "in buffer<bf16>"], [{"mul": ["tokens", count]}, 1, 1])
        launch = ops[name]["impl"]["launches"][0]
        launch["params"] = norm + ["i32", "f32"]
        launch["args"] = [{"param": 0}, {"param": 0}, {"param": 1}, i(d), {"f32": c["rms_norm_eps"]}]
    op(
        "embedding",
        "embedding",
        ["in buffer<i64>", "in buffer<bf16>", "out buffer<bf16>"],
        ["tokens", 1, 1],
        extra=[i(h), i(vocab)],
    )
    op(
        "rope",
        "rope",
        ["inout buffer<bf16>", "inout buffer<bf16>", "in buffer<i64>"],
        ["tokens", heads + kvheads, 1],
        extra=[i(heads), i(kvheads), i(d), {"f32": c["rope_theta"]}],
    )
    op(
        "cache",
        "cache_write",
        ["in buffer<bf16>", "in buffer<bf16>", "inout state", "in buffer<i64>"],
        ["tokens", 1, 1],
        extra=[i(kvdim)],
    )
    op(
        "attention",
        "attention",
        [
            "out buffer<bf16>",
            "in buffer<bf16>",
            "in state",
            "in buffer<i32>",
            "in buffer<i32>",
            "in buffer<i32>",
            "i32",
        ],
        ["tokens", heads, 1],
        extra=[i(context // 16), i(heads), i(kvheads), i(d)],
        block=32,
    )
    op(
        "add",
        "add_kernel",
        ["in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "i32"],
        [{"ceil_div": [{"mul": ["tokens", h]}, 256]}, 1, 1],
        module="elementwise",
    )
    op(
        "silu",
        "silu_mul_kernel",
        ["in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "i32"],
        [{"ceil_div": [{"mul": ["tokens", f]}, 256]}, 1, 1],
        module="elementwise",
    )
    op("argmax", "argmax", ["in buffer<bf16>", "out buffer<i64>"], ["seqs", 1, 1], extra=[i(vocab)])

    calls = []

    def call(label, name, *args):
        calls.append({"label": label, "op": name, "args": list(args)})

    def gemm(label, src, dst, wname, outdim, indim, tensor=None):
        w = weight(wname, [outdim, indim], tensor)
        call(label, "gemm", b(src), w, b(dst), rows, i(outdim), i(indim))

    embed = weight("model.embed_tokens.weight", [vocab, h])
    call("embed", "embedding", b("token_ids"), embed, b("h"))
    for layer in range(layers):
        p, label = f"model.layers.{layer}", f"l{layer}"
        call(f"{label}.input_norm", "norm", b("x"), b("h"), weight(f"{p}.input_layernorm.weight", [h]))
        for proj, width in (("q", qdim), ("k", kvdim), ("v", kvdim)):
            gemm(f"{label}.{proj}_proj", "x", proj, f"{p}.self_attn.{proj}_proj.weight", width, h)
        for proj in ("q", "k"):
            call(
                f"{label}.{proj}_norm",
                f"{proj}_norm",
                b(proj),
                weight(f"{p}.self_attn.{proj}_norm.weight", [d]),
            )
        call(f"{label}.rope", "rope", b("q"), b("k"), b("positions"))
        call(f"{label}.cache", "cache", b("k"), b("v"), s(f"kv.{layer}"), b("slots"))
        call(
            f"{label}.attention",
            "attention",
            b("attention"),
            b("q"),
            s(f"kv.{layer}"),
            b("pages"),
            b("lengths"),
            b("offsets"),
            groups,
        )
        gemm(f"{label}.o_proj", "attention", "x", f"{p}.self_attn.o_proj.weight", h, qdim)
        call(f"{label}.attention_residual", "add", b("h"), b("x"), b("h"), {"expr": {"mul": ["tokens", h]}})
        call(
            f"{label}.post_norm", "norm", b("x"), b("h"), weight(f"{p}.post_attention_layernorm.weight", [h])
        )
        for proj in ("gate", "up"):
            gemm(f"{label}.{proj}_proj", "x", proj, f"{p}.mlp.{proj}_proj.weight", f, h)
        call(f"{label}.silu", "silu", b("gate"), b("up"), b("act"), {"expr": {"mul": ["tokens", f]}})
        gemm(f"{label}.down_proj", "act", "x", f"{p}.mlp.down_proj.weight", h, f)
        call(f"{label}.residual", "add", b("h"), b("x"), b("h"), {"expr": {"mul": ["tokens", h]}})
    prefill = copy.deepcopy(calls)
    call("final_norm", "norm", b("x"), b("h"), weight("model.norm.weight", [h]))
    head = "model.embed_tokens.weight" if c.get("tie_word_embeddings") else "lm_head.weight"
    # Reuse the embedding buffer for tied checkpoints rather than allocating it twice.
    head_weight = b(head) if c.get("tie_word_embeddings") else weight(head, [vocab, h])
    call("lm_head", "gemm", b("x"), head_weight, b("logits"), groups, i(vocab), i(h))
    call("sample", "argmax", b("logits"), b("next_token"))
    return {
        "schema_version": 7,
        "model": "qwen3",
        "chat": {"tool_parser": "hermes"},
        "vars": {"tokens": {"max": tokens, "axis": "rows"}, "seqs": {"max": seqs, "axis": "groups"}},
        "states": {f"kv.{j}": {"bytes_per_token": 4 * kvdim} for j in range(layers)},
        "buffers": buffers,
        "modules": modules,
        "ops": ops,
        "programs": {
            "prefill": {"batch": {"groups": 1, "rows": "tokens"}, "calls": prefill},
            "decode": {"batch": {"groups": 1, "rows": 1}, "graph": True, "calls": calls},
            "decode_batch": {"batch": {"groups": seqs, "rows": 1}, "graph": True, "calls": calls},
        },
    }


def generate(checkpoint, kernels, out, context, tokens, seqs):
    config = json.loads((checkpoint / "config.json").read_text())
    build = json.loads((kernels / "build.json").read_text())
    m = manifest(config, build["modules"], context=context, tokens=tokens, seqs=seqs)
    out.mkdir(parents=True, exist_ok=True)
    bundle = out / "kernels"
    bundle.mkdir(exist_ok=True)
    for module in m["modules"].values():
        source = kernels / module["source"]
        actual = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual != module["sha256"]:
            raise ValueError(f"hash mismatch for {source.name}: expected {module['sha256']}, got {actual}")
        shutil.copyfile(source, bundle / source.name)
    (out / "model.json").write_text(json.dumps(m, indent=2) + "\n")
    # Local generated configuration is ignored by Git; no private paths enter source docs.
    (out / "kern.toml").write_text(
        '[targets.qwen3]\nmanifest = "model.json"\nkernels = "kernels"\n'
        + f"weights = [{json.dumps(str(checkpoint.resolve()))}]\n"
    )
    (out / "build.json").write_text(json.dumps(build, indent=2) + "\n")
    print(f"Generated {out / 'model.json'} ({config['num_hidden_layers']} layers, {build['architecture']})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--kernels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-seqs", type=int, default=32)
    args = parser.parse_args()
    generate(args.checkpoint, args.kernels, args.out, args.context, args.max_tokens, args.max_seqs)
