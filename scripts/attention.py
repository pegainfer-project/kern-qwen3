"""Pegainfer's FlashInfer attention launch contract, expressed as kern data.

The compiler supplies struct bytes and offsets. One GPU preparation op converts
rectangular page tables to CSR and makes a compact split-KV plan for all layers.
Fixed 64-token splits keep a sequence's reduction order independent of its peers.
"""

SPLIT = 64


def configure(buffers, ops, metadata, *, dim, heads, kvheads, context, tokens, seqs):
    group = heads // kvheads
    if group not in (1, 2, 4, 8):
        raise ValueError(f"FlashInfer attention build does not include GQA group {group}")
    facts = metadata["variants"][str(dim)]
    splits = (context + SPLIT - 1) // SPLIT
    slots = seqs * splits
    tiles = (tokens * group + 63) // 64
    width = context // 16
    b = lambda name, **kw: {"buf": name, **kw}
    i = lambda value: {"i32": value}
    arrays = {
        "indices": ("i32", [seqs * width]),
        "indptr": ("i32", [seqs + 1]),
        "last": ("i32", [seqs]),
        "request": ("i32", [slots + tiles]),
        "tile": ("i32", [slots + tiles]),
        "merge": ("i32", [seqs + 1]),
        "valid": ("u8", [slots]),
        "chunk": ("i32", [1]),
        "partial": ("bf16", [slots, heads * dim]),
        "lse": ("f32", [slots, heads]),
    }
    for name, (dtype, shape) in arrays.items():
        buffers[f"attn.{name}"] = {"dtype": dtype, "shape": shape, "kind": "workspace"}
    names = ["indices", "indptr", "last", "request", "tile", "merge", "valid", "chunk"]
    plan_params = (
        ["in buffer<i32>", "in buffer<i32>"] + [f"out buffer<{arrays[n][0]}>" for n in names] + ["i32"]
    )
    ops["attention_plan"] = {
        "params": plan_params,
        "impl": {
            "launches": [
                {
                    "module": "attention_plan",
                    "entry": "attention_plan",
                    "grid": [1, 1, 1],
                    "block": [256, 1, 1],
                    "params": plan_params + ["i32"] * 5,
                    "args": [{"param": p} for p in range(len(plan_params))]
                    + [i(width), i(SPLIT), i(splits), i(tiles), i(slots)],
                }
            ]
        },
    }
    plan = {
        "label": "attention_plan",
        "op": "attention_plan",
        "args": [b("pages"), b("lengths")] + [b("attn." + n) for n in names] + [{"var": "seqs"}],
    }
    common = ["out buffer<bf16>", "in buffer<bf16>", "in state", "in state"] + ["in buffer<i32>"] * 6
    scalar = {
        "num_qo_heads": i(heads),
        "q_stride_n": i(heads * dim),
        "q_stride_h": i(dim),
        "paged_kv.num_heads": i(kvheads),
        "paged_kv.batch_size": {"var": "seqs"},
        "paged_kv.stride_page": i(16 * 2 * kvheads * dim),
        "paged_kv.stride_n": i(2 * kvheads * dim),
        "paged_kv.stride_h": i(dim),
    }
    pointers = {
        "q": 1,
        "o": 0,
        "paged_kv.k_data": 2,
        "paged_kv.v_data": 3,
        "paged_kv.indices": 4,
        "paged_kv.indptr": 5,
        "paged_kv.last_page_len": 6,
        "request_indices": 7,
        "kv_tile_indices": 8,
        "kv_chunk_size_ptr": 9,
    }

    def pack(kind, fields):
        abi = facts["abi"][f"{kind}_g{group}"]
        offsets = abi["offsets"]
        dynamic, covered = [], set()
        for name, src in fields.items():
            at = offsets[name]
            width = 8 if "param" in src else 4
            covered.update(range(at, at + width))
            dynamic.append({"at": at, **src})
        initial = [
            {"at": at, "u8": value} for at, value in enumerate(abi["bytes"]) if value and at not in covered
        ]
        return {"pack": {"size": abi["size"], "fields": initial + dynamic}}

    def launch(kind, fields, grid, block, shared, entry):
        packed = pack(kind, fields)
        return {
            "module": f"attention_{dim}",
            "entry": entry,
            "grid": grid,
            "block": block,
            "shared_mem": shared,
            "params": [f"bytes<{packed['pack']['size']}>"],
            "args": [packed],
        }

    fields = scalar | {name: {"param": param} for name, param in pointers.items()}
    pf = fields | {
        "q_indptr": {"param": 10},
        "o_indptr": {"param": 10},
        "qo_tile_indices": {"param": 11},
        "max_total_num_rows": {"var": "tokens"},
        "padded_batch_size": {"expr": {"ceil_div": [{"mul": ["tokens", group]}, 64]}},
    }
    ops["attention_prefill"] = {
        "params": common + ["in buffer<i32>"] * 2,
        "impl": {
            "launches": [
                launch(
                    "prefill",
                    pf,
                    [{"ceil_div": [{"mul": ["tokens", group]}, 64]}, 1, kvheads],
                    facts["prefill_block"],
                    facts["prefill_shared"],
                    "prefill",
                )
            ]
        },
    }
    x, y = dim // 8, group
    threads = max(128, x * y)
    z = threads // (x * y)
    tile = 4 if group == 1 else 1
    shared = 2 * 2 * tile * y * z * dim * 2 + max(tile * threads * 8, 2 * y * z * 4)
    dec = fields | {
        "o": {"param": 12},
        "lse": {"param": 13},
        "block_valid_mask": {"param": 10},
        "o_indptr": {"param": 11},
        "padded_batch_size": {"expr": {"mul": ["seqs", splits]}},
    }
    ops["attention_decode"] = {
        "params": common + ["in buffer<u8>", "in buffer<i32>", "out buffer<bf16>", "out buffer<f32>", "i32"],
        "impl": {
            "launches": [
                launch(
                    "decode",
                    dec,
                    [{"mul": ["seqs", splits]}, kvheads, 1],
                    [x, y, z],
                    shared,
                    f"decode_g{group}",
                ),
                {
                    "module": f"attention_{dim}",
                    "entry": facts["merge_entry"],
                    "grid": [{"mul": ["seqs", heads]}, 1, 1],
                    "block": [x, 128 // x, 1],
                    "shared_mem": 4 * (128 // x) * dim * 2 + 128 * 4,
                    "params": [
                        "in buffer<bf16>",
                        "in buffer<f32>",
                        "in buffer<i32>",
                        "out buffer<bf16>",
                        "i64",
                        "i32",
                        "i64",
                        "i32",
                    ],
                    "args": [
                        {"param": 12},
                        {"param": 13},
                        {"param": 11},
                        {"param": 0},
                        {"i64": 0},
                        {"param": 14},
                        {"i64": 0},
                        i(heads),
                    ],
                },
            ]
        },
    }
    # The prefill plan lives after the fixed maximum decode-plan capacity.
    decode_args = [
        b("attn." + n)
        for n in ["indices", "indptr", "last", "request", "tile", "chunk", "valid", "merge", "partial", "lse"]
    ] + [{"var": "seqs"}]
    prefill_args = [b("attn." + n) for n in ["indices", "indptr", "last"]] + [
        b("attn.request", offset=slots * 4),
        b("attn.request", offset=slots * 4),
        b("attn.chunk"),
        b("offsets"),
        b("attn.tile", offset=slots * 4),
    ]
    return plan, decode_args, prefill_args
