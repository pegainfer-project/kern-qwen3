"""kern's FlashInfer sampling chain, with resource limits from the target GPU.

Adapted from kern/tools/sampling.py at the runtime revision in kernels.toml.
Template symbols and storage sizes come from the built cubin and CUB headers.
"""

import ctypes

BLOCK = 1024
VEC = 4


def target(architecture, gpu=0, sms=None, shared_mem=None):
    """Resolve a deployment GPU; offline generation requires both resource limits."""
    if (sms is None) != (shared_mem is None):
        raise ValueError("sampling SM count and shared memory must be supplied together")
    if sms is None:
        driver = ctypes.CDLL("libcuda.so.1")

        def checked(status):
            if status:
                raise RuntimeError(f"CUDA driver status {status} while querying sampling resources")

        checked(driver.cuInit(0))
        device = ctypes.c_int()
        checked(driver.cuDeviceGet(ctypes.byref(device), gpu))

        def attribute(number):
            value = ctypes.c_int()
            checked(driver.cuDeviceGetAttribute(ctypes.byref(value), number, device))
            return value.value

        actual = f"sm_{attribute(75)}{attribute(76)}"
        if actual != architecture:
            raise ValueError(
                f"cubins target {architecture}, GPU {gpu} is {actual}; supply offline sampling limits"
            )
        sms, shared_mem = attribute(16), attribute(97)
    if sms < 1 or shared_mem < 1:
        raise ValueError("sampling resource limits must be positive")
    return {"sms": sms, "shared_mem": shared_mem}


def fills():
    """The sampling fills, one entry per sequence of a call."""
    return {
        "temperature": {"dtype": "f32", "shape": ["seqs"], "kind": "input", "fill": "temperature"},
        "top_k": {"dtype": "i32", "shape": ["seqs"], "kind": "input", "fill": "top_k"},
        "top_p": {"dtype": "f32", "shape": ["seqs"], "kind": "input", "fill": "top_p"},
        "min_p": {"dtype": "f32", "shape": ["seqs"], "kind": "input", "fill": "min_p"},
        "seed": {"dtype": "i64", "shape": ["seqs"], "kind": "input", "fill": "seed"},
    }


def args(seq_lens="seq_lens", logits="logits", tokens="next_token"):
    """A call's args: the op's interface is (logits, seq_lens, the five fills, tokens, seqs)."""
    bufs = [logits, seq_lens, "temperature", "top_k", "top_p", "min_p", "seed", tokens]
    return [{"buf": b} for b in bufs] + [{"var": "seqs"}]


def ceil_div(a, b):
    return -(-a // b)


def radix(vocab, metadata, target):
    """CTAs per row, elements per CTA, row groups in flight and the dynamic shared memory of a CTA."""
    fixed = metadata["sizes"]["radix_fixed"]
    static = metadata["entries"]["RadixTopKRenormProbKernel_MultiCTA"]["static_shared"]
    limit = (target["shared_mem"] - static - fixed) // (4 * VEC) * VEC
    if vocab < 1 or vocab % VEC or limit < VEC * BLOCK:
        raise ValueError(
            "sampling requires a positive vocabulary divisible by four and sufficient shared memory"
        )
    ctas = ceil_div(vocab, limit)
    if ctas > target["sms"]:
        raise ValueError(f"sampling requires {ctas} resident CTAs, target has {target['sms']} SMs")
    chunk = ceil_div(ceil_div(vocab, ctas), VEC) * VEC
    return ctas, chunk, target["sms"] // ctas, fixed + chunk * 4


def op(vocab, seqs_max, metadata, target):
    ctas, chunk, groups, radix_smem = radix(vocab, metadata, target)
    chain, glue = "sampling", "sample_glue"
    sizes = metadata["sizes"]

    def launch(v, name, params, block, grid, args, smem=None):
        entry = metadata["entries"][name]["entry"] if v == chain else name
        s = {
            "entry": entry,
            "params": params,
            "block": [block, 1, 1],
            "grid": grid,
            "args": args,
            "module": v,
        }
        if smem is not None:
            s["shared_mem"] = smem
        return s

    def p(i):
        return {"param": i}

    def scr(n):
        return {"scratch": n}

    def i32(v):
        return {"i32": v}

    def i64(v):
        return {"i64": v}

    def f32(v):
        return {"f32": v}

    groups = min(groups, seqs_max)
    S = "seqs"
    logits, seq_lens, temperature, top_k, top_p, min_p, seed, tokens, seqs = (p(i) for i in range(9))
    probs = scr("probs")
    return {
        "params": [
            "in buffer<bf16>",
            "in buffer<i32>",
            "in buffer<f32>",
            "in buffer<i32>",
            "in buffer<f32>",
            "in buffer<f32>",
            "in buffer<i64>",
            "out buffer<i64>",
            "i32",
        ],
        "impl": {
            "scratch": {
                "greedy": {"dtype": "i64", "shape": [S]},
                "probs": {"dtype": "f32", "shape": [S, vocab]},
                "chain_temperature": {"dtype": "f32", "shape": [S]},
                "chain_top_k": {"dtype": "i32", "shape": [S]},
                "offset": {"dtype": "u64", "shape": [S]},
                "states": {"dtype": "u32", "shape": [groups, sizes["state_words"]]},
                "drawn": {"dtype": "i32", "shape": [S]},
                "valid": {"dtype": "u8", "shape": [S]},
            },
            "launches": [
                launch(
                    "qwen3",
                    "argmax",
                    ["in buffer<bf16>", "out buffer<i64>", "i32"],
                    256,
                    [S, 1, 1],
                    [logits, scr("greedy"), i32(vocab)],
                ),
                launch(
                    glue,
                    "kern_sample_cast_bf16",
                    ["in buffer<bf16>", "out buffer<f32>", "i32"],
                    BLOCK,
                    [S, ceil_div(vocab, BLOCK), 1],
                    [logits, probs, i32(vocab)],
                ),
                launch(
                    glue,
                    "kern_sample_prep",
                    [
                        "in buffer<f32>",
                        "in buffer<i32>",
                        "in buffer<i32>",
                        "out buffer<f32>",
                        "out buffer<i32>",
                        "out buffer<u64>",
                        "out buffer<u32>",
                        "i32",
                        "i32",
                        "i32",
                    ],
                    BLOCK,
                    [1, 1, 1],
                    [
                        temperature,
                        top_k,
                        seq_lens,
                        scr("chain_temperature"),
                        scr("chain_top_k"),
                        scr("offset"),
                        scr("states"),
                        seqs,
                        i32(vocab),
                        i32(groups * sizes["state_words"]),
                    ],
                ),
                launch(
                    chain,
                    "OnlineSoftmaxFusedKernel",
                    ["inout buffer<f32>", "inout buffer<f32>", "in buffer<f32>", "f32", "i32"],
                    BLOCK,
                    [S, 1, 1],
                    [probs, probs, scr("chain_temperature"), f32(1.0), i32(vocab)],
                    sizes["softmax"],
                ),
                launch(
                    chain,
                    "RadixTopKRenormProbKernel_MultiCTA",
                    [
                        "inout buffer<f32>",
                        "inout buffer<f32>",
                        "in buffer<i32>",
                        "i32",
                        "i32",
                        "i32",
                        "inout buffer<u32>",
                        "i32",
                        "i32",
                    ],
                    BLOCK,
                    [{"mul": [{"clamp": [S, 0, groups]}, ctas]}, 1, 1],
                    [
                        probs,
                        probs,
                        scr("chain_top_k"),
                        i32(0),
                        i32(vocab),
                        seqs,
                        scr("states"),
                        i32(chunk),
                        i32(ctas),
                    ],
                    radix_smem,
                ),
                launch(
                    chain,
                    "TopPRenormProbKernel",
                    ["inout buffer<f32>", "inout buffer<f32>", "in buffer<f32>", "f32", "i32"],
                    BLOCK,
                    [S, 1, 1],
                    [probs, probs, top_p, f32(0.0), i32(vocab)],
                    sizes["topp"],
                ),
                # indices null (a row is its block), the scalar seed /
                # offset / min-p unread behind their arrays, strides 1
                launch(
                    chain,
                    "MinPSamplingFromProbKernel",
                    [
                        "inout buffer<f32>",
                        "in buffer<f32>",
                        "out buffer<i32>",
                        "out buffer<u8>",
                        "i64",
                        "f32",
                        "i32",
                        "in buffer<i64>",
                        "i64",
                        "in buffer<u64>",
                        "i64",
                        "i32",
                        "i32",
                    ],
                    BLOCK,
                    [S, 1, 1],
                    [
                        probs,
                        min_p,
                        scr("drawn"),
                        scr("valid"),
                        i64(0),
                        f32(0.0),
                        i32(vocab),
                        seed,
                        i64(0),
                        scr("offset"),
                        i64(0),
                        i32(1),
                        i32(1),
                    ],
                    sizes["minp"],
                ),
                launch(
                    glue,
                    "kern_sample_select",
                    [
                        "in buffer<f32>",
                        "in buffer<i32>",
                        "in buffer<i64>",
                        "in buffer<u8>",
                        "out buffer<i64>",
                        "i32",
                    ],
                    BLOCK,
                    [1, 1, 1],
                    [temperature, scr("drawn"), scr("greedy"), scr("valid"), tokens, seqs],
                ),
            ],
        },
    }
