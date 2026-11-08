#!/usr/bin/env python3
"""Compare batched and singleton decode through kern's public Python runtime.

Each sequence has a different prefix and shuffled physical pages. This exercises
all layers, weights, the actual manifest wiring, and state isolation. Install the
kern-llm wheel from the runtime revision in kernels.toml before running.
"""

import argparse
import json
import tempfile
from pathlib import Path

import kern
import torch
from kern.host import host_region, region


def check(args):
    torch.manual_seed(928)
    m = json.loads((args.bundle / "model.json").read_text())
    c = json.loads((args.checkpoint / "config.json").read_text())
    maximum = m["vars"]["seqs"]["max"]
    groups_to_test = sorted({1, min(3, maximum), min(17, maximum), maximum})
    heads, dim = c["num_key_value_heads"], c["head_dim"]
    count = maximum * 4 + 1
    # Host bindings bound cache memory for the test, preserving the generated layout.
    caches = {
        name: torch.zeros(count, 16, 2, heads, dim, device="cuda", dtype=torch.bfloat16)
        for name in m["states"]
    }
    for name, tensor in caches.items():
        m["states"][name] = {
            "host": {"dtype": "bf16", "shape": [0, 16, 2, heads, dim], "strides": list(tensor.stride())}
        }
    with tempfile.TemporaryDirectory() as directory:
        manifest = Path(directory) / "model.json"
        manifest.write_text(json.dumps(m))
        rt = kern.Runtime(str(manifest), str(args.bundle / "kernels"), [str(args.checkpoint)], 0)
    rt.bind_host({name: host_region(t) for name, t in caches.items()})
    buffers = {name: region(rt, m, name) for name in m["buffers"] if m["buffers"][name]["kind"] != "weight"}
    pages = (torch.randperm(count - 1, device="cuda") + 1).to(torch.int32).view(maximum, 4)
    prefixes = [torch.randint(10, 10000, (2 + seq % 29,), device="cuda") for seq in range(maximum)]

    def stage(sequences, tokens, positions, offsets):
        n, groups = len(tokens), len(sequences)
        buffers["token_ids"][:n].copy_(tokens)
        buffers["positions"][:n].copy_(positions)
        buffers["offsets"][: groups + 1].copy_(torch.tensor(offsets, device="cuda", dtype=torch.int32))
        for j, seq in enumerate(sequences):
            buffers["pages"][j, :4].copy_(pages[seq])
            lo, hi = offsets[j : j + 2]
            pos = positions[lo:hi]
            buffers["slots"][lo:hi].copy_(pages[seq, pos // 16].long() * 16 + pos % 16)
            buffers["lengths"][j] = positions[hi - 1] + 1
        torch.cuda.synchronize()
        return {"tokens": n, "seqs": groups}

    for seq, ids in enumerate(prefixes):
        pos = torch.arange(len(ids), device="cuda")
        rt.run("prefill", stage([seq], ids, pos, [0, len(ids)]))
    results = []
    current = torch.randint(10, 10000, (maximum,), device="cuda")
    for groups in groups_to_test:
        for step in range(3):
            positions = torch.tensor([len(prefixes[j]) + step for j in range(groups)], device="cuda")
            ids = current[:groups].clone()
            rt.run("decode_batch", stage(list(range(groups)), ids, positions, list(range(groups + 1))))
            batch = buffers["logits"][:groups].float().clone()
            next_ids = buffers["next_token"][:groups].clone()
            individual = []
            for seq in range(groups):
                rt.run("decode", stage([seq], ids[seq : seq + 1], positions[seq : seq + 1], [0, 1]))
                individual.append(buffers["logits"][0].float().clone())
            single = torch.stack(individual)
            a, b = single.log_softmax(-1), batch.log_softmax(-1)
            kl = (a.exp() * (a - b)).sum(-1)
            top = single.topk(2, dim=-1)
            different = single.argmax(-1) != batch.argmax(-1)
            result = {
                "groups": groups,
                "step": step,
                "mean_kl": kl.mean().item(),
                "max_kl": kl.max().item(),
                "top1_matches": (~different).sum().item(),
                "mismatch_margins": (top.values[:, 0] - top.values[:, 1])[different].tolist(),
            }
            results.append(result)
            print(json.dumps(result), flush=True)
            if not torch.isfinite(kl).all() or kl.max().item() > args.kl:
                raise RuntimeError("batched/singleton logits disagree")
            current[:groups] = next_ids
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {"device": torch.cuda.get_device_name(), "threshold": args.kl, "results": results}, indent=2
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("out/batch.json"))
    parser.add_argument("--kl", type=float, default=0.01)
    check(parser.parse_args())
