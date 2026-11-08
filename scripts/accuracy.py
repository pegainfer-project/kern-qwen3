#!/usr/bin/env python3
"""Compare actual kern logits with transformers on the same teacher-forced path.

Runs kern first and releases it before loading transformers, so the gate fits
on one GPU. Probe runs are eager; a separate graph-mode smoke is also required.
"""

import argparse
import json
import subprocess
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "The capital of France is",
    "Explain how a database B-tree stores keys.",
    "计算 17 加 25，并解释计算过程。",
    "A database stores pages on disk. " * 40,
]


def check(args):
    if args.steps < 1 or not args.chunks or min(args.chunks) < 1 or args.kl <= 0:
        raise ValueError("steps, chunks and KL threshold must be positive")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    vocab = json.loads((Path(args.checkpoint) / "config.json").read_text())["vocab_size"]
    traces = []
    prompts = json.loads(args.prompt_file.read_text()) if args.prompt_file else PROMPTS
    for number, prompt in enumerate(prompts):
        for chunk in args.chunks:
            directory = args.out / f"prompt{number}-chunk{chunk}"
            directory.mkdir(parents=True, exist_ok=True)
            config_path = directory / "input.toml"
            config_path.write_text(
                "[targets.qwen3]\n"
                + f"manifest = {json.dumps(str((args.bundle / 'model.json').resolve()))}\n"
                + f"kernels = {json.dumps(str((args.bundle / 'kernels').resolve()))}\n"
                + f"weights = [{json.dumps(str(Path(args.checkpoint).resolve()))}]\n"
                + f"[run]\nprompt = {json.dumps(prompt, ensure_ascii=False)}\n"
            )
            command = [
                str(args.kern),
                "run",
                "--config",
                str(config_path),
                "--steps",
                str(args.steps),
                "--capacity",
                str(args.capacity),
                "--chunk",
                str(chunk),
                "--probe-dir",
                str(directory),
                "--probe-labels",
                "",
                "--probe-steps",
                str(args.steps),
            ]
            subprocess.run(command, check=True)
            tokens, logits = [], []
            for step in range(args.steps):
                tokens.append(
                    torch.frombuffer(
                        bytearray((directory / f"decode{step}.tokens.bin").read_bytes()), dtype=torch.int64
                    )[0].item()
                )
                logits.append(
                    torch.frombuffer(
                        bytearray((directory / f"decode{step}.logits.bin").read_bytes()), dtype=torch.bfloat16
                    )[:vocab]
                )
            traces.append((number, chunk, prompt, tokens, torch.stack(logits).float()))
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.checkpoint,
            dtype=torch.bfloat16,
            attn_implementation=args.reference_attention,
            local_files_only=True,
        )
        .to("cuda")
        .eval()
    )
    results = []
    with torch.inference_mode():
        for number, chunk, prompt, tokens, candidate in traces:
            ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
            reference = []
            past = None
            # Bound reference activations too: a long-prompt oracle must fit on
            # the same device as the candidate, without a second model copy.
            if ids.shape[1] > 2048:
                for part in ids[:, :-1].split(1024, dim=1):
                    output = model(input_ids=part, past_key_values=past, use_cache=True, logits_to_keep=1)
                    past = output.past_key_values
                ids = ids[:, -1:]
            for step in range(args.steps):
                output = model(input_ids=ids, past_key_values=past, use_cache=True, logits_to_keep=1)
                reference.append(output.logits[0, -1].float().cpu())
                past = output.past_key_values
                ids = torch.tensor([[tokens[step]]], device="cuda")
            reference = torch.stack(reference)
            a, b = reference.log_softmax(-1), candidate.log_softmax(-1)
            kl = (a.exp() * (a - b)).sum(-1)
            top = reference.topk(2, dim=-1)
            same = candidate.argmax(-1) == top.indices[:, 0]
            if not torch.isfinite(kl).all():
                raise RuntimeError("non-finite logits or KL divergence")
            result = {
                "prompt": number,
                "chunk": chunk,
                "steps": args.steps,
                "mean_kl": kl.mean().item(),
                "max_kl": kl.max().item(),
                "top1_matches": same.sum().item(),
                "mismatch_margins": (top.values[:, 0] - top.values[:, 1])[~same].tolist(),
            }
            results.append(result)
            print(json.dumps(result), flush=True)
    report = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "checkpoint": Path(args.checkpoint).name,
        "transformers": __import__("transformers").__version__,
        "threshold": args.kl,
        "results": results,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "accuracy.json").write_text(json.dumps(report, indent=2) + "\n")
    if any(row["mean_kl"] > args.kl for row in results):
        raise SystemExit("FAIL: mean KL exceeds the threshold")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kern", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", type=Path, default=Path("out/accuracy"))
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--chunks", type=int, nargs="+", default=[16, 256])
    parser.add_argument("--prompt-file", type=Path, help="JSON list of prompts")
    parser.add_argument("--capacity", type=int, default=1024)
    parser.add_argument("--reference-attention", choices=["eager", "sdpa"], default="eager")
    parser.add_argument("--kl", type=float, default=0.01)
    check(parser.parse_args())
