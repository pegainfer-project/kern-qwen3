#!/usr/bin/env python3
"""Build portable cubins and record their hashes, compiler and upstream revision."""

import argparse
import hashlib
import json
import shutil
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def extract_function(source, name):
    start = source.index(f"__global__ void {name}(")
    opening = source.index("{", start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return 'extern "C" ' + source[start:end]


def build(architectures, nvcc):
    config = tomllib.loads((ROOT / "kernels.toml").read_text())
    upstream = config["upstreams"]["pegainfer"]
    path = ROOT / upstream["path"]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
    if revision != upstream["revision"]:
        raise ValueError(f"pegainfer revision: expected {upstream['revision']}, got {revision}")
    relative = "pegainfer-kernels/csrc/shared/elementwise.cu"
    source = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=path, text=True)
    generated = ROOT / "build/upstream/elementwise.cu"
    generated.parent.mkdir(parents=True, exist_ok=True)
    # Compile only the two kernels used here; their function bodies are unchanged.
    generated.write_text(
        "#include <cuda_bf16.h>\n"
        + "\n".join(extract_function(source, name) for name in ("add_kernel", "silu_mul_kernel"))
    )
    compiler = subprocess.check_output([nvcc, "--version"], text=True).strip()
    supported = subprocess.check_output([nvcc, "--list-gpu-code"], text=True).split()
    for sm in architectures:
        if sm not in config["architectures"] or sm not in supported:
            raise ValueError(f"unsupported architecture {sm}; compiler supports {supported}")
    for sm in architectures:
        out = ROOT / "build" / sm
        out.mkdir(parents=True, exist_ok=True)
        modules = {}
        for name, src in (("qwen3", "source/qwen3.cu"), ("elementwise", "build/upstream/elementwise.cu")):
            target = out / f"{name}.cubin"
            command = [
                nvcc,
                "--cubin",
                "-std=c++17",
                "-O3",
                "--fmad=false",
                f"-arch={sm}",
                src,
                "-o",
                str(target.relative_to(ROOT)),
            ]
            subprocess.run(command, cwd=ROOT, check=True)
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            modules[name] = {"source": target.name, "sha256": digest}
        (out / "build.json").write_text(
            json.dumps(
                {"architecture": sm, "compiler": compiler, "upstream": revision, "modules": modules}, indent=2
            )
            + "\n"
        )
        print(f"{sm}: {len(modules)} modules built", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sm", nargs="+", required=True, help="SM names, or all")
    parser.add_argument("--nvcc", default=shutil.which("nvcc"))
    args = parser.parse_args()
    if not args.nvcc:
        parser.error("nvcc is required; pass --nvcc or put it on PATH")
    config = tomllib.loads((ROOT / "kernels.toml").read_text())
    build(config["architectures"] if args.sm == ["all"] else args.sm, args.nvcc)
