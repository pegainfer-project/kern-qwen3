#!/usr/bin/env python3
"""Run kern test against an independently compiled FMA-enabled reference.

This checks compiler arithmetic sensitivity, not an optimization claim or an
independent model oracle. accuracy.py supplies the Transformers model oracle.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check(args):
    bundle = args.bundle.resolve()
    build = json.loads((bundle / "build.json").read_text())
    reference = json.loads((bundle / "model.json").read_text())
    out = ROOT / "build/reference"
    out.mkdir(parents=True, exist_ok=True)
    cubin = out / "qwen3.cubin"
    subprocess.run(
        [
            args.nvcc,
            "--cubin",
            "-O3",
            "-std=c++17",
            f"-arch={build['architecture']}",
            "source/qwen3.cu",
            "-o",
            str(cubin),
        ],
        cwd=ROOT,
        check=True,
    )
    digest = hashlib.sha256(cubin.read_bytes()).hexdigest()
    filename = f"{digest}.cubin"
    (bundle / "kernels" / filename).write_bytes(cubin.read_bytes())
    reference["modules"].pop("qwen3")
    reference["modules"]["qwen3_fma"] = {"source": filename, "sha256": digest}
    for op in reference["ops"].values():
        for launch in op["impl"]["launches"]:
            if launch.get("module") == "qwen3":
                launch["module"] = "qwen3_fma"
    path = out / "model.json"
    path.write_text(json.dumps(reference))
    subprocess.run(
        [
            args.kern,
            "test",
            "--config",
            str(bundle / "kern.toml"),
            "--reference",
            str(path),
            "--capacity",
            "128",
            "--prefill",
            "33",
            "--chunk",
            "16",
            "--decode-steps",
            "4",
            "--iters",
            "3",
            "--no-sweep",
            "--out",
            str(args.out),
        ],
        check=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--nvcc", required=True)
    parser.add_argument("--kern", default="kern")
    parser.add_argument("--out", type=Path, default=Path("out/compiler-check.json"))
    check(parser.parse_args())
