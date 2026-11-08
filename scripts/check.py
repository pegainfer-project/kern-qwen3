#!/usr/bin/env python3
"""Check submodule provenance, every built cubin hash, and generated manifests."""

import argparse
import hashlib
import json
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check(kern):
    config = tomllib.loads((ROOT / "kernels.toml").read_text())
    for name, upstream in config["upstreams"].items():
        actual = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT / upstream["path"], text=True
        ).strip()
        if actual != upstream["revision"]:
            raise ValueError(f"{name} mismatch: expected {upstream['revision']}, got {actual}")
    records = sorted((ROOT / "build").glob("sm_*/build.json"))
    if not records:
        raise ValueError("no builds found; run scripts/build.py first")
    for record in records:
        build = json.loads(record.read_text())
        for module in build["modules"].values():
            source = record.parent / module["source"]
            if hashlib.sha256(source.read_bytes()).hexdigest() != module["sha256"]:
                raise ValueError(f"hash mismatch: {source}")
        print(f"{build['architecture']}: hashes OK")
    manifests = sorted((ROOT / "out").glob("*/model.json"))
    if not manifests:
        raise ValueError("no manifests found; run scripts/gen.py first")
    for path in manifests:
        subprocess.run([kern, "verify", str(path)], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kern", default="kern")
    check(parser.parse_args().kern)
