#!/usr/bin/env python3
"""Exercise sampling request fields through a temporary local kern-serve instance."""

import argparse
import json
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


def check(args):
    args.out.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    command = [
        args.server,
        "--manifest",
        str(args.bundle / "model.json"),
        "--kernels",
        str(args.bundle / "kernels"),
        "--weights",
        str(args.checkpoint),
        "--port",
        str(port),
        "--max-seqs",
        "4",
        "--capacity",
        "2048",
    ]
    with (args.out / "server.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 90
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"server exited with {process.returncode}; see {log.name}")
                try:
                    with urlopen(base + "/v1/models", timeout=1) as response:
                        model = json.load(response)["data"][0]["id"]
                    break
                except URLError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("server did not become ready") from None
                    time.sleep(0.2)

            def complete(seed, temperature=0.8):
                body = {
                    "model": model,
                    "prompt": "Invent a short name for a friendly robot:",
                    "max_tokens": 16,
                    "temperature": temperature,
                    "top_k": 20,
                    "top_p": 0.9,
                    "min_p": 0.05,
                    "seed": seed,
                }
                request = Request(
                    base + "/v1/completions",
                    data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urlopen(request, timeout=60) as response:
                    result = json.load(response)
                return {
                    "seed": seed,
                    "temperature": temperature,
                    "text": result["choices"][0]["text"],
                    "usage": result["usage"],
                }

            first, repeat, changed = complete(123), complete(123), complete(124)
            if first["text"] != repeat["text"]:
                raise AssertionError("same seed did not replay at singleton batch layout")
            if first["text"] == changed["text"]:
                raise AssertionError("seed variation did not change this test continuation")
            with ThreadPoolExecutor(max_workers=4) as pool:
                mixed = list(pool.map(lambda item: complete(*item), [(0, 0), (4, 0.8), (7, 0), (9, 1.1)]))
            report = {
                "same_seed_equal": True,
                "different_seed_changed": True,
                "sequential": [first, repeat, changed],
                "concurrent": mixed,
            }
            (args.out / "http.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report), flush=True)
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="kern-serve")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("out/http"))
    check(parser.parse_args())
