# kern-qwen3

Dense BF16 Qwen3 on the [kern runtime](https://github.com/pegainfer-project/kern):
checkpoint-driven manifests, CUDA sources, a pinned pegainfer submodule, and
numerical gates. Model geometry and KV layout live here; kern is unchanged.

The initial implementation supports single-GPU greedy generation, chunked prefill,
and batched decode. Qwen3-4B has been run on one RTX 5070 Ti. Other dense Qwen3
geometries are read from their checkpoint's `config.json`; they have not yet passed
a full-model numerical gate here. MoE, quantized checkpoints, attention bias,
sliding-window attention, and RoPE scaling are rejected.

## Build and generate

Python 3.11+, uv, CUDA 13.3, and a CUDA-compatible NVIDIA driver are required.
Manifest generation uses only the Python standard library. Install kern from the
revision recorded in `kernels.toml`:

```sh
cargo install --git https://github.com/pegainfer-project/kern \
  --rev 248e7aa3ab782d9c1cc80c4e3519dd2368fe45d1 --locked kern-run

git submodule update --init third_party/pegainfer
uv run python scripts/build.py --sm sm_120 --nvcc /usr/local/cuda/bin/nvcc
uv run python scripts/gen.py --checkpoint /path/to/Qwen3-4B \
  --kernels build/sm_120 --out out/qwen3-4b \
  --context 4096 --max-tokens 256 --max-seqs 32
kern verify out/qwen3-4b/model.json
kern run --config out/qwen3-4b/kern.toml \
  --prompt 'The capital of France is' --steps 32 --capacity 1024
```

Use `--sm all` to compile every architecture in `kernels.toml`, or pass several
SM names. Build outputs contain two cubins plus `build.json` with compiler version,
upstream revision and module SHA-256 hashes. The generator verifies those hashes
and bundles the cubins with `model.json` and a local `kern.toml`. Generated output
and checkpoint paths are ignored by Git. A bundle is specific to one SM; select
the matching bundle on each machine.

| Architecture | Build with CUDA 13.3 | Runtime / numerical validation |
|---|---|---|
| SM80, SM86, SM87, SM88, SM89 | Passed | Not run |
| SM90, SM100, SM103, SM110 | Passed | Not run |
| SM120 | Passed | Qwen3-4B, RTX 5070 Ti |
| SM121 | Passed | Not run |

SM75 is excluded because this implementation uses native BF16. Compiling a cubin
checks code generation, not driver, cuBLAS, memory capacity, or numerical behavior
on that hardware. The SM87 build does not establish Jetson software compatibility.

## Organization and execution

- `source/qwen3.cu`: RMSNorm, RoPE, embedding, paged cache writes, causal attention,
  and greedy argmax, all with flat pointer/scalar ABIs.
- `scripts/build.py`: extracts pegainfer's `add_kernel` and `silu_mul_kernel` from
  the pinned Git object, adds C linkage, and compiles their unchanged bodies.
  No nested pegainfer submodules or host framework are needed.
- `scripts/gen.py`: reads layer/head/MLP sizes, epsilon, RoPE base, vocabulary,
  context limit, and tied embeddings from the checkpoint; binds original
  safetensors names and emits kern schema v7.
- `tests/` and `scripts/*check.py`, `scripts/accuracy.py`: real-GPU and runtime gates.
- `bench/smoke.toml`, `results/`: workload and measured results.

Matrix products use kern's cuBLASLt BF16 extern. Norms, activation and RoPE preserve
BF16 materialization boundaries used by Transformers. The cache is
`[physical token slot, K/V, KV head, dimension]`, with 16 slots per page.
Each attention row reads its own sequence's page table and causal prefix.

Attention is a portable baseline: one warp per query/head, FP32 online softmax,
serial traversal of the visible cache. Long-context throughput is a known limit.
There is no claim of performance parity with pegainfer's FlashInfer attention.
No architecture-specific fast path or multi-GPU execution is included.

## Validation

```sh
uv sync --group test
uv run --group test python -m unittest discover -s tests -v
uv run python scripts/check.py --kern kern
uv run --group test python scripts/accuracy.py --kern /path/to/kern \
  --bundle out/qwen3-4b --checkpoint /path/to/Qwen3-4B --out out/accuracy
kern bench --config out/qwen3-4b/kern.toml \
  --workload bench/smoke.toml --out out/bench.json
```

`accuracy.py` records real kern logits, releases the process, then teacher-forces
Transformers along those tokens. Four prompts (English, Chinese, and a longer
prefix), chunk sizes 16/256, and eight steps per case are the default gate.
It rejects non-finite logits and mean KL above 0.01, and records top-1 agreement
and the reference margin at disagreements. GPU tests check cache writes and
attention against contiguous PyTorch SDPA, using shuffled pages, ragged prefixes,
and head dimensions 64/128/256.

The complete batched-model gate additionally needs the `kern-llm` Python wheel
built from the same runtime revision. From that runtime checkout:

```sh
uvx maturin build --manifest-path crates/kern-py/Cargo.toml --out /tmp/kern-wheels
```

Then, from this repository:

```sh
uv pip install /tmp/kern-wheels/kern_llm-*.whl
uv run --no-sync python scripts/batch_check.py \
  --bundle out/qwen3-4b --checkpoint /path/to/Qwen3-4B
```

This compares batched and singleton logits after distinct prefixes and shuffled
page allocation, at 1/3/17/32 sequences and three steps each. It uses the public
runtime API with host-bound caches to bound test memory. It tests eager runtime
execution; `kern run` separately exercises CUDA Graph decode.

See [results/README.md](results/README.md) for measured evidence and limitations.

## Licenses

Apache-2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE). The pegainfer submodule
retains its own license and notices. The build reads only the pinned committed
source; it does not modify the upstream checkout.
