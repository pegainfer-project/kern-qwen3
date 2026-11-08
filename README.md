# kern-qwen3

Dense BF16 Qwen3 on the [kern runtime](https://github.com/pegainfer-project/kern):
checkpoint-driven manifests, CUDA sources, a pinned pegainfer submodule, and
numerical gates. Model geometry and KV layout live here; kern is unchanged.

The initial implementation supports single-GPU greedy and stochastic generation, chunked prefill,
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

git submodule update --init third_party/pegainfer third_party/flashinfer
git -C third_party/pegainfer submodule update --init pegainfer-kernels/third_party/flashinfer
uv run python scripts/build.py --sm sm_120 --nvcc /usr/local/cuda/bin/nvcc
uv run python scripts/gen.py --checkpoint /path/to/Qwen3-4B \
  --kernels build/sm_120 --out out/qwen3-4b \
  --context 4096 --max-tokens 256 --max-seqs 32
kern verify out/qwen3-4b/model.json
kern run --config out/qwen3-4b/kern.toml \
  --prompt 'The capital of France is' --steps 32 --capacity 1024
```

Use `--sm all` to compile every architecture in `kernels.toml`, or pass several
SM names. Build outputs contain eight cubins plus `build.json` with compiler version,
upstream revisions, module SHA-256 hashes, template symbols, attention parameter layouts, and sampling storage sizes. The generator verifies those hashes
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

## Sampling

The manifest accepts `temperature`, `top_k`, `top_p`, `min_p`, and `seed` for every
sequence. Temperature 0 selects greedy argmax; positive temperature runs the
FlashInfer chain. For example:

```sh
kern run --config out/qwen3-4b/kern.toml \
  --prompt 'Invent a short name for a friendly robot:' --steps 16 \
  --temperature 0.8 --top-k 20 --top-p 0.9 --min-p 0.05 --seed 123
```

The same fields work in kern-serve completion requests. `top_k < 1`, `top_p=1`,
and `min_p=0` disable their respective filters. Top-k retains ties at its boundary.
Philox uses the request seed, sequence position as offset, and batch row as
subsequence: replay is deterministic for the same batch layout. Moving a request
to a different batch row can change its draws.

The multi-block top-k kernel needs all blocks of a cooperating group resident.
`gen.py` queries the deployment GPU (`--gpu 0` by default), verifies that its SM
matches the cubins, and derives group count and chunk size from its SM count and
opt-in shared memory limit. Static shared memory is read from each cubin; dynamic
storage sizes come from the exact FlashInfer/CUB headers, with no SM103a constants.
Regenerate the manifest for a different GPU resource configuration, even within
the same SM architecture. For offline generation, supply both target facts:
`--sampling-sms <SM-count> --sampling-smem <opt-in-bytes-per-block>`.
The generator rejects insufficient resources. The sampler's vectorized kernels
require vocabulary size divisible by four, as in the supported Qwen3 checkpoints.

## Organization and execution

- `source/qwen3.cu`: RMSNorm, RoPE, embedding, paged cache writes,
  and greedy argmax, all with flat pointer/scalar ABIs.
- `scripts/build.py`: extracts pegainfer's `add_kernel` and `silu_mul_kernel` from
  the pinned Git object, adds C linkage, and compiles their unchanged bodies.
  Sampling uses the separate FlashInfer v0.7.2 submodule. Attention uses pegainfer's pinned nested FlashInfer submodule.
- `source/attention.cu`, `scripts/attention.py`: FlashInfer causal paged prefill,
  Split-KV decode, and merge; compiler-probed aggregate ABIs.
- `source/attention_plan.cu`: per-forward GPU page-table and split-plan preparation.
- `scripts/gen.py`: reads layer/head/MLP sizes, epsilon, RoPE base, vocabulary,
  context limit, and tied embeddings from the checkpoint; binds original
  safetensors names and emits kern schema v7.
- `scripts/sampling.py`, `source/sampling.cu`, `source/sample_glue.cu`: kern's FlashInfer
  softmax → top-k → top-p → min-p sampling chain and its per-sequence fills.
- `tests/` and `scripts/*check.py`, `scripts/accuracy.py`: real-GPU and runtime gates.
- `bench/smoke.toml`, `results/`: workload and measured results.

Matrix products use kern's cuBLASLt BF16 extern. Norms, activation and RoPE preserve
BF16 materialization boundaries used by Transformers. The cache is
`[physical token slot, K/V, KV head, dimension]`, with 16 slots per page.
Each attention row reads its own sequence's page table and causal prefix.

Attention uses the FlashInfer device routines behind pegainfer's paged attention
launchers. Prefill uses tensor-core tiles of 64 packed query rows; decode uses
fixed 64-token KV splits followed by FlashInfer's variable-length merge. Supported
GQA ratios are 1, 2, 4 and 8, with head dimensions 64, 128 and 256. Metadata is
prepared once per forward on the GPU and reused by every layer. No model-specific
runtime changes or host planner callbacks are required.

The default generated limits remain 4096 context tokens, 256 prefill rows, and
32 decode sequences. Regenerate with `--context 40960 --max-tokens 1024` for a
Qwen3-4B bundle configured for 40K context and 1024-token chunks. The checkpoint's
own limit is enforced. The 40K configuration is not a completed long-context
numerical or serving gate; measured attention correctness currently covers 8K.

With kern-serve built from the runtime repository:

```sh
kern-serve --manifest out/qwen3-4b/model.json --kernels out/qwen3-4b/kernels \
  --weights /path/to/Qwen3-4B --served-model-name qwen3-4b \
  --gpu 0 --port 8000 --chunk 256 --capacity 4096 --max-seqs 8
```

`--capacity` is the KV token pool shared by all requests; it is separate from the
per-request context bound and the concurrency limit. `--chunk` can reduce the
manifest's prefill bound but cannot raise it. For the regenerated 40K bundle,
use `--chunk 1024 --capacity 40960` if device memory permits.

See [attention results](results/attention.md) for measured before/after GPU times
and the numerical gates. No HTTP TTFT/TPOT or cross-engine speed claim is made.

## Validation

```sh
uv sync --group test
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

The sampling integration tests and complete batched-model gate need the `kern-llm` Python wheel
built from the same runtime revision. From that runtime checkout:

```sh
uvx maturin build --manifest-path crates/kern-py/Cargo.toml --out /tmp/kern-wheels
```

Then, from this repository:

```sh
uv pip install /tmp/kern-wheels/kern_llm-*.whl
uv run --no-sync python -m unittest discover -s tests -v
uv run --no-sync python scripts/batch_check.py \
  --bundle out/qwen3-4b --checkpoint /path/to/Qwen3-4B
```

This compares batched and singleton logits after distinct prefixes and shuffled
page allocation, at 1/3/17/32 sequences and three steps each. It uses the public
runtime API with host-bound caches to bound test memory. It also runs a sampled 32-sequence forward and checks seeded replay and top-k membership.
The standalone sampling tests cover mixed parameters, candidate sets, 4096 draws,
and CUDA Graph replay. `kern run` exercises complete-model CUDA Graph decode.

To check the HTTP request path with a temporary local server:

```sh
uv run python scripts/http_check.py --server /path/to/kern-serve \
  --bundle out/qwen3-4b --checkpoint /path/to/Qwen3-4B
```

See [results/sampling.md](results/sampling.md) for sampling validation and measured overhead,
and [results/README.md](results/README.md) for the initial forward baseline.

## Licenses

Apache-2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE). The pegainfer submodule
retains its own license and notices. The build uses the pinned upstream revisions and does not modify either checkout.
