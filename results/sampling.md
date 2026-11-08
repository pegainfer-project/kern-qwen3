# Sampling validation

Date: 2026-11-08. Device: RTX 5070 Ti × 1, 16 GB, SM120, 70 SMs.
CUDA compiler 13.3.33; kern revision `248e7aa3ab782d9c1cc80c4e3519dd2368fe45d1`.
FlashInfer v0.7.2 (`d84335d8c54e0c8f9f7e22faa15c7e34a58abf94`).

## Implementation

The sample op follows kern's existing chain: greedy argmax, BF16-to-FP32 cast,
per-sequence preparation, temperature softmax, deterministic radix top-k,
top-p renormalization, seeded min-p draw, and temperature-based selection.
An invalid FlashInfer draw fails on the device instead of becoming an output token.

Builds extract each cubin's symbols and static shared memory usage. A host metadata
probe gets storage sizes from the exact FlashInfer/CUB headers. The generator
subtracts static storage from the deployment GPU's shared memory limit and limits
the cooperating top-k blocks to its SM count. On the measured GPU the target facts
are 70 SMs and 101376 bytes of opt-in shared memory per block. These facts are
recorded in each generated bundle's `build.json`; another GPU configuration needs
a regenerated manifest.

All four modules compiled for SM80, SM86, SM87, SM88, SM89, SM90, SM100, SM103,
SM110, SM120 and SM121. [sampling-builds.json](sampling-builds.json) records hashes,
compiler, template symbols and storage sizes. Only SM120 has runtime evidence.

## Correctness

- The actual generated op ran through `kern.Runtime` with vocabularies 32, 1024
  and 151936, each with 65 rows of mixed sampling parameters. This exceeds the
  persistent radix kernel's concurrent row groups on the test GPU.
- Every sampled token stayed within the CPU oracle's filtered candidate set.
  Cases cover greedy, top-k=1, disabled and out-of-range top-k, top-p=0/1,
  min-p=0/1, and combinations. Ties at filter boundaries are retained.
- Identical seeds and offsets replayed identically. Advancing offsets or changing
  seeds changed draws. Eager and CUDA Graph outputs agreed, including graph replay
  after input updates.
- 4096 draws after top-k/top-p/min-p filtering passed the distribution check:
  chi-square 7.3559, acceptance limit 40; no token outside the expected support.
- Full Qwen3-4B batched/singleton logits still passed at 1/3/17/32 sequences.
  A 32-sequence sampled forward at temperature 0.8, top-k 20, top-p 0.9,
  min-p 0.05 replayed identically; all outputs belonged to top-k, and 9/32 differed
  from greedy. See [sampling-batch.json](sampling-batch.json).
- `kern run` with those parameters and seed 123 replayed all 16 tokens identically
  through complete-model CUDA Graph decode. [sampling.json](sampling.json) records
  the tokens and test parameters.
- A temporary local `kern-serve` passed seven completion requests: seed 123 twice,
  seed 124, and four concurrent requests mixing greedy and sampled decoding.
  Same-seed singleton responses matched, different-seed responses differed, and
  both greedy responses in the mixed batch matched. All completed 16 tokens.
  See [sampling-http.json](sampling-http.json). The server was stopped afterward.

Replay promises apply to the same batch layout. FlashInfer includes the row index
in its Philox subsequence; replay across different row assignments is not promised.

## Greedy regression and overhead

`kern test` compared the original argmax implementation (A) with the full sampling
chain at temperature 0 (B), with 33 prefill tokens, chunks of 16, four decode rows,
capacity 128 and three timing iterations. The sample interface and model forward
were identical; A's sample implementation launches only the original argmax.

**PASS: bit-identical at every changed span; noise check 2/2 clean.**

| Measurement | Argmax A | Sampling chain B |
|---|---:|---:|
| Sample span | 20.5 µs | 124.9 µs |
| Decode eager | 13.168 ms | 13.365 ms |
| Decode CUDA Graph / token | 11.686 ms | 11.788 ms |

The complete chain adds about 0.102 ms / 0.9% to greedy graph decode in this small
workload. It currently executes the filters even for greedy rows and selects the
argmax afterward. This is a measured feature cost, not an optimization result.
The raw report is [sampling-ab.json.gz](sampling-ab.json.gz); local manifest paths
were replaced with portable labels. Other values are unchanged.

Reproduce the kernel and HTTP gates using the commands in [../README.md](../README.md).
The pre-existing forward/Transformers baseline remains in [README.md](README.md).

To reconstruct the greedy reference for the A/B gate:

```sh
uv run --no-project python - <<'PY'
import json
from pathlib import Path
m = json.loads(Path('out/qwen3-4b/model.json').read_text())
vocab = m['buffers']['logits']['shape'][1]
m['ops']['sample']['impl'] = {'launches': [{
    'module': 'qwen3', 'entry': 'argmax',
    'params': ['in buffer<bf16>', 'out buffer<i64>', 'i32'],
    'grid': ['seqs', 1, 1], 'block': [256, 1, 1],
    'args': [{'param': 0}, {'param': 7}, {'i32': vocab}]}]}
m['modules'] = {k: v for k, v in m['modules'].items() if k in ('qwen3', 'elementwise')}
Path('out/greedy-reference.json').write_text(json.dumps(m))
PY
kern test --config out/qwen3-4b/kern.toml --reference out/greedy-reference.json \
  --capacity 128 --prefill 33 --chunk 16 --decode-steps 4 --iters 3 \
  --no-sweep --out out/sampling-ab.json
```
