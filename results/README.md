# Validation record

Date: 2026-11-08. Hardware: **RTX 5070 Ti × 1**, 16 GB, SM120, 70 SMs.
CUDA compiler: 13.3.33. Driver API: 13.3. Runtime revision:
`248e7aa3ab782d9c1cc80c4e3519dd2368fe45d1` (debug build).
Model: Qwen3-4B, BF16; context bound 4096, maximum chunk 256, maximum batch 32.
Python gates used PyTorch 2.15.0+cu132 and Transformers 5.22.2.

## Numerical gates

- **Manifest verification:** passed, including the serving protocol.
- **Cross-compilation:** both modules built for all 11 architectures in
  `kernels.toml`. [builds.json](builds.json) records compiler and cubin hashes.
  Only SM120 has runtime evidence.
- **Cache + attention integration:** passed against PyTorch SDPA. Shuffled pages,
  ragged causal prefixes, 1/3/17/32 groups, GQA ratios 1/4/8/2, and head dimensions
  64/128/128/256. The reserved page remained untouched.
- **Transformers teacher-forcing:** passed, 4 prompts × 2 chunk sizes × 8 steps.
  Mean KL per case was 0.000614–0.001908 (threshold 0.01); the largest individual
  step KL was 0.007848. Top-1 agreed on 62/64 rows; both disagreements had a zero
  reference top-1/top-2 margin. [accuracy.json](accuracy.json) contains every case.
- **Full-model batch isolation:** passed at 1/3/17/32 sequences × 3 steps, comparing
  batched and singleton decode with distinct prefixes and shuffled physical pages.
  Largest per-sequence KL: 0.002694 (threshold 0.01). Top-1 agreed on 158/159
  rows; the remaining row had a zero singleton top-1/top-2 margin. See
  [batch.json](batch.json).
- **Compiler A/B through `kern test`:** passed. A enables default FP32 contraction;
  B is the shipped `--fmad=false` build. Maximum end-to-end KL: 0.000531 on 4 rows,
  argmax agrees. This is compiler-sensitivity evidence, not an independent model
  oracle or a performance improvement claim. Raw report:
  [compiler-check.json.gz](compiler-check.json.gz). Its two local manifest paths
  were replaced with portable labels; numerical measurements are unchanged.
- **CUDA Graph decode smoke:** produced `Paris. The capital of Germany is Berlin`
  after `The capital of France is` (8 generated tokens).

The compiler A/B gate can be reproduced with:

```sh
uv run python scripts/compiler_check.py --bundle out/qwen3-4b \
  --nvcc /usr/local/cuda/bin/nvcc --kern /path/to/kern
```

## Measured baseline

`kern bench`, [../bench/smoke.toml](../bench/smoke.toml), 12 CUDA-event samples per
scenario, median whole-program graph time. Weights and kernel shapes are those
of the generated Qwen3-4B bundle. This is GPU execution time, not server latency.

| Program | Sequences | New rows / sequence | Existing context | Median ms |
|---|---:|---:|---:|---:|
| Decode | 1 | 1 | 128 | 15.252 |
| Decode | 1 | 1 | 512 | 27.827 |
| Decode | 4 | 1 | 128 | 15.965 |
| Decode | 4 | 1 | 512 | 28.304 |
| Decode | 32 | 1 | 128 | 15.880 |
| Decode | 32 | 1 | 512 | 27.785 |
| Prefill | 1 | 16 | 0 | 11.357 |
| Prefill | 1 | 128 | 0 | 17.999 |

[bench.json.gz](bench.json.gz) retains raw samples, hardware, hashes, workload and
per-call attribution, including the two extra one-row prefill scenarios.
No comparison with another engine was performed. Serial traversal of context in
the portable attention kernel limits long-context throughput. Other model sizes,
other GPUs, stochastic sampling, serving concurrency and multi-GPU execution are
not validated by this record.
