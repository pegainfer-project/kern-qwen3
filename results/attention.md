# FlashInfer attention switch

Measured on 2026-11-08, RTX 5070 Ti × 1 (16 GB, SM120), CUDA 13.3.
The baseline is the portable attention from repository revision `30fba56`.
The candidate uses pegainfer's pinned FlashInfer revision
`19f1a41e6b21f0c422d775e377b6fdf9a1fc9d23`, with GPU metadata preparation,
causal paged prefill, fixed 64-token split decode, and FlashInfer merge.
The runtime executable and artifact hashes are recorded in the reports.

## Measured A/B

Both bundles were measured on the same device with `bench/attention.toml`:
18 scenarios, 12 samples each. Values below are median complete-program CUDA
Graph GPU times in milliseconds, not HTTP TTFT or TPOT.

| Program | Sequences | Rows per sequence | Existing context | Before | After |
|---|---:|---:|---:|---:|---:|
| Decode | 1 | 1 | 128 | 15.498 | 11.478 |
| Decode | 1 | 1 | 512 | 28.080 | 11.554 |
| Decode | 1 | 1 | 2048 | 78.299 | 11.798 |
| Decode | 1 | 1 | 3584 | 128.537 | 12.078 |
| Decode | 32 | 1 | 512 | 28.111 | 15.187 |
| Prefill | 1 | 128 | 2048 | 203.160 | 15.694 |
| Prefill | 1 | 256 | 0 | 40.446 | 24.649 |
| Prefill | 1 | 256 | 2048 | 379.196 | 28.595 |

Raw reports: [before](attention-before-bench.json.gz),
[after](attention-after-bench.json.gz), [kern test A/B](attention-ab.json.gz).
Local artifact paths have been replaced with portable labels.

## Completed validation

- `kern test`: PASS, end-to-end KL at most 4.16e-4 over four decode rows;
  argmax agrees. Prefill length 513, chunk 128, capacity 1024, five timing iterations.
- [Transformers teacher forcing](attention-accuracy.json): four prompts,
  chunks 16/256, eight decode steps each; all mean KL values below 0.01.
- [Batched full-model check](attention-batch.json): 1/3/17/32 sequences with
  different prefixes and shuffled pages, plus sampled replay and top-k membership.
- The actual generated attention launches agree with FP32 PyTorch SDPA for
  dimensions 64/128/256 and GQA ratios 1/2/4/8, shuffled pages and causal prefixes
  up to 8192 tokens. Decode covers 1/3/17/32 sequences and page/split boundaries.
  CUDA Graph replay changes both the physical page mapping and active split count.
- All eight modules compile for SM80/86/87/88/89/90/100/103/110/120/121.
  Only SM120 has GPU execution evidence.

The 40960-context, 1024-chunk bundle passes manifest verification. Its full-model
long-context comparison was stopped at the user's request, so no completed 40K
numerical gate or long-context serving performance claim is made. Existing HTTP
sampling evidence predates this attention switch. No cross-engine benchmark was run.
