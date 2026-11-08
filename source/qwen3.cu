// SPDX-License-Identifier: Apache-2.0
// Flat CUDA entry points for dense Qwen3. The cache is [slot, K/V, head, dim].
// Attention uses FP32 online softmax and no architecture-specific instructions.
#include <cuda_bf16.h>
#include <stdint.h>
using B = __nv_bfloat16;

__device__ float warp_sum(float x) {
  for (int d = 16; d; d >>= 1) x += __shfl_xor_sync(0xffffffff, x, d);
  return x;
}

// One block owns a row; heads are independent rows for Q/K normalization.
extern "C" __global__ void rms_norm(B* out, const B* in, const B* weight,
                                    int width, float eps) {
  __shared__ float sums[8];
  int t = threadIdx.x;
  int64_t base = int64_t(blockIdx.x) * width;
  float sum = 0;
  for (int i = t; i < width; i += 256) {
    float v = float(in[base + i]);
    sum += v * v;
  }
  sum = warp_sum(sum);
  if ((t & 31) == 0) sums[t / 32] = sum;
  __syncthreads();
  sum = warp_sum(t < 8 ? sums[t] : 0.f);
  if (t == 0) sums[0] = rsqrtf(sum / width + eps);
  __syncthreads();
  for (int i = t; i < width; i += 256) {
    // HF materializes the normalized BF16 value before applying the weight.
    B normalized = B(float(in[base + i]) * sums[0]);
    out[base + i] = B(float(normalized) * float(weight[i]));
  }
}

extern "C" __global__ void rope(B* q, B* k, const int64_t* pos,
                                int qheads, int kvheads, int dim, float theta) {
  int row = blockIdx.x, head = blockIdx.y;
  bool is_q = head < qheads;
  int heads = is_q ? qheads : kvheads;
  int local = is_q ? head : head - qheads;
  B* x = (is_q ? q : k) + (int64_t(row) * heads + local) * dim;
  for (int j = threadIdx.x; j < dim / 2; j += blockDim.x) {
    float angle = float(pos[row]) / powf(theta, float(2 * j) / dim);
    float c = float(B(cosf(angle))), s = float(B(sinf(angle)));
    float a = float(x[j]), b = float(x[j + dim / 2]);
    // Separate BF16 multiplies match transformers' apply_rotary_pos_emb.
    x[j] = B(float(B(a * c)) - float(B(b * s)));
    x[j + dim / 2] = B(float(B(b * c)) + float(B(a * s)));
  }
}

extern "C" __global__ void cache_write(const B* k, const B* v, B* cache,
                                       const int64_t* slots, int width) {
  int row = blockIdx.x;
  int64_t slot = slots[row];
  if (slot < 0) return;
  for (int i = threadIdx.x; i < width; i += blockDim.x) {
    cache[slot * (2 * width) + i] = k[int64_t(row) * width + i];
    cache[slot * (2 * width) + width + i] = v[int64_t(row) * width + i];
  }
}

// One warp per (query, head), with a causal bound derived from packed offsets.
// This deliberately small baseline trades throughput for broad SM coverage.
extern "C" __global__ void attention(B* out, const B* q, const B* cache,
    const int* pages, const int* lengths, const int* offsets,
    int groups, int page_width, int qheads, int kvheads, int dim) {
  int row = blockIdx.x, head = blockIdx.y, lane = threadIdx.x;
  int seq = 0;
  while (seq + 1 < groups && row >= offsets[seq + 1]) ++seq;
  int end = lengths[seq] - (offsets[seq + 1] - row - 1);
  int kh = head / (qheads / kvheads);
  float query[8], acc[8];
  for (int j = 0; j < dim / 32; ++j) {
    query[j] = float(q[(int64_t(row) * qheads + head) * dim + j * 32 + lane]);
    acc[j] = 0.f;
  }
  float maximum = -INFINITY, denominator = 0.f;
  for (int p = 0; p < end; ++p) {
    int page = pages[seq * page_width + p / 16];
    int64_t base = (int64_t(page) * 16 + p % 16) * (2 * kvheads * dim) + kh * dim;
    float dot = 0.f;
    for (int j = 0; j < dim / 32; ++j)
      dot += query[j] * float(cache[base + j * 32 + lane]);
    dot = warp_sum(dot) * rsqrtf(float(dim));
    float next = fmaxf(maximum, dot);
    float old = expf(maximum - next), probability = expf(dot - next);
    denominator = denominator * old + probability;
    for (int j = 0; j < dim / 32; ++j)
      acc[j] = acc[j] * old + probability * float(cache[base + kvheads * dim + j * 32 + lane]);
    maximum = next;
  }
  for (int j = 0; j < dim / 32; ++j)
    out[(int64_t(row) * qheads + head) * dim + j * 32 + lane] = B(acc[j] / denominator);
}

extern "C" __global__ void embedding(const int64_t* ids, const B* table,
                                     B* out, int hidden, int vocab) {
  int row = blockIdx.x;
  int64_t id = ids[row];
  if (id < 0 || id >= vocab) __trap();
  for (int j = threadIdx.x; j < hidden; j += blockDim.x)
    out[int64_t(row) * hidden + j] = table[id * hidden + j];
}

// Stable ties choose the smallest vocabulary index.
extern "C" __global__ void argmax(const B* logits, int64_t* out, int vocab) {
  __shared__ float values[256];
  __shared__ int indices[256];
  int t = threadIdx.x;
  float best = -INFINITY;
  int index = 0x7fffffff;
  for (int j = t; j < vocab; j += 256) {
    float value = float(logits[int64_t(blockIdx.x) * vocab + j]);
    if (value > best || (value == best && j < index)) { best = value; index = j; }
  }
  values[t] = best; indices[t] = index;
  __syncthreads();
  for (int stride = 128; stride; stride >>= 1) {
    if (t < stride) {
      float other = values[t + stride]; int idx = indices[t + stride];
      if (other > values[t] || (other == values[t] && idx < indices[t])) {
        values[t] = other; indices[t] = idx;
      }
    }
    __syncthreads();
  }
  if (t == 0) out[blockIdx.x] = indices[0];
}
