// SPDX-License-Identifier: Apache-2.0
// Only the ABI is adapted: the device routines are FlashInfer's, as in pegainfer.
#include "attention.cuh"

extern "C" __global__ __launch_bounds__(Traits::NUM_THREADS)
void prefill(const __grid_constant__ Prefill params) {
  extern __shared__ uint8_t shared[];
  auto& storage = reinterpret_cast<Traits::SharedStoragePaged&>(shared);
  BatchPrefillWithPagedKVCacheDevice<Traits>(params, storage);
}

#define DECODE(G) \
extern "C" __global__ void decode_g##G(const __grid_constant__ Decode params) { \
  extern __shared__ uint8_t shared[]; \
  constexpr int Z = (X * G > 128 ? X * G : 128) / (X * G); \
  BatchDecodeWithPagedKVCacheDevice<PosEncodingMode::kNone, 2, G == 1 ? 4 : 1, \
      VEC, X, G, Z, Variant>(params, shared); \
}
DECODE(1)
DECODE(2)
DECODE(4)
DECODE(8)

template __global__ void flashinfer::PersistentVariableLengthMergeStatesKernel<
    VEC, X, 128 / X, 4, B, B, int>(B*, float*, int*, B*, float*, uint32_t, uint32_t*, uint32_t);
