// SPDX-License-Identifier: Apache-2.0
// Instantiations from kern/tools/flashinfer-sampling at the pinned runtime revision.
#include <flashinfer/sampling.cuh>

namespace fs = flashinfer::sampling;

constexpr uint32_t BLOCK = 1024;
constexpr uint32_t VEC = 4;

template __global__ void fs::OnlineSoftmaxFusedKernel<BLOCK, VEC, float, false>(float*, float*, float*, float,
                                                                               uint32_t);

template __global__ void fs::RadixTopKRenormProbKernel_MultiCTA<BLOCK, VEC, false, true, float, int>(
    float*, float*, int*, uint32_t, uint32_t, uint32_t, fs::RadixRowState*, uint32_t, uint32_t);

template __global__ void fs::TopPRenormProbKernel<BLOCK, fs::REDUCE_ALGO, VEC, float>(float*, float*, float*, float,
                                                                                     uint32_t);

template __global__ void fs::MinPSamplingFromProbKernel<BLOCK, fs::SCAN_ALGO, fs::REDUCE_ALGO, VEC, true, float, int>(
    float*, float*, int*, bool*, int*, float, uint32_t, uint64_t*, uint64_t, uint64_t*, uint64_t, uint32_t, uint32_t);
