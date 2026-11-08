// SPDX-License-Identifier: Apache-2.0
// Host-only sizeof facts for the exact FlashInfer/CUB template instantiations.
#include <cstdio>
#include <flashinfer/sampling.cuh>
namespace fs = flashinfer::sampling;
int main() {
  printf("{\"softmax\":%zu,\"topp\":%zu,\"minp\":%zu,\"state_words\":%zu,\"radix_fixed\":%zu}\n",
    sizeof(fs::OnlineSoftmaxTempStorage<1024>),
    sizeof(fs::RenormTempStorage<1024, fs::REDUCE_ALGO>),
    sizeof(fs::SamplingTempStorage<1024, fs::SCAN_ALGO, fs::REDUCE_ALGO>),
    sizeof(fs::RadixRowState) / sizeof(unsigned),
    ((sizeof(unsigned) * (256 + 256 + 4) + sizeof(float) + 15) / 16) * 16);
}
