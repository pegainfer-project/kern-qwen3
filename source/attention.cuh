// SPDX-License-Identifier: Apache-2.0
// FlashInfer types and launch geometry used by pegainfer's paged_launch.cuh.
#pragma once
#include <flashinfer/attention/decode.cuh>
#include <flashinfer/attention/prefill.cuh>
#include <flashinfer/attention/default_decode_params.cuh>
#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/variants.cuh>
using namespace flashinfer;
using B = __nv_bfloat16;
using Decode = BatchDecodeParams<B, B, B, int>;
using Prefill = BatchPrefillPagedParams<B, B, B, int>;
using Variant = DefaultAttention<false, false, false, false>;
constexpr int D = KERN_HEAD_DIM;
constexpr int VEC = 8, X = D / VEC;
// A fixed 64-query tile and two KV MMA tiles fit SM80 through SM121.
using Traits = KernelTraits<MaskMode::kCausal, 64, get_num_mma_q(64), 2,
    D / 16, D / 16, get_num_warps_q(64), get_num_warps_kv(64),
    PosEncodingMode::kNone, B, B, B, float, int, Variant>;
static_assert(!Traits::IsInvalid());
