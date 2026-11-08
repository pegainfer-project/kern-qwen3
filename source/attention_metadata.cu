// SPDX-License-Identifier: Apache-2.0
// Host-derived ABI offsets and initialized bytes; no copied struct layout.
#include "attention.cuh"
#include <cstdio>
#include <cstddef>
#include <cstring>

template<class P> void emit(const char* kind, int group) {
  P p;
  // Clear padding too: the byte image is reproducible across invocations.
  memset(&p, 0, sizeof(p));
  p.paged_kv.page_size = uint_fastdiv(16);
  p.paged_kv.head_dim = D;
  p.window_left = -1;
  p.sm_scale = 1.f / sqrtf(float(D));
  if constexpr (std::is_same_v<P, Prefill>) p.group_size = uint_fastdiv(group);
  if constexpr (std::is_same_v<P, Decode>) p.partition_kv = true;
  printf("\"%s_g%d\":{\"size\":%zu,\"bytes\":[", kind, group, sizeof(p));
  for (size_t j=0;j<sizeof(p);j++) printf("%s%u", j?",":"", ((unsigned char*)&p)[j]);
  printf("],\"offsets\":{");
#define FIELD(f) printf("\"" #f "\":%zu,", offsetof(P, f))
  FIELD(q); FIELD(o); FIELD(lse); FIELD(num_qo_heads); FIELD(q_stride_n); FIELD(q_stride_h);
  FIELD(request_indices); FIELD(kv_tile_indices); FIELD(o_indptr); FIELD(kv_chunk_size_ptr);
  FIELD(block_valid_mask); FIELD(padded_batch_size);
  FIELD(paged_kv.num_heads); FIELD(paged_kv.batch_size); FIELD(paged_kv.stride_page);
  FIELD(paged_kv.stride_n); FIELD(paged_kv.stride_h); FIELD(paged_kv.k_data);
  FIELD(paged_kv.v_data); FIELD(paged_kv.indices); FIELD(paged_kv.indptr); FIELD(paged_kv.last_page_len);
  if constexpr (std::is_same_v<P, Prefill>) {
    FIELD(q_indptr); FIELD(qo_tile_indices); FIELD(max_total_num_rows); FIELD(total_num_rows);
  }
#undef FIELD
  printf("\"partition_kv\":%zu}}", offsetof(P, partition_kv));
}
int main() {
  printf("{\"prefill_shared\":%zu,\"prefill_block\":[32,%u,%u],\"abi\":{",
      sizeof(Traits::SharedStoragePaged), get_num_warps_q(64), get_num_warps_kv(64));
  for (int g : {1,2,4,8}) { if (g!=1) printf(","); emit<Prefill>("prefill",g); printf(","); emit<Decode>("decode",g); }
  printf("}}\n");
}
