#ifndef TVM_SDPA_DECODE_H_
#define TVM_SDPA_DECODE_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int32_t tvm_sdpa_decode(
    const void* Q,
    const void* K_cache,
    const void* V_cache,
    const void* mask,
    void* output,
    int32_t num_q_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    int32_t max_cache_len);

#ifdef __cplusplus
}
#endif

#endif  /* TVM_SDPA_DECODE_H_ */
