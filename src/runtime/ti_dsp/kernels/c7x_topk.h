#ifndef TVM_C7X_TOPK_H_
#define TVM_C7X_TOPK_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int32_t c7x_topk(const void* data, void* out_val, void* out_idx, int32_t batch, int32_t n,
                  int32_t k);

#ifdef __cplusplus
}
#endif

#endif /* TVM_C7X_TOPK_H_ */
