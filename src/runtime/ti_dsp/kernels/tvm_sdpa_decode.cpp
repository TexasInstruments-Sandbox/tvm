/*
 * Scaled Dot-Product Attention for decode (seq_q=1).
 *
 * Fuses GQA head expansion + Q×K^T + softmax + attn×V into a single
 * kernel that reads K/V cache in native layout without materializing
 * transposed intermediates.
 *
 * Generic across GQA transformer models (Llama, Mistral, SmolLM, etc.)
 */

#include <stdint.h>
#include <math.h>

#ifdef __C7000__

#include <c7x.h>

#define FLOAT_VEC 8  /* C7504: 256-bit = 8 × float32 */

static inline float hsum_f8(__float8 v)
{
    return __get_vector_element(v, 0u) + __get_vector_element(v, 1u)
         + __get_vector_element(v, 2u) + __get_vector_element(v, 3u)
         + __get_vector_element(v, 4u) + __get_vector_element(v, 5u)
         + __get_vector_element(v, 6u) + __get_vector_element(v, 7u);
}

extern "C"
int32_t tvm_sdpa_decode(
    const void* Q_ptr,
    const void* K_cache_ptr,
    const void* V_cache_ptr,
    const void* mask_ptr,
    void* output_ptr,
    int32_t num_q_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    int32_t max_cache_len)
{
    const float* Q = (const float*)Q_ptr;
    const float* K_cache = (const float*)K_cache_ptr;
    const float* V_cache = (const float*)V_cache_ptr;
    const float* mask = (const float*)mask_ptr;
    float* output = (float*)output_ptr;

    const int32_t heads_per_group = num_q_heads / num_kv_heads;
    const float scale = 1.0f / sqrtf((float)head_dim);
    const int32_t hd_vecs = head_dim / FLOAT_VEC;

    /* SE template: streams one KV head's cache [max_cache_len, head_dim].
     * Initialized once, reused for each SE0_OPEN with different addresses. */
    __SE_TEMPLATE_v1 se_kv = __gen_SE_TEMPLATE_v1();
    se_kv.ELETYPE = __SE_ELETYPE_32BIT;
    se_kv.VECLEN  = __SE_VECLEN_8ELEMS;
    se_kv.DIMFMT  = __SE_DIMFMT_2D;
    se_kv.ICNT0   = head_dim;
    se_kv.ICNT1   = max_cache_len;
    se_kv.DIM1    = head_dim;

    float scores[1024];

    for (int32_t qh = 0; qh < num_q_heads; qh++) {
        const int32_t kv_h = qh / heads_per_group;
        const float* q_vec = Q + qh * head_dim;
        const float* k_head = K_cache + (int64_t)kv_h * max_cache_len * head_dim;
        const float* v_head = V_cache + (int64_t)kv_h * max_cache_len * head_dim;

        /* Load Q into registers (head_dim/8 vectors, stays in register file) */
        __float8 q_regs[16];
        for (int32_t d = 0; d < hd_vecs; d++) {
            q_regs[d] = *(__float8*)(q_vec + d * FLOAT_VEC);
        }

        /* --- Q × K^T: stream K rows via SE, dot with Q registers --- */
        __SE0_OPEN((void*)k_head, se_kv);

        for (int32_t pos = 0; pos < max_cache_len; pos++) {
            __float8 acc = (__float8)(0.0f);
            for (int32_t d = 0; d < hd_vecs; d++) {
                __float8 kv = __SE0ADV(float8);
                acc = acc + q_regs[d] * kv;
            }
            scores[pos] = hsum_f8(acc) * scale + mask[pos];
        }

        __SE0_CLOSE();

        /* --- Softmax --- */
        float max_s = scores[0];
        for (int32_t pos = 1; pos < max_cache_len; pos++) {
            if (scores[pos] > max_s) max_s = scores[pos];
        }

        float sum_exp = 0.0f;
        for (int32_t pos = 0; pos < max_cache_len; pos++) {
            scores[pos] = expf(scores[pos] - max_s);
            sum_exp += scores[pos];
        }

        float inv_sum = 1.0f / sum_exp;
        for (int32_t pos = 0; pos < max_cache_len; pos++) {
            scores[pos] *= inv_sum;
        }

        /* --- Weighted sum: output = scores × V via SE --- */
        float* out_head = output + qh * head_dim;
        __float8 out_regs[16];
        for (int32_t d = 0; d < hd_vecs; d++) {
            out_regs[d] = (__float8)(0.0f);
        }

        __SE0_OPEN((void*)v_head, se_kv);

        for (int32_t pos = 0; pos < max_cache_len; pos++) {
            float w = scores[pos];
            if (w != 0.0f) {
                __float8 wv = (__float8)(w);
                for (int32_t d = 0; d < hd_vecs; d++) {
                    __float8 vv = __SE0ADV(float8);
                    out_regs[d] = out_regs[d] + wv * vv;
                }
            } else {
                for (int32_t d = 0; d < hd_vecs; d++) {
                    (void)__SE0ADV(float8);
                }
            }
        }

        __SE0_CLOSE();

        for (int32_t d = 0; d < hd_vecs; d++) {
            *(__float8*)(out_head + d * FLOAT_VEC) = out_regs[d];
        }
    }

    return 0;
}

#else /* Host emulation fallback */

extern "C"
int32_t tvm_sdpa_decode(
    const void* Q_ptr,
    const void* K_cache_ptr,
    const void* V_cache_ptr,
    const void* mask_ptr,
    void* output_ptr,
    int32_t num_q_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    int32_t max_cache_len)
{
    const float* Q = (const float*)Q_ptr;
    const float* K_cache = (const float*)K_cache_ptr;
    const float* V_cache = (const float*)V_cache_ptr;
    const float* mask = (const float*)mask_ptr;
    float* output = (float*)output_ptr;

    const int32_t heads_per_group = num_q_heads / num_kv_heads;
    const float scale = 1.0f / sqrtf((float)head_dim);

    for (int32_t qh = 0; qh < num_q_heads; qh++) {
        const int32_t kv_h = qh / heads_per_group;
        const float* q_vec = Q + qh * head_dim;
        const float* k_head = K_cache + kv_h * max_cache_len * head_dim;
        const float* v_head = V_cache + kv_h * max_cache_len * head_dim;

        float scores[4096];

        for (int32_t pos = 0; pos < max_cache_len; pos++) {
            const float* k_row = k_head + pos * head_dim;
            float dot = 0.0f;
            for (int32_t d = 0; d < head_dim; d++) {
                dot += q_vec[d] * k_row[d];
            }
            scores[pos] = dot * scale + mask[pos];
        }

        float max_s = scores[0];
        for (int32_t pos = 1; pos < max_cache_len; pos++) {
            if (scores[pos] > max_s) max_s = scores[pos];
        }

        float sum_exp = 0.0f;
        for (int32_t pos = 0; pos < max_cache_len; pos++) {
            scores[pos] = expf(scores[pos] - max_s);
            sum_exp += scores[pos];
        }

        float inv_sum = 1.0f / sum_exp;
        for (int32_t pos = 0; pos < max_cache_len; pos++) {
            scores[pos] *= inv_sum;
        }

        float* out_head = output + qh * head_dim;
        for (int32_t d = 0; d < head_dim; d++) {
            out_head[d] = 0.0f;
        }
        for (int32_t pos = 0; pos < max_cache_len; pos++) {
            float w = scores[pos];
            if (w == 0.0f) continue;
            const float* v_row = v_head + pos * head_dim;
            for (int32_t d = 0; d < head_dim; d++) {
                out_head[d] += w * v_row[d];
            }
        }
    }

    return 0;
}

#endif /* __C7000__ */
