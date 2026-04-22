/* ISPC kernels for paged attention algorithm on CPU */
#include "dtype.h"

#define K_TILE_WIDTH 2
export void qk_product(
  const uniform int cur_layer,
  const uniform int num_blocks,
  const uniform int seq_len,

  const uniform data_t q[], // [NUM_Q_HEADS, HEAD_DIM]
  const uniform data_t k_cache[], // [..., NUM_LAYERS, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM]
  const uniform int block_table[], // [seq_len]
  
  uniform itmd_t a[] // [seq_len, NUM_KV_HEADS, HEAD_DIM]
) {
  uniform int imax = seq_len / BLOCK_SIZE + 1;
  for (uniform int i = 0; i < imax; i++) {
    const uniform data_t * k = k_cache + 
      (1ll * cur_layer * num_blocks + block_table[i]) * BLOCK_NELEM;
    for (uniform int j = 0; j < NUM_KV_HEADS; j++) {
      uniform int q_off = j * QH_PER_KVH * HEAD_DIM;
      uniform int k_off = j * BLOCK_SIZE * HEAD_DIM;
      uniform int a_off = i * BLOCK_SIZE * NUM_Q_HEADS + j * QH_PER_KVH;
      uniform int tmax = min(BLOCK_SIZE, seq_len - i * BLOCK_SIZE);
      uniform int t;
      for (t = 0; t < tmax - K_TILE_WIDTH + 1; t += K_TILE_WIDTH) {
        itmd_t sum[QH_PER_KVH][K_TILE_WIDTH];
        for (uniform int h = 0; h < QH_PER_KVH; h++) {
          for (uniform int g = 0; g < K_TILE_WIDTH; g++) {
            sum[h][g] = 0;
          }
        }
        foreach (l = 0 ... HEAD_DIM) {
          for (uniform int g = 0; g < K_TILE_WIDTH; g++) {
            itmd_t k_val = k[k_off + g * HEAD_DIM + l];
            for (uniform int h = 0; h < QH_PER_KVH; h++) {
              sum[h][g] += q[q_off + h * HEAD_DIM + l] * k_val;
            }
          }
        }
        for (uniform int h = 0; h < QH_PER_KVH; h++) {
          for (uniform int g = 0; g < K_TILE_WIDTH; g++) {
            a[a_off + (t + g) * NUM_Q_HEADS + h] = reduce_add(sum[h][g]);
          }
        }
        k_off += HEAD_DIM * K_TILE_WIDTH;
      }
      for (; t < tmax; t++) {
        itmd_t sum[QH_PER_KVH];
        for (uniform int h = 0; h < QH_PER_KVH; h++) {
          sum[h] = 0;
        }
        foreach (l = 0 ... HEAD_DIM) {
          itmd_t k_val = k[k_off + l];
          for (uniform int h = 0; h < QH_PER_KVH; h++) {
            sum[h] += q[q_off + h * HEAD_DIM + l] * k_val;
          }
        }
        for (uniform int h = 0; h < QH_PER_KVH; h++) {
          a[a_off + t * NUM_Q_HEADS + h] = reduce_add(sum[h]);
        }
        k_off += HEAD_DIM;
      }
    }
  }
}

export void av_product(
  const uniform int cur_layer,
  const uniform int num_blocks,
  const uniform int seq_len,

  const uniform itmd_t a[], // [seq_len, NUM_KV_HEADS, HEAD_DIM]
  const uniform data_t v_cache[], // [..., NUM_LAYERS, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM]
  const uniform int block_table[], // [seq_len]
  
  uniform otpt_t o[] // [NUM_Q_HEADS, HEAD_DIM]
) {
  uniform int imax = seq_len / BLOCK_SIZE + 1;

  memset(o, 0, NUM_Q_HEADS * HEAD_DIM * sizeof(uniform otpt_t));
  for (uniform int i = 0; i < imax; i++) {
    const uniform data_t * uniform v = v_cache + 
      (1ll * cur_layer * num_blocks + block_table[i]) * BLOCK_NELEM;
    for (uniform int j = 0; j < NUM_KV_HEADS; j++) {
      uniform int o_off = j * QH_PER_KVH * HEAD_DIM;
      uniform int v_off = j * BLOCK_SIZE * HEAD_DIM;
      uniform int tmax = min(BLOCK_SIZE, seq_len - i * BLOCK_SIZE);
      for (uniform int t = 0; t < tmax; t++) {
        uniform int a_off = (i * BLOCK_SIZE + t) * NUM_Q_HEADS + j * QH_PER_KVH;
        foreach (l = 0 ... HEAD_DIM) {
          otpt_t v_val = v[v_off + l];
          for (uniform int h = 0; h < QH_PER_KVH; h++) {
            o[o_off + h * HEAD_DIM + l] += v_val * a[a_off + h];
          }
        }
        v_off += HEAD_DIM;
      }
    }
  }
}

void softmax(
  const uniform int seq_len,
  const uniform itmd_t softmax_scale,

  uniform itmd_t a[], // [seq_len, NUM_Q_HEADS]
  uniform itmd_t asb[] // [NUM_Q_HEADS]
) {
  uniform itmd_t amb[NUM_Q_HEADS];

  foreach (h = 0 ... NUM_Q_HEADS) {
    amb[h] = -1e20;
    for (uniform int i = 0; i < seq_len; i++) {
      a[i * NUM_Q_HEADS + h] *= softmax_scale;
      amb[h] = max(amb[h], a[i * NUM_Q_HEADS + h]);
    }
  }

  foreach (h = 0 ... NUM_Q_HEADS) {
    asb[h] = 0;
    for (uniform int i = 0; i < seq_len; i++) {
      a[i * NUM_Q_HEADS + h] = exp(a[i * NUM_Q_HEADS + h] - amb[h]);
      asb[h] += a[i * NUM_Q_HEADS + h];
    }
  }

  foreach (h = 0 ... NUM_Q_HEADS) {
    for (uniform int i = 0; i < seq_len; i++) {
      a[i * NUM_Q_HEADS + h] /= asb[h];
    }
    asb[h] = log(asb[h]) + amb[h];
  }
}

export void attn_one_seq(
  const uniform int cur_layer,
  const uniform int num_blocks,
  const uniform int seq_len,
  const uniform itmd_t softmax_scale,

  const uniform data_t q[], // [NUM_Q_HEADS, HEAD_DIM]
  const uniform data_t k_cache[], // [..., num_blocks, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM]
  const uniform data_t v_cache[], // [..., num_blocks, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM]
  const uniform int block_table[], // [seq_len]

  uniform itmd_t a[], // [seq_len, NUM_KV_HEADS, HEAD_DIM]
  uniform otpt_t o[], // [NUM_Q_HEADS, HEAD_DIM]
  uniform itmd_t asb[] // [NUM_Q_HEADS]
) {
  qk_product(cur_layer, num_blocks, seq_len, q, k_cache, block_table, a);
  softmax(seq_len, softmax_scale, a, asb);
  av_product(cur_layer, num_blocks, seq_len, a, v_cache, block_table, o);
}

// ---------------------------------------------------------------------------
// INT8 variants (M6: --int8-cpu-kv)
// ---------------------------------------------------------------------------
// These mirror qk_product / av_product / attn_one_seq above, but read the
// K / V cache as INT8 and dequantize on the fly using a parallel FP16 scale
// tensor. Scale layout (per-token granularity, the default):
//     k_scale / v_scale : [..., NUM_LAYERS, num_blocks, NUM_KV_HEADS, BLOCK_SIZE, 1]
// One FP16 scalar per (layer, block, kv_head, token) — broadcast over head_dim.
//
// Design rationale and constants come from docs/int8-design.md:
//   - Symmetric INT8  : x ≈ q * s
//   - Per-token       : one scale per token within a block
//   - Dequant location: fused here, inside the ISPC inner loop
//
// Layout of SCALE_NELEM mirrors BLOCK_NELEM but with head_dim folded to 1:
//   SCALE_NELEM = NUM_KV_HEADS * BLOCK_SIZE  (defined in dtype.h)
// ---------------------------------------------------------------------------
export void qk_product_int8(
  const uniform int cur_layer,
  const uniform int num_blocks,
  const uniform int seq_len,

  const uniform data_t q[],       // [NUM_Q_HEADS, HEAD_DIM]             fp16
  const uniform int8 k_cache[],   // INT8, shape mirrors fp16 k_cache
  const uniform data_t k_scale[], // [..., num_blocks, NUM_KV_HEADS, BLOCK_SIZE, 1]
  const uniform int block_table[],

  uniform itmd_t a[]              // [seq_len, NUM_KV_HEADS, HEAD_DIM]
) {
  uniform int imax = seq_len / BLOCK_SIZE + 1;
  for (uniform int i = 0; i < imax; i++) {
    uniform int64 blk_base = 1ll * cur_layer * num_blocks + block_table[i];
    const uniform int8   * uniform k  = k_cache + blk_base * BLOCK_NELEM;
    const uniform data_t * uniform ks = k_scale + blk_base * SCALE_NELEM;
    for (uniform int j = 0; j < NUM_KV_HEADS; j++) {
      uniform int q_off  = j * QH_PER_KVH * HEAD_DIM;
      uniform int k_off  = j * BLOCK_SIZE * HEAD_DIM;
      uniform int ks_off = j * BLOCK_SIZE;                 // one scale per token
      uniform int a_off  = i * BLOCK_SIZE * NUM_Q_HEADS + j * QH_PER_KVH;
      uniform int tmax   = min(BLOCK_SIZE, seq_len - i * BLOCK_SIZE);
      // Single-tile path — the K_TILE_WIDTH tile optimization is possible
      // but obscures the dequant step; leave it for a follow-up once the
      // fp32 dequant cost is measured.
      for (uniform int t = 0; t < tmax; t++) {
        uniform itmd_t scale = (itmd_t) ks[ks_off + t];
        itmd_t sum[QH_PER_KVH];
        for (uniform int h = 0; h < QH_PER_KVH; h++) { sum[h] = 0; }
        foreach (l = 0 ... HEAD_DIM) {
          itmd_t k_val = ((itmd_t) k[k_off + l]) * scale;
          for (uniform int h = 0; h < QH_PER_KVH; h++) {
            sum[h] += q[q_off + h * HEAD_DIM + l] * k_val;
          }
        }
        for (uniform int h = 0; h < QH_PER_KVH; h++) {
          a[a_off + t * NUM_Q_HEADS + h] = reduce_add(sum[h]);
        }
        k_off += HEAD_DIM;
      }
    }
  }
}

export void av_product_int8(
  const uniform int cur_layer,
  const uniform int num_blocks,
  const uniform int seq_len,

  const uniform itmd_t a[],
  const uniform int8 v_cache[],
  const uniform data_t v_scale[],
  const uniform int block_table[],

  uniform otpt_t o[]
) {
  uniform int imax = seq_len / BLOCK_SIZE + 1;

  memset(o, 0, NUM_Q_HEADS * HEAD_DIM * sizeof(uniform otpt_t));
  for (uniform int i = 0; i < imax; i++) {
    uniform int64 blk_base = 1ll * cur_layer * num_blocks + block_table[i];
    const uniform int8   * uniform v  = v_cache + blk_base * BLOCK_NELEM;
    const uniform data_t * uniform vs = v_scale + blk_base * SCALE_NELEM;
    for (uniform int j = 0; j < NUM_KV_HEADS; j++) {
      uniform int o_off  = j * QH_PER_KVH * HEAD_DIM;
      uniform int v_off  = j * BLOCK_SIZE * HEAD_DIM;
      uniform int vs_off = j * BLOCK_SIZE;
      uniform int tmax   = min(BLOCK_SIZE, seq_len - i * BLOCK_SIZE);
      for (uniform int t = 0; t < tmax; t++) {
        uniform int a_off = (i * BLOCK_SIZE + t) * NUM_Q_HEADS + j * QH_PER_KVH;
        uniform itmd_t scale = (itmd_t) vs[vs_off + t];
        foreach (l = 0 ... HEAD_DIM) {
          otpt_t v_val = ((otpt_t) v[v_off + l]) * ((otpt_t) scale);
          for (uniform int h = 0; h < QH_PER_KVH; h++) {
            o[o_off + h * HEAD_DIM + l] += v_val * a[a_off + h];
          }
        }
        v_off += HEAD_DIM;
      }
    }
  }
}

export void attn_one_seq_int8(
  const uniform int cur_layer,
  const uniform int num_blocks,
  const uniform int seq_len,
  const uniform itmd_t softmax_scale,

  const uniform data_t q[],
  const uniform int8 k_cache[],
  const uniform int8 v_cache[],
  const uniform data_t k_scale[],
  const uniform data_t v_scale[],
  const uniform int block_table[],

  uniform itmd_t a[],
  uniform otpt_t o[],
  uniform itmd_t asb[]
) {
  qk_product_int8(cur_layer, num_blocks, seq_len, q, k_cache, k_scale, block_table, a);
  softmax(seq_len, softmax_scale, a, asb);
  av_product_int8(cur_layer, num_blocks, seq_len, a, v_cache, v_scale, block_table, o);
}

export void gather_output_one_seq(
  const uniform int num_blocks,
  const uniform otpt_t o_buf[], // [num_blocks, NUM_Q_HEADS, HEAD_DIM]

  uniform itmd_t as_buf[], // [num_blocks, NUM_Q_HEADS]
  uniform otpt_t o[] // [NUM_Q_HEADS, HEAD_DIM]
) {
  uniform itmd_t as_all[NUM_Q_HEADS];
  uniform itmd_t am_all[NUM_Q_HEADS];

  foreach(h = 0 ... NUM_Q_HEADS) {
    am_all[h] = -1e20;
    for (uniform int i = 0; i < num_blocks; i++) {
      am_all[h] = max(am_all[h], as_buf[i * NUM_Q_HEADS + h]);
    }
  }

  foreach(h = 0 ... NUM_Q_HEADS) {
    as_all[h] = 0;
    for (uniform int i = 0; i < num_blocks; i++) {
      as_buf[i * NUM_Q_HEADS + h] = exp(as_buf[i * NUM_Q_HEADS + h] - am_all[h]);
      as_all[h] += as_buf[i * NUM_Q_HEADS + h];
    }
  }

  foreach(h = 0 ... NUM_Q_HEADS) {
    for (uniform int i = 0; i < num_blocks; i++) {
      as_buf[i * NUM_Q_HEADS + h] /= as_all[h];
    }
  }

  memset(o, 0, NUM_Q_HEADS * HEAD_DIM * sizeof(uniform otpt_t));
  for (uniform int i = 0; i < num_blocks; i++) {
    uniform int o_off = 0;
    uniform int o_buf_off = i * NUM_Q_HEADS * HEAD_DIM;
    for (uniform int j = 0; j < NUM_Q_HEADS; j++) {
      uniform itmd_t scale = as_buf[i * NUM_Q_HEADS + j];
      foreach (l = 0 ... HEAD_DIM) {
        o[o_off + l] += o_buf[o_buf_off + l] * scale;
      }
      o_off += HEAD_DIM;
      o_buf_off += HEAD_DIM;
    }
  }
}