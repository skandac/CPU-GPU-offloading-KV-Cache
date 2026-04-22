#include <math.h>
#include <string.h>
#include <omp.h>
#include <vector>
#include <algorithm>
#include "dtype.h"
#include "pacpu_ispc.h"

namespace brute {

void store_kv(
  int cur_layer,
  int num_blocks,
  int seq_len,
  data_t k[],
  data_t v[],
  data_t k_cache[],
  data_t v_cache[],
  int block_table[]
) {
  int block_pos = (seq_len - 1) / BLOCK_SIZE;
  int block_id = block_table[block_pos];
  int block_off = (seq_len - 1) % BLOCK_SIZE;
  int64_t cache_off = (1ll * cur_layer * num_blocks + block_id) * BLOCK_NELEM + block_off * HEAD_DIM;
  data_t* kp = k_cache + cache_off;
  data_t* vp = v_cache + cache_off;
  for (int i = 0; i < NUM_KV_HEADS; i++) {
    memcpy(kp + i * BLOCK_SIZE * HEAD_DIM, k + i * HEAD_DIM, HEAD_DIM * sizeof(data_t));
    memcpy(vp + i * BLOCK_SIZE * HEAD_DIM, v + i * HEAD_DIM, HEAD_DIM * sizeof(data_t));
  }
}

// M6: quantize the single new (k, v) token being appended to an INT8 CPU
// cache with a parallel FP16 per-token scale tensor. Matches the
// (num_kv_heads, block_size, head_dim) block layout; scale layout is
// (num_kv_heads, block_size, 1) — one FP16 per (head, token). See
// docs/int8-design.md §3–§4.
inline void store_kv_int8(
  int cur_layer,
  int num_blocks,
  int seq_len,
  data_t k[],        // fp16 [num_kv_heads, head_dim]
  data_t v[],        // fp16 [num_kv_heads, head_dim]
  int8_t k_cache[],  // int8 [num_layers, num_blocks, num_kv_heads, block_size, head_dim]
  int8_t v_cache[],  // int8 same
  data_t k_scale[],  // fp16 [num_layers, num_blocks, num_kv_heads, block_size]
  data_t v_scale[],  // fp16 same
  int block_table[]
) {
  int block_pos = (seq_len - 1) / BLOCK_SIZE;
  int block_id = block_table[block_pos];
  int block_off = (seq_len - 1) % BLOCK_SIZE;
  int64_t cache_off = (1ll * cur_layer * num_blocks + block_id) * BLOCK_NELEM + block_off * HEAD_DIM;
  int64_t scale_off = (1ll * cur_layer * num_blocks + block_id) * SCALE_NELEM + block_off;
  int8_t* kp = k_cache + cache_off;
  int8_t* vp = v_cache + cache_off;
  data_t* ksp = k_scale + scale_off;
  data_t* vsp = v_scale + scale_off;

  for (int h = 0; h < NUM_KV_HEADS; h++) {
    const data_t* kh = k + h * HEAD_DIM;
    const data_t* vh = v + h * HEAD_DIM;

    // amax per (head, token) -> symmetric int8 scale = amax / 127.
    itmd_t kmax = 0.0f, vmax = 0.0f;
    for (int d = 0; d < HEAD_DIM; d++) {
      itmd_t kv = (itmd_t) kh[d];
      itmd_t vv = (itmd_t) vh[d];
      if (kv < 0) kv = -kv;
      if (vv < 0) vv = -vv;
      if (kv > kmax) kmax = kv;
      if (vv > vmax) vmax = vv;
    }
    itmd_t ks_f = kmax / 127.0f;
    itmd_t vs_f = vmax / 127.0f;
    // Guard against all-zero tokens — avoid div-by-zero on quantize.
    itmd_t ks_inv = (ks_f > 0.0f) ? (1.0f / ks_f) : 0.0f;
    itmd_t vs_inv = (vs_f > 0.0f) ? (1.0f / vs_f) : 0.0f;

    ksp[h * BLOCK_SIZE] = (data_t) ks_f;
    vsp[h * BLOCK_SIZE] = (data_t) vs_f;

    int8_t* kpo = kp + h * BLOCK_SIZE * HEAD_DIM;
    int8_t* vpo = vp + h * BLOCK_SIZE * HEAD_DIM;
    for (int d = 0; d < HEAD_DIM; d++) {
      itmd_t kf = ((itmd_t) kh[d]) * ks_inv;
      itmd_t vf = ((itmd_t) vh[d]) * vs_inv;
      int kq = (int) (kf + (kf >= 0.0f ? 0.5f : -0.5f));
      int vq = (int) (vf + (vf >= 0.0f ? 0.5f : -0.5f));
      if (kq > 127) kq = 127; else if (kq < -127) kq = -127;
      if (vq > 127) vq = 127; else if (vq < -127) vq = -127;
      kpo[d] = (int8_t) kq;
      vpo[d] = (int8_t) vq;
    }
  }
}

void qk_product(
  int cur_layer,
  int num_blocks,
  int seq_len,

  data_t q[],
  data_t k_cache[],
  int block_table[],

  itmd_t a[]
) {
  for (auto j = 0; j < seq_len; j += BLOCK_SIZE) {
    auto kp = k_cache + (1ll * cur_layer * num_blocks + block_table[j / BLOCK_SIZE]) * BLOCK_NELEM;
    auto tlim = std::min(BLOCK_SIZE, seq_len - j);
    for (auto h = 0; h < NUM_KV_HEADS; h++) {
      for (auto t = 0; t < tlim; t++) {
        for (auto d = 0; d < HEAD_DIM; d++) {
          for (auto l = 0; l < QH_PER_KVH; l++) {
            a[(j + t) * NUM_Q_HEADS + h * QH_PER_KVH + l] += 
              q[(h * QH_PER_KVH + l) * HEAD_DIM + d] * kp[(h * BLOCK_SIZE + t) * HEAD_DIM + d];
          }
        }
      }
    }
  }
}

void av_product(
  int cur_layer,
  int num_blocks,
  int seq_len,

  itmd_t a[],
  data_t v_cache[],
  int block_table[],

  otpt_t o[]
) {
  memset(o, 0, NUM_Q_HEADS * HEAD_DIM * sizeof(otpt_t));
  for (auto j = 0; j < seq_len; j += BLOCK_SIZE) {
    auto vjp = v_cache + (1ll * cur_layer * num_blocks + block_table[j / BLOCK_SIZE]) * BLOCK_NELEM;
    auto vp = vjp;
    auto tlim = std::min(BLOCK_SIZE, seq_len - j);
    for (auto h = 0; h < NUM_KV_HEADS; h++) {
      for (auto t = 0; t < tlim; t++) {
        for (auto d = 0; d < HEAD_DIM; d++) {
          for (auto l = 0; l < QH_PER_KVH; l++) {
            o[(h * QH_PER_KVH + l) * HEAD_DIM + d] += 
              a[(j + t) * NUM_Q_HEADS + h * QH_PER_KVH + l] * vp[(h * BLOCK_SIZE + t) * HEAD_DIM + d];
          }
        }
      }
    }
  }
}

void softmax(
  int seq_len,
  itmd_t softmax_scale,
  itmd_t a[],
  itmd_t s[],
  itmd_t m[]
) {
  for (auto h = 0; h < NUM_Q_HEADS; h++) {
    s[h] = 0;
    m[h] = -1e20;
  }
  
  auto ap = a;
  for (auto j = 0; j < seq_len; j++) {
    for (auto h = 0; h < NUM_Q_HEADS; h++) {
      ap[h] *= softmax_scale;
      m[h] = std::max(m[h], ap[h]);
    }
    ap += NUM_Q_HEADS;
  }

  ap = a;
  for (auto j = 0; j < seq_len; j++) {
    for (auto h = 0; h < NUM_Q_HEADS; h++) {
      ap[h] = std::exp(ap[h] - m[h]);
      s[h] += ap[h];
    }
    ap += NUM_Q_HEADS;
  }

  ap = a;
  for (auto j = 0; j < seq_len; j++) {
    for (auto h = 0; h < NUM_Q_HEADS; h++) {
      ap[h] /= s[h];
    }
    ap += NUM_Q_HEADS;
  }
}

}

void brute_attention(
  int cur_layer,
  int num_blocks,
  int batch_size,
  int block_table_width,
  double softmax_scale,
  const std::vector<int64_t> &seq_ids,
	const std::vector<int64_t> &seq_lengths,
  
  data_t qbatch_p[],
  data_t kbatch_p[],
  data_t vbatch_p[],
  otpt_t obatch_p[],
  data_t kcache_p[],
  data_t vcache_p[],
  int block_table_p[]
){
  auto max_seq_len = *std::max_element(seq_lengths.begin(), seq_lengths.end());
  itmd_t* attn_score_buf = new itmd_t[max_seq_len * NUM_Q_HEADS];
  itmd_t* attn_sum_buf = new itmd_t[NUM_Q_HEADS];
  itmd_t* attn_max_buf = new itmd_t[NUM_Q_HEADS];

  for (auto i = 0; i < batch_size; i++) {
    int seq_id = seq_ids[i];
    int seq_len = seq_lengths[i];
    auto qip = qbatch_p + i * NUM_Q_HEADS * HEAD_DIM;
    auto kip = kbatch_p + i * NUM_KV_HEADS * HEAD_DIM;
    auto vip = vbatch_p + i * NUM_KV_HEADS * HEAD_DIM;
    auto oip = obatch_p + i * NUM_Q_HEADS * HEAD_DIM;
    auto btp = block_table_p + seq_id * block_table_width;
    memset(attn_score_buf, 0, seq_len * NUM_Q_HEADS * sizeof(itmd_t));

    brute::store_kv(cur_layer, num_blocks, seq_len, kip, vip, kcache_p, vcache_p, btp);
    brute::qk_product(cur_layer, num_blocks, seq_len, qip, kcache_p, btp, attn_score_buf);
    brute::softmax(seq_len, softmax_scale, attn_score_buf, attn_sum_buf, attn_max_buf);
    brute::av_product(cur_layer, num_blocks, seq_len, attn_score_buf, vcache_p, btp, oip);
  }
  delete [] attn_score_buf;
  delete [] attn_sum_buf;
  delete [] attn_max_buf;
}

void ispc_attention(
  int cur_layer,
  int num_blocks,
  int batch_size,
  int block_table_width,
  double softmax_scale,
  const std::vector<int64_t> &seq_ids,
  const std::vector<int64_t> &seq_lengths,
  
  data_t qbatch_p[],
  data_t kbatch_p[],
  data_t vbatch_p[],
  otpt_t obatch_p[],
  data_t kcache_p[],
  data_t vcache_p[],
  int block_table_p[]
) {
  int max_seq_len = *std::max_element(seq_lengths.begin(), seq_lengths.end());
  itmd_t* attn_score_buf = new itmd_t[max_seq_len * NUM_Q_HEADS];
  itmd_t attn_sum_buf[NUM_Q_HEADS];

  for (auto i = 0; i < batch_size; i++) {
    int seq_id = seq_ids[i];
    int seq_len = seq_lengths[i];
    auto qip = qbatch_p + i * NUM_Q_HEADS * HEAD_DIM;
    auto kip = kbatch_p + i * NUM_KV_HEADS * HEAD_DIM;
    auto vip = vbatch_p + i * NUM_KV_HEADS * HEAD_DIM;
    auto oip = obatch_p + i * NUM_Q_HEADS * HEAD_DIM;
    auto btp = block_table_p + seq_id * block_table_width;

    brute::store_kv(
      cur_layer, num_blocks, seq_len, 
      kip, vip, kcache_p, vcache_p, btp
    );

    ispc::attn_one_seq(
      cur_layer, num_blocks, seq_len, softmax_scale,
      qip, kcache_p, vcache_p, btp, 
      attn_score_buf, oip, attn_sum_buf
    );
  }

  delete [] attn_score_buf;
}

// Here we use global buffers to store intermediate results
itmd_t attn_score_buf[MAX_TOK_NUM * NUM_Q_HEADS];
otpt_t o_buf [MAX_TASK_NUM * NUM_Q_HEADS * HEAD_DIM];
itmd_t attn_sum_buf[MAX_TASK_NUM * NUM_Q_HEADS];

void ispc_attention_tasks(
  int cur_layer,
  int num_blocks,
  int batch_size,
  int block_table_width,
  double softmax_scale,
  const std::vector<int64_t> &seq_ids,
  const std::vector<int64_t> &seq_lengths,
  
  data_t qbatch_p[],
  data_t kbatch_p[],
  data_t vbatch_p[],
  otpt_t obatch_p[],
  data_t kcache_p[],
  data_t vcache_p[],
  int block_table_p[]
) {
  int ws = omp_get_max_threads();
  int bch_blk_size = (batch_size - 1) / ws + 1;
  int tot_blks = 0;
  for (auto i = 0; i < batch_size; i++) {
    tot_blks += (seq_lengths[i] - 1) / BLOCK_SIZE + 1;
  }

  int thrd_rst_blks[MAX_WS];
  for (auto i = 0; i < ws; i++) {
    thrd_rst_blks[i] = tot_blks / ws + (i < tot_blks % ws);
  }

  // Distribute tasks to threads, each thread processes no more than thrd_max_blks blocks
  std::vector<std::tuple<int, int, int, int> > tasks; // specs of each task (batch_id, seq_offs, seg_len, cum_seg_len)
  int* thrd_start_task = new int[ws + 1];             // starting task id for each thread
  int* seq_start_task = new int[batch_size + 1];      // starting task id for each sequence
  int cur_thrd = 0;
  int cum_seg_len = 0;
  thrd_start_task[0] = 0;
  for (int i = 0; i < batch_size; i++) {
    int seq_offs = 0;
    int seq_len = seq_lengths[i];
    seq_start_task[i] = tasks.size();
    while(seq_offs < seq_len) {
      if (thrd_rst_blks[cur_thrd] == 0) {
        thrd_start_task[++cur_thrd] = tasks.size();
      }
      int seg_len = std::min(seq_len - seq_offs, thrd_rst_blks[cur_thrd] * BLOCK_SIZE);
      tasks.emplace_back(i, seq_offs, seg_len, cum_seg_len);
      seq_offs += seg_len;
      cum_seg_len += seg_len;
      thrd_rst_blks[cur_thrd] -= (seg_len - 1) / BLOCK_SIZE + 1;
    }
  }
  seq_start_task[batch_size] = tasks.size();
  for (;cur_thrd < ws; cur_thrd++) {
    thrd_start_task[cur_thrd + 1] = tasks.size();
  }

  // for (int i = 0; i <= batch_size; i++) {
  //   printf("seq_start_task[%d] = %d\n", i, seq_start_task[i]);
  // }
  // for (int i = 0; i <= ws; i++) {
  //   printf("thrd_start_task[%d] = %d\n", i, thrd_start_task[i]);
  // }

  # pragma omp parallel
  {
    // Step 0:
    //   store the kv_cache
    int tid = omp_get_thread_num();
    int l = tid * bch_blk_size, r = std::min((tid + 1) * bch_blk_size, batch_size);
    // NOTE: l >= r when batch_size < omp_get_max_threads()
    for (auto i = l; i < r; i++) {
      int seq_id = seq_ids[i];
      int seq_len = seq_lengths[i];
      auto kip = kbatch_p + i * NUM_KV_HEADS * HEAD_DIM;
      auto vip = vbatch_p + i * NUM_KV_HEADS * HEAD_DIM;
      auto btp = block_table_p + seq_id * block_table_width;
      
      brute::store_kv(
        cur_layer, num_blocks, seq_len, kip, vip, kcache_p, vcache_p, btp
      );
    }
    # pragma omp barrier

    // Step 1: 
    //   compute intermediate output for each sequence block
    //   output is stored in o_buf and attn_sum_buf
    for (auto t = thrd_start_task[tid]; t < thrd_start_task[tid + 1]; t++) {
      int i, seq_offs, seg_len, cum_seg_len;
      std::tie(i, seq_offs, seg_len, cum_seg_len) = tasks[t];
      auto qip = qbatch_p + i * NUM_Q_HEADS * HEAD_DIM;
      auto oip = obatch_p + i * NUM_Q_HEADS * HEAD_DIM;
      auto btp = block_table_p + seq_ids[i] * block_table_width;
      ispc::attn_one_seq(
        cur_layer, num_blocks, seg_len, softmax_scale,
        qip, kcache_p, vcache_p, btp + seq_offs / BLOCK_SIZE,
        attn_score_buf + cum_seg_len * NUM_Q_HEADS,
        seg_len == seq_lengths[i] ? oip : o_buf + t * NUM_Q_HEADS * HEAD_DIM,
        attn_sum_buf + t * NUM_Q_HEADS
      );
    }
    # pragma omp barrier

    // Step 2:
    // Gather intermediate output to final output
    for (auto i = l; i < r; i++) {
      int num_tasks = seq_start_task[i + 1] - seq_start_task[i];
      if (num_tasks > 1) {
        int o_off = seq_start_task[i] * NUM_Q_HEADS * HEAD_DIM;
        int as_off = seq_start_task[i] * NUM_Q_HEADS;
        auto oip = obatch_p + i * NUM_Q_HEADS * HEAD_DIM;
        ispc::gather_output_one_seq(
          num_tasks,
          o_buf + o_off,
          attn_sum_buf + as_off,
          oip
        );
      }
    }
  }
}

// M6: INT8 CPU KV cache variant. Mirrors ispc_attention_tasks above but
//   - kcache_p / vcache_p are INT8 blocks ([layer, block, head, token, dim])
//   - kscale_p / vscale_p are FP16 per-token scales ([layer, block, head, token])
// Quantize-on-append happens in brute::store_kv_int8; dequantize is fused
// into ispc::attn_one_seq_int8. See docs/int8-design.md §5.
void ispc_attention_tasks_int8(
  int cur_layer,
  int num_blocks,
  int batch_size,
  int block_table_width,
  double softmax_scale,
  const std::vector<int64_t> &seq_ids,
  const std::vector<int64_t> &seq_lengths,

  data_t qbatch_p[],
  data_t kbatch_p[],
  data_t vbatch_p[],
  otpt_t obatch_p[],
  int8_t kcache_p[],
  int8_t vcache_p[],
  data_t kscale_p[],
  data_t vscale_p[],
  int block_table_p[]
) {
  int ws = omp_get_max_threads();
  int bch_blk_size = (batch_size - 1) / ws + 1;
  int tot_blks = 0;
  for (auto i = 0; i < batch_size; i++) {
    tot_blks += (seq_lengths[i] - 1) / BLOCK_SIZE + 1;
  }

  int thrd_rst_blks[MAX_WS];
  for (auto i = 0; i < ws; i++) {
    thrd_rst_blks[i] = tot_blks / ws + (i < tot_blks % ws);
  }

  std::vector<std::tuple<int, int, int, int> > tasks;
  int* thrd_start_task = new int[ws + 1];
  int* seq_start_task = new int[batch_size + 1];
  int cur_thrd = 0;
  int cum_seg_len = 0;
  thrd_start_task[0] = 0;
  for (int i = 0; i < batch_size; i++) {
    int seq_offs = 0;
    int seq_len = seq_lengths[i];
    seq_start_task[i] = tasks.size();
    while(seq_offs < seq_len) {
      if (thrd_rst_blks[cur_thrd] == 0) {
        thrd_start_task[++cur_thrd] = tasks.size();
      }
      int seg_len = std::min(seq_len - seq_offs, thrd_rst_blks[cur_thrd] * BLOCK_SIZE);
      tasks.emplace_back(i, seq_offs, seg_len, cum_seg_len);
      seq_offs += seg_len;
      cum_seg_len += seg_len;
      thrd_rst_blks[cur_thrd] -= (seg_len - 1) / BLOCK_SIZE + 1;
    }
  }
  seq_start_task[batch_size] = tasks.size();
  for (;cur_thrd < ws; cur_thrd++) {
    thrd_start_task[cur_thrd + 1] = tasks.size();
  }

  # pragma omp parallel
  {
    // Step 0: quantize + store the new (k, v) token into the int8 cache.
    int tid = omp_get_thread_num();
    int l = tid * bch_blk_size, r = std::min((tid + 1) * bch_blk_size, batch_size);
    for (auto i = l; i < r; i++) {
      int seq_id = seq_ids[i];
      int seq_len = seq_lengths[i];
      auto kip = kbatch_p + i * NUM_KV_HEADS * HEAD_DIM;
      auto vip = vbatch_p + i * NUM_KV_HEADS * HEAD_DIM;
      auto btp = block_table_p + seq_id * block_table_width;

      brute::store_kv_int8(
        cur_layer, num_blocks, seq_len,
        kip, vip, kcache_p, vcache_p, kscale_p, vscale_p, btp
      );
    }
    # pragma omp barrier

    // Step 1: per-segment attention (fused dequant inside the ISPC kernel).
    for (auto t = thrd_start_task[tid]; t < thrd_start_task[tid + 1]; t++) {
      int i, seq_offs, seg_len, cum_seg_len;
      std::tie(i, seq_offs, seg_len, cum_seg_len) = tasks[t];
      auto qip = qbatch_p + i * NUM_Q_HEADS * HEAD_DIM;
      auto oip = obatch_p + i * NUM_Q_HEADS * HEAD_DIM;
      auto btp = block_table_p + seq_ids[i] * block_table_width;
      ispc::attn_one_seq_int8(
        cur_layer, num_blocks, seg_len, softmax_scale,
        qip, kcache_p, vcache_p, kscale_p, vscale_p,
        btp + seq_offs / BLOCK_SIZE,
        attn_score_buf + cum_seg_len * NUM_Q_HEADS,
        seg_len == seq_lengths[i] ? oip : o_buf + t * NUM_Q_HEADS * HEAD_DIM,
        attn_sum_buf + t * NUM_Q_HEADS
      );
    }
    # pragma omp barrier

    // Step 2: gather partial outputs (identical to the fp16 path).
    for (auto i = l; i < r; i++) {
      int num_tasks = seq_start_task[i + 1] - seq_start_task[i];
      if (num_tasks > 1) {
        int o_off = seq_start_task[i] * NUM_Q_HEADS * HEAD_DIM;
        int as_off = seq_start_task[i] * NUM_Q_HEADS;
        auto oip = obatch_p + i * NUM_Q_HEADS * HEAD_DIM;
        ispc::gather_output_one_seq(
          num_tasks,
          o_buf + o_off,
          attn_sum_buf + as_off,
          oip
        );
      }
    }
  }

  delete [] thrd_start_task;
  delete [] seq_start_task;
}
