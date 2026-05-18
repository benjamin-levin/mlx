// Copyright © 2026 (WS1 fused_qsdpa).
//
// Two-pass fused quantized SDPA primitive with threadgroup-shared K/V
// dequantization. Designed for decode (T_q == 1) with GQA (gqa_factor query
// heads share one kv head). Each threadgroup loads and dequantizes one
// K/V row into threadgroup memory ONCE per kpos, then all GQA query heads
// scan it cooperatively. This amortizes the dequant cost over gqa_factor —
// the structural advantage over the v3 metal_kernel prototype.
//
// Layout:
//   threadgroup_dims = (BD=32, GQA_FACTOR, T_q=1)
//   grid_dims        = (KV_H*BD, B*GQA_FACTOR, blocks*T_q)
//
// pass1 partials:
//   partial_o: (B*n_q_heads, T_q, blocks, D) -- T
//   partial_m, partial_s: (B*n_q_heads, T_q, blocks) -- float

#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;

// Function constants for pass1 (id 30 chosen to not collide with sdpa_vector's
// 20..26 range).
constant bool fqsdpa_do_causal [[function_constant(30)]];
// id 31: left-padding mask. When true, kernel reads left_padding[batch_idx]
// and treats kpos < left_padding[batch_idx] as invalid (score=-INFINITY).
// When false, the kernel ignores the left_padding buffer entirely (it's
// still bound to a zero buffer on the C++ side for buffer-index stability).
constant bool fqsdpa_has_left_padding [[function_constant(31)]];

// ---------------------------------------------------------------------------
// Pass 1: build per-block partial outputs + (max, sum_exp) LSE state.
// Template parameters bake the major shapes; T_kv, NUM_KV_HEADS, N_BLOCKS,
// scale stay runtime (per decode step) via constant buffers.
// ---------------------------------------------------------------------------
template <typename T, int D, int GROUP_SIZE, int BITS, int GQA_FACTOR>
[[kernel]] void fused_qsdpa_pass1(
    const device T* queries [[buffer(0)]],
    const device uint32_t* k_packed [[buffer(1)]],
    const device T* k_scales [[buffer(2)]],
    const device T* k_biases [[buffer(3)]],
    const device uint32_t* v_packed [[buffer(4)]],
    const device T* v_scales [[buffer(5)]],
    const device T* v_biases [[buffer(6)]],
    device T* partial_o [[buffer(7)]],
    device float* partial_m [[buffer(8)]],
    device float* partial_s [[buffer(9)]],
    const constant int& T_kv [[buffer(10)]],
    const constant int& NUM_KV_HEADS [[buffer(11)]],
    const constant int& N_BLOCKS [[buffer(12)]],
    const constant float& scale [[buffer(13)]],
    const device int* left_padding [[buffer(14)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tidtg [[thread_position_in_threadgroup]],
    uint3 tptg [[threads_per_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BD = 32;
  constexpr int V_DIM = D;
  constexpr int EL_PER_INT = 32 / BITS;
  constexpr uint MASK_BITS = (1u << BITS) - 1u;
  constexpr int QK_PER_THREAD = D / BD;
  constexpr int V_PER_THREAD = V_DIM / BD;
  constexpr int K_UINT_PER_THREAD = QK_PER_THREAD / EL_PER_INT;
  constexpr int V_UINT_PER_THREAD = V_PER_THREAD / EL_PER_INT;

  typedef float U;

  // ---- Grid / threadgroup decoding ----
  const uint kv_head_idx = tid.x;
  const uint batch_idx = tid.y;
  const uint block_idx = tid.z;
  const uint lane = tidtg.x; // [0, BD)
  const uint q_head_in_grp = tidtg.y; // [0, GQA_FACTOR)
  const uint q_seq_idx = tidtg.z; // [0, T_q) -- always 0 for now

  const uint num_q_heads = (uint)NUM_KV_HEADS * (uint)GQA_FACTOR;
  const uint q_head_idx = kv_head_idx * (uint)GQA_FACTOR + q_head_in_grp;
  const uint q_batch_head_idx = batch_idx * num_q_heads + q_head_idx;

  // Per-thread D-element offset into a single head_dim row.
  const uint thread_d_off = lane * (uint)QK_PER_THREAD;

  // K/V offsets in packed and scales arrays.
  constexpr uint k_per_seq_packed = (uint)(D / EL_PER_INT);
  constexpr uint k_per_seq_scales = (uint)(D / GROUP_SIZE);
  const uint kv_batch_head_idx = batch_idx * (uint)NUM_KV_HEADS + kv_head_idx;
  const uint kv_off_packed = kv_batch_head_idx * (uint)T_kv * k_per_seq_packed;
  const uint kv_off_scales = kv_batch_head_idx * (uint)T_kv * k_per_seq_scales;

  // ---- Read Q row once per query head ----
  // queries layout: (B, n_q_heads, T_q, D).
  thread U q[QK_PER_THREAD];
  {
    const device T* q_ptr = queries +
        (q_batch_head_idx * (uint)1 + q_seq_idx) * (uint)D + thread_d_off;
    for (int i = 0; i < QK_PER_THREAD; i++) {
      q[i] = (U)scale * static_cast<U>(q_ptr[i]);
    }
  }
  U sum_q = 0;
  for (int i = 0; i < QK_PER_THREAD; i++) {
    sum_q += q[i];
  }

  // Online-softmax accumulators (float for numerical stability).
  thread U o[V_PER_THREAD];
  for (int i = 0; i < V_PER_THREAD; i++) {
    o[i] = 0;
  }
  U max_score = -INFINITY;
  U sum_exp_score = 0;

  // Multi-row tiled K/V dequant: each outer iteration processes TILE_KPOS K
  // positions in parallel. K rows are written into TG memory by each
  // simdgroup independently (one row per simdgroup), then all simdgroups
  // scan all rows. This amortizes dequant + barrier cost over TILE_KPOS
  // kpos at once. Requires TILE_KPOS <= GQA_FACTOR (so each row's dequant
  // can be claimed by a unique simdgroup) — for Qwen3.6 GQA_FACTOR=8 we
  // set TILE_KPOS = GQA_FACTOR = 8.
  constexpr uint TILE_KPOS = GQA_FACTOR;
  threadgroup U tg_k[TILE_KPOS * D];
  threadgroup U tg_v[TILE_KPOS * V_DIM];

  const uint k_group_idx = thread_d_off / (uint)GROUP_SIZE;

  // Strided partitioning: each threadgroup processes kpos = block_idx,
  // block_idx + N_BLOCKS, ... up to T_kv. Each outer iter handles TILE_KPOS
  // kpos at simdgroup-strided offsets, so all 8 simdgroups can dequant in
  // parallel. Strided pattern matches MLX's reference 2-pass layout and
  // benches a touch faster than contiguous-chunk on M4 Max for our shapes
  // (the strided reads still hit L2 well, and the workload distribution
  // across SMs is more even at irregular T_kv / N_BLOCKS ratios).
  const uint inner_stride = (uint)N_BLOCKS;
  const uint outer_stride = (uint)N_BLOCKS * TILE_KPOS;

  for (uint tile_base = block_idx; tile_base < (uint)T_kv;
       tile_base += outer_stride) {
    // ---- Cooperative dequant: each simdgroup dequants one K row + one V row
    // ----
    {
      const uint kpos_for_dequant = tile_base + q_head_in_grp * inner_stride;
      if (kpos_for_dequant < (uint)T_kv) {
        // K
        {
          const uint k_packed_base = kv_off_packed +
              kpos_for_dequant * k_per_seq_packed +
              thread_d_off / (uint)EL_PER_INT;
          const uint k_scales_base =
              kv_off_scales + kpos_for_dequant * k_per_seq_scales + k_group_idx;
          U k_scale = static_cast<U>(k_scales[k_scales_base]);
          U k_bias = static_cast<U>(k_biases[k_scales_base]);
          int idx = 0;
          for (int u = 0; u < K_UINT_PER_THREAD; u++) {
            uint packed = k_packed[k_packed_base + u];
            for (int e = 0; e < EL_PER_INT; e++) {
              uint raw = (packed >> (e * BITS)) & MASK_BITS;
              tg_k[q_head_in_grp * (uint)D + thread_d_off + idx] =
                  k_scale * static_cast<U>(raw) + k_bias;
              idx++;
            }
          }
        }
        // V
        {
          const uint v_packed_base = kv_off_packed +
              kpos_for_dequant * k_per_seq_packed +
              thread_d_off / (uint)EL_PER_INT;
          const uint v_scales_base =
              kv_off_scales + kpos_for_dequant * k_per_seq_scales + k_group_idx;
          U v_scale = static_cast<U>(v_scales[v_scales_base]);
          U v_bias = static_cast<U>(v_biases[v_scales_base]);
          int idx = 0;
          for (int u = 0; u < V_UINT_PER_THREAD; u++) {
            uint packed = v_packed[v_packed_base + u];
            for (int e = 0; e < EL_PER_INT; e++) {
              uint raw = (packed >> (e * BITS)) & MASK_BITS;
              tg_v[q_head_in_grp * (uint)V_DIM + thread_d_off + idx] =
                  v_scale * static_cast<U>(raw) + v_bias;
              idx++;
            }
          }
        }
      } else {
        // Mark this row as invalid by writing -INFINITY into the K tile
        // for the score (we'll mask its contribution below). We zero the
        // V row defensively.
        for (int i = 0; i < QK_PER_THREAD; i++) {
          tg_k[q_head_in_grp * (uint)D + thread_d_off + i] = (U)0;
        }
        for (int i = 0; i < V_PER_THREAD; i++) {
          tg_v[q_head_in_grp * (uint)V_DIM + thread_d_off + i] = (U)0;
        }
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Now process TILE_KPOS rows: each q-head scans all rows ----
    // Online softmax update across all TILE_KPOS scores at once.
    //
    // Left-padding: when fqsdpa_has_left_padding is set, the per-sequence
    // valid region starts at left_padding[batch_idx]. K positions before
    // that are masked out (score = -INFINITY). When unset, the compiler
    // eliminates both the buffer load and the per-row comparison entirely.
    int lp_thresh = 0;
    if (fqsdpa_has_left_padding) {
      lp_thresh = left_padding[batch_idx];
    }
    U scores[TILE_KPOS];
    bool valid[TILE_KPOS];
#pragma clang loop unroll(full)
    for (uint r = 0; r < TILE_KPOS; r++) {
      uint kpos_r = tile_base + r * inner_stride;
      bool in_bounds = (kpos_r < (uint)T_kv);
      bool past_padding =
          !fqsdpa_has_left_padding || ((int)kpos_r >= lp_thresh);
      valid[r] = in_bounds && past_padding;
      U raw_dot = 0;
      for (int i = 0; i < QK_PER_THREAD; i++) {
        raw_dot += q[i] * tg_k[r * (uint)D + thread_d_off + i];
      }
      U s = simd_sum(raw_dot);
      scores[r] = valid[r] ? s : (U)(-INFINITY);
    }

    // Find max across this tile + previous max.
    U new_max = max_score;
#pragma clang loop unroll(full)
    for (uint r = 0; r < TILE_KPOS; r++) {
      new_max = max(new_max, scores[r]);
    }
    // Guard against the all-invalid case. When both max_score and
    // new_max are -INFINITY (no valid K position has been seen across
    // any tile processed by this threadgroup so far — possible under
    // left_padding when the strided kpos pattern lands entirely inside
    // the padded prefix), ``max_score - new_max`` is NaN and would
    // poison o[] and sum_exp_score. The safe action is to skip the
    // online update for this tile entirely; o[] stays 0 and accumulates
    // from later valid tiles. The threadgroup barrier still fires
    // unconditionally to avoid divergent control flow across simdgroups.
    if (new_max > -INFINITY) {
      U factor = fast::exp(max_score - new_max);
      // Online update sum_exp_score and o[].
      sum_exp_score = sum_exp_score * factor;
#pragma clang loop unroll(full)
      for (int i = 0; i < V_PER_THREAD; i++) {
        o[i] = o[i] * factor;
      }

#pragma clang loop unroll(full)
      for (uint r = 0; r < TILE_KPOS; r++) {
        U exp_score = valid[r] ? fast::exp(scores[r] - new_max) : (U)0;
        sum_exp_score += exp_score;
#pragma clang loop unroll(full)
        for (int i = 0; i < V_PER_THREAD; i++) {
          o[i] += exp_score * tg_v[r * (uint)V_DIM + thread_d_off + i];
        }
      }
      max_score = new_max;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // ---- Write partials ----
  // partial_o layout: (B*n_q_heads, T_q, blocks, D)
  uint partial_offset =
      (q_batch_head_idx * (uint)1 + q_seq_idx) * (uint)N_BLOCKS + block_idx;
  device T* po_ptr = partial_o + partial_offset * (uint)D + thread_d_off;
  for (int i = 0; i < V_PER_THREAD; i++) {
    po_ptr[i] = static_cast<T>(o[i]);
  }
  if (lane == 0) {
    partial_m[partial_offset] = max_score;
    partial_s[partial_offset] = sum_exp_score;
  }
}

// ---------------------------------------------------------------------------
// Pass 2: reduce per-block partials into final output.
// Threadgroup is (1024, 1, 1) -> 32 simdgroups, BN=32.
// Each simdgroup g handles a contiguous group of D/32 lanes that strides
// through the N_BLOCKS partials. N_BLOCKS is a runtime constant (per launch)
// but assumed to be a multiple of 32.
// ---------------------------------------------------------------------------
template <typename T, int D>
[[kernel]] void fused_qsdpa_pass2(
    const device T* partial_o [[buffer(0)]],
    const device float* partial_m [[buffer(1)]],
    const device float* partial_s [[buffer(2)]],
    device T* out [[buffer(3)]],
    const constant int& N_BLOCKS [[buffer(4)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BD = 32;
  constexpr int BN = 32;
  constexpr int ELEM_PER_THREAD = D / BD;

  typedef float U;

  threadgroup U outputs_tg[BN * BD];

  const uint head_idx = tid.x;
  const uint q_seq_idx = tid.y;
  const uint q_offset = head_idx * tpg.y + q_seq_idx;

  const device U* maxs_base = partial_m + q_offset * (uint)N_BLOCKS;
  const device U* sums_base = partial_s + q_offset * (uint)N_BLOCKS;

  const int n_per_lane = N_BLOCKS / BN;

  // ---- Reduce max across all blocks ----
  U max_score = -INFINITY;
  for (int b = 0; b < n_per_lane; b++) {
    U m = maxs_base[simd_lid + BN * b];
    max_score = max(max_score, m);
  }
  max_score = simd_max(max_score);

  // ---- Reduce sum_exp ----
  U sum_exp = 0;
  for (int b = 0; b < n_per_lane; b++) {
    U f = fast::exp(maxs_base[simd_lid + BN * b] - max_score);
    sum_exp += f * sums_base[simd_lid + BN * b];
  }
  sum_exp = simd_sum(sum_exp);

  // ---- Weighted reduce of partial_o ----
  thread U o[ELEM_PER_THREAD];
  for (int i = 0; i < ELEM_PER_THREAD; i++) {
    o[i] = 0;
  }

  const device T* po_ptr = partial_o + q_offset * (uint)N_BLOCKS * (uint)D +
      simd_gid * (uint)D + simd_lid * (uint)ELEM_PER_THREAD;
  const device U* maxs_ptr = maxs_base + simd_gid;
  for (int b = 0; b < n_per_lane; b++) {
    U f = fast::exp(maxs_ptr[0] - max_score);
    for (int i = 0; i < ELEM_PER_THREAD; i++) {
      o[i] += f * static_cast<U>(po_ptr[i]);
    }
    po_ptr += BN * (uint)D;
    maxs_ptr += BN;
  }

  // ---- Transpose-reduce across simdgroups ----
  for (int i = 0; i < ELEM_PER_THREAD; i++) {
    outputs_tg[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs_tg[simd_gid * BD + simd_lid]);
    o[i] = (sum_exp == 0) ? o[i] : (o[i] / sum_exp);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (simd_lid == 0) {
    device T* out_ptr =
        out + q_offset * (uint)D + simd_gid * (uint)ELEM_PER_THREAD;
    for (int i = 0; i < ELEM_PER_THREAD; i++) {
      out_ptr[i] = static_cast<T>(o[i]);
    }
  }
}
