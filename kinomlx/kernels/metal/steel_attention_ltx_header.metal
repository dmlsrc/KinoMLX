// BF16-only MLX STEEL subset for the LTX-2.3 no-mask attention path.
// Derived from Apple MLX STEEL attention sources.
// Copyright (c) 2024-25 Apple Inc.
// SPDX-License-Identifier: MIT
#define MLX_METAL_JIT 1
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

#define STEEL_CONST static constant constexpr const
#define STEEL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")

#pragma METAL internals : enable

namespace mlx {
namespace steel {

template <
    short kRows,
    short kCols,
    short kDstStrRow,
    short kDstStrCol,
    short kActiveThreads,
    bool kAllActive = false>
struct BF16BlockLoader {
  STEEL_CONST short vec_size = (kRows * kCols) / kActiveThreads;
  STEEL_CONST short kThreadCols = kCols / vec_size;
  static_assert(
      (kRows * kCols) == (kActiveThreads * vec_size),
      "STEEL loader expects exact work partition.");
  static_assert(
      (kThreadCols * vec_size) == kCols,
      "STEEL loader expects row-aligned contiguous chunks.");
  static_assert(
      (kActiveThreads / kThreadCols) == kRows,
      "STEEL loader expects one row stripe.");

  const int tile_stride;
  const short thread_idx;
  const bool active;
  const short row;
  const short col;
  threadgroup bfloat* dst;
  const device bfloat* src;

  METAL_FUNC BF16BlockLoader(
      const device bfloat* src_,
      const int src_ld_,
      threadgroup bfloat* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]])
      : tile_stride(kRows * src_ld_),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        active(kAllActive || thread_idx < kActiveThreads),
        row(kAllActive ? short(thread_idx / kThreadCols)
                       : (active ? short(thread_idx / kThreadCols) : short(0))),
        col(kAllActive ? short(vec_size * (thread_idx % kThreadCols))
                       : (active ? short(vec_size * (thread_idx % kThreadCols))
                                 : short(0))),
        dst(dst_ + row * kDstStrRow + col * kDstStrCol),
        src(src_ + row * src_ld_ + col) {}

  METAL_FUNC void load_unsafe() const {
    if constexpr (!kAllActive) {
      if (!active) {
        return;
      }
    }
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < vec_size; j++) {
      dst[j * kDstStrCol] = src[j];
    }
  }

  METAL_FUNC void load_safe(short2 src_tile_dim) const {
    if constexpr (!kAllActive) {
      if (!active) {
        return;
      }
    }
    src_tile_dim = src_tile_dim - short2(col, row);
    const bool row_valid = src_tile_dim.y > 0;

    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < vec_size; j++) {
      dst[j * kDstStrCol] =
          (row_valid && j < src_tile_dim.x) ? src[j] : bfloat(0);
    }
  }

  METAL_FUNC void next() {
    src += tile_stride;
  }
};

typedef metal::vec<float, 2> mma_frag_t;
typedef metal::simdgroup_matrix<float, 8, 8> mma_mat_t;

METAL_FUNC static void mma_fragment(
    thread mma_frag_t& D,
    thread mma_frag_t& A,
    thread mma_frag_t& B,
    thread mma_frag_t& C) {
  mma_mat_t D_mat;
  mma_mat_t A_mat;
  mma_mat_t B_mat;
  mma_mat_t C_mat;

  reinterpret_cast<thread mma_frag_t&>(A_mat.thread_elements()) = A;
  reinterpret_cast<thread mma_frag_t&>(B_mat.thread_elements()) = B;
  reinterpret_cast<thread mma_frag_t&>(C_mat.thread_elements()) = C;

  simdgroup_multiply_accumulate(D_mat, A_mat, B_mat, C_mat);
  D = reinterpret_cast<thread mma_frag_t&>(D_mat.thread_elements());
}

template <int COLS>
struct RowTile {
  STEEL_CONST int kFragCols = 8;

  mma_frag_t val_frags[COLS];

  METAL_FUNC constexpr void clear() {
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < COLS; ++j) {
      val_frags[j] = mma_frag_t(0);
    }
  }

  METAL_FUNC constexpr thread mma_frag_t& frag_at(const short j) {
    return val_frags[j];
  }

  METAL_FUNC void row_max(thread float* vals) const {
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < COLS; ++j) {
      float thr_reduce = metal::max(val_frags[j].x, val_frags[j].y);
      float qgr_reduce = simd_shuffle_xor(thr_reduce, ushort(1));
      qgr_reduce = metal::max(thr_reduce, qgr_reduce);
      float sgr_reduce = simd_shuffle_xor(qgr_reduce, ushort(8));
      sgr_reduce = metal::max(qgr_reduce, sgr_reduce);
      vals[0] = metal::max(vals[0], sgr_reduce);
    }
  }

  METAL_FUNC void row_max_allcols(thread float* vals) const {
    mma_frag_t vec_reduce = val_frags[0];
    STEEL_PRAGMA_UNROLL
    for (short j = 1; j < COLS; ++j) {
      vec_reduce = metal::max(vec_reduce, val_frags[j]);
    }
    float thr_reduce = metal::max(vec_reduce.x, vec_reduce.y);
    float qgr_reduce = simd_shuffle_xor(thr_reduce, ushort(1));
    qgr_reduce = metal::max(thr_reduce, qgr_reduce);
    float sgr_reduce = simd_shuffle_xor(qgr_reduce, ushort(8));
    sgr_reduce = metal::max(qgr_reduce, sgr_reduce);
    vals[0] = metal::max(vals[0], sgr_reduce);
  }

  METAL_FUNC void row_sum(thread float* vals) const {
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < COLS; ++j) {
      float thr_reduce = val_frags[j].x + val_frags[j].y;
      float qgr_reduce = simd_shuffle_xor(thr_reduce, ushort(1));
      qgr_reduce = thr_reduce + qgr_reduce;
      float sgr_reduce = simd_shuffle_xor(qgr_reduce, ushort(8));
      sgr_reduce = qgr_reduce + sgr_reduce;
      vals[0] += sgr_reduce;
    }
  }

  METAL_FUNC void row_sum_allcols(thread float* vals) const {
    mma_frag_t vec_reduce = val_frags[0];
    STEEL_PRAGMA_UNROLL
    for (short j = 1; j < COLS; ++j) {
      vec_reduce += val_frags[j];
    }
    float thr_reduce = vec_reduce.x + vec_reduce.y;
    float qgr_reduce = simd_shuffle_xor(thr_reduce, ushort(1));
    qgr_reduce = thr_reduce + qgr_reduce;
    float sgr_reduce = simd_shuffle_xor(qgr_reduce, ushort(8));
    vals[0] += qgr_reduce + sgr_reduce;
  }

  METAL_FUNC void scale_by(float scale) {
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < COLS; ++j) {
      val_frags[j][0] *= scale;
      val_frags[j][1] *= scale;
    }
  }

  METAL_FUNC void exp2_sub(thread float* vals) {
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < COLS; ++j) {
      val_frags[j][0] = fast::exp2(val_frags[j][0] - vals[0]);
      val_frags[j][1] = fast::exp2(val_frags[j][1] - vals[0]);
    }
  }

  METAL_FUNC void exp2_scaled_sub(float scale, thread float* vals) {
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < COLS; ++j) {
      val_frags[j][0] = fast::exp2(metal::fma(val_frags[j][0], scale, -vals[0]));
      val_frags[j][1] = fast::exp2(metal::fma(val_frags[j][1], scale, -vals[0]));
    }
  }

  METAL_FUNC void mul_by(thread float* vals) {
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < COLS; ++j) {
      val_frags[j][0] *= vals[0];
      val_frags[j][1] *= vals[0];
    }
  }

  METAL_FUNC void div_by(thread float* vals) {
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < COLS; ++j) {
      val_frags[j][0] /= vals[0];
      val_frags[j][1] /= vals[0];
    }
  }

  METAL_FUNC void load(const threadgroup bfloat* src) {
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < COLS; ++j) {
      const short off = j * kFragCols;
      val_frags[j][0] = static_cast<float>(src[off]);
      val_frags[j][1] = static_cast<float>(src[off + 1]);
    }
  }

  METAL_FUNC void store(device bfloat* dst) const {
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < COLS; ++j) {
      const short off = j * kFragCols;
      dst[off] = static_cast<bfloat>(val_frags[j][0]);
      dst[off + 1] = static_cast<bfloat>(val_frags[j][1]);
    }
  }

  METAL_FUNC void
  store_safe(device bfloat* dst, const short2 dst_tile_dims) const {
    STEEL_PRAGMA_UNROLL
    for (int j = 0; j < COLS; ++j) {
      const short off = j * kFragCols;
      if (off < dst_tile_dims.x) {
        dst[off] = static_cast<bfloat>(val_frags[j][0]);
      }
      if ((off + 1) < dst_tile_dims.x) {
        dst[off + 1] = static_cast<bfloat>(val_frags[j][1]);
      }
    }
  }
};

template <
    int SCORE_COLS,
    int OUT_COLS,
    bool ReduceAllCols,
    bool ScaleInExp,
    bool SkipUnitFactor>
METAL_FUNC static void apply_online_softmax_rescale(
    thread RowTile<SCORE_COLS>& scores,
    thread RowTile<OUT_COLS>& output,
    const float scale,
    const float neg_inf,
    thread float* max_score,
    thread float* sum_score) {
  float new_max[1] = {max_score[0]};

  if constexpr (ScaleInExp) {
    float tile_max[1] = {neg_inf};
    if constexpr (ReduceAllCols) {
      scores.row_max_allcols(tile_max);
    } else {
      scores.row_max(tile_max);
    }
    new_max[0] = metal::max(new_max[0], tile_max[0] * scale);
    scores.exp2_scaled_sub(scale, new_max);
  } else {
    if constexpr (ReduceAllCols) {
      scores.row_max_allcols(new_max);
    } else {
      scores.row_max(new_max);
    }
    scores.exp2_sub(new_max);
  }

  float factor[1];
  if constexpr (SkipUnitFactor) {
    factor[0] =
        (new_max[0] == max_score[0]) ? 1.0f : fast::exp2(max_score[0] - new_max[0]);
  } else {
    factor[0] = fast::exp2(max_score[0] - new_max[0]);
  }
  max_score[0] = new_max[0];

  float sum_score_tmp[1] = {0};
  if constexpr (ReduceAllCols) {
    scores.row_sum_allcols(sum_score_tmp);
  } else {
    scores.row_sum(sum_score_tmp);
  }

  sum_score[0] = sum_score[0] * factor[0] + sum_score_tmp[0];
  output.mul_by(factor);
}

} // namespace steel
} // namespace mlx

#pragma METAL internals : disable

using namespace mlx::steel;
