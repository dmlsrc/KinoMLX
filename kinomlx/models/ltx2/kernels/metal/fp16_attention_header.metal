// FP16 MLX STEEL attention primitives shared by the dense and the Sol-Attn
// routed attention kernels, plus the conversion the Sol-Attn preparation pass
// uses.
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

namespace sol {

// Converts an input element to the kernel type. BF16 to FP16 saturates at
// FP16's largest finite value; inside FP16's normal range it is exact, because
// FP16 has more mantissa bits than BF16.
template <typename T, typename I>
METAL_FUNC T sol_convert(I x) {
  return static_cast<T>(x);
}

template <>
METAL_FUNC half sol_convert<half, bfloat>(bfloat x) {
  return static_cast<half>(metal::clamp(static_cast<float>(x), -65504.0f, 65504.0f));
}

// Copies a kRows x kCols tile from device rows (stride src_ld) into threadgroup
// memory at (row * kDstStrRow + col * kDstStrCol). Each thread moves one
// contiguous chunk of one row.
template <
    typename T,
    short kRows,
    short kCols,
    short kDstStrRow,
    short kDstStrCol,
    short kThreads>
struct BlockLoader {
  STEEL_CONST short vec_size = (kRows * kCols) / kThreads;
  STEEL_CONST short kThreadCols = kCols / vec_size;
  static_assert((kRows * kCols) == (kThreads * vec_size), "exact work partition");
  static_assert((kThreadCols * vec_size) == kCols, "row-aligned contiguous chunks");
  static_assert((kThreads / kThreadCols) == kRows, "one row stripe");

  const int tile_stride;
  const short row;
  const short col;
  threadgroup T* dst;
  const device T* src;

  METAL_FUNC BlockLoader(
      const device T* src_,
      const int src_ld_,
      threadgroup T* dst_,
      ushort simd_group_id,
      ushort simd_lane_id) thread
      : tile_stride(kRows * src_ld_),
        row(short((simd_group_id * 32 + simd_lane_id) / kThreadCols)),
        col(short(vec_size * ((simd_group_id * 32 + simd_lane_id) % kThreadCols))),
        dst(dst_ + row * kDstStrRow + col * kDstStrCol),
        src(src_ + row * src_ld_ + col) {}

  METAL_FUNC void load_unsafe() const thread {
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < vec_size; j++) {
      dst[j * kDstStrCol] = src[j];
    }
  }

  // Rows at or past `valid_rows` are filled with zeros and never read from
  // device memory.
  METAL_FUNC void load_rows(short valid_rows) const thread {
    const bool row_valid = row < valid_rows;
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < vec_size; j++) {
      dst[j * kDstStrCol] = row_valid ? src[j] : T(0);
    }
  }

  METAL_FUNC void next() thread {
    src += tile_stride;
  }
};

// Staging for kBK-key K tiles (16 or 32) of kD = 64 or 128 dimensions with four
// SIMD groups, stored transposed with row stride kLdk. Lane l of SIMD group s
// takes key l % kBK and the 4-element chunks at (kD / 4) s + 4 (l / kBK) +
// kStride i, so each store spreads over more threadgroup-memory banks than a
// row-stripe loader. fetch() reads the lane's device elements into registers
// and commit() stores them, so the reads can be issued before the barrier that
// guards the tile.
template <typename T, short kD, short kBK, short kLdk>
struct SpreadKLoader {
  static_assert(kD == 64 || kD == 128, "head dimension 64 or 128");
  static_assert(kBK == 16 || kBK == 32, "16- or 32-key tiles");
  STEEL_CONST short kDims = kD / 4;
  STEEL_CONST short kStride = 4 * (32 / kBK);
  STEEL_CONST short kChunks = kDims / kStride;
  const int tile_stride;
  const short key;
  const short d0;
  threadgroup T* dst;
  const device T* src;
  vec<T, 4> regs[kChunks];

  METAL_FUNC SpreadKLoader(
      const device T* src_,
      const int src_ld_,
      threadgroup T* dst_,
      ushort simd_group_id,
      ushort simd_lane_id) thread
      : tile_stride(kBK * src_ld_),
        key(short(simd_lane_id % kBK)),
        d0(short(kDims * simd_group_id + 4 * (simd_lane_id / kBK))),
        dst(dst_ + key + d0 * kLdk),
        src(src_ + key * src_ld_ + d0) {}

  METAL_FUNC void fetch() thread {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kChunks; i++) {
      regs[i] = *reinterpret_cast<const device vec<T, 4>*>(src + kStride * i);
    }
  }

  METAL_FUNC void commit() const thread {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kChunks; i++) {
      STEEL_PRAGMA_UNROLL
      for (short e = 0; e < 4; e++) {
        dst[(kStride * i + e) * kLdk] = regs[i][e];
      }
    }
  }

  // fetch() for a partial tile: keys at or past `valid_rows` are staged as
  // zeros and never read from device memory.
  METAL_FUNC void fetch_rows(short valid_rows) thread {
    if (key < valid_rows) {
      fetch();
    } else {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < kChunks; i++) {
        regs[i] = vec<T, 4>(0);
      }
    }
  }

  // Element-wise staging of a partial tile, as the Sol-Attn kernel's partial
  // blocks use it. With fetch_rows() and commit() in their place that kernel
  // compiled differently and its 30,576-token call ran 2.9% slower.
  METAL_FUNC void load_rows(short valid_rows) const thread {
    const bool row_valid = key < valid_rows;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kChunks; i++) {
      STEEL_PRAGMA_UNROLL
      for (short e = 0; e < 4; e++) {
        dst[(kStride * i + e) * kLdk] = row_valid ? src[kStride * i + e] : T(0);
      }
    }
  }

  METAL_FUNC void next() thread {
    src += tile_stride;
  }
};

// V tiles of kBK rows (16 or 32) by whole rows: SIMD group s stages the kBK / 4
// rows from (kBK / 4) s and lane l the kD / 32 elements at (kD / 32) l, so one
// SIMD group's stores cover one whole row.
template <typename T, short kD, short kBK, short kLdv>
struct RowVLoader {
  static_assert(kD == 64 || kD == 128, "head dimension 64 or 128");
  static_assert(kBK == 16 || kBK == 32, "16- or 32-key tiles");
  STEEL_CONST short kRows = kBK / 4;
  STEEL_CONST short kVec = kD / 32;
  const int tile_stride;
  const int src_ld;
  const short row0;
  const short col;
  threadgroup T* dst;
  const device T* src;
  vec<T, kVec> regs[kRows];

  METAL_FUNC RowVLoader(
      const device T* src_,
      const int src_ld_,
      threadgroup T* dst_,
      ushort simd_group_id,
      ushort simd_lane_id) thread
      : tile_stride(kBK * src_ld_),
        src_ld(src_ld_),
        row0(short(kRows * simd_group_id)),
        col(short(kVec * simd_lane_id)),
        dst(dst_ + row0 * kLdv + col),
        src(src_ + row0 * src_ld_ + col) {}

  METAL_FUNC void fetch() thread {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRows; i++) {
      regs[i] = *reinterpret_cast<const device vec<T, kVec>*>(src + i * src_ld);
    }
  }

  METAL_FUNC void commit() const thread {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRows; i++) {
      *reinterpret_cast<threadgroup vec<T, kVec>*>(dst + i * kLdv) = regs[i];
    }
  }

  // fetch() for a partial tile: rows at or past `valid_rows` are staged as
  // zeros and never read from device memory.
  METAL_FUNC void fetch_rows(short valid_rows) thread {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRows; i++) {
      if (row0 + i < valid_rows) {
        regs[i] = *reinterpret_cast<const device vec<T, kVec>*>(src + i * src_ld);
      } else {
        regs[i] = vec<T, kVec>(0);
      }
    }
  }

  // Element-wise staging of a partial tile; see SpreadKLoader::load_rows.
  METAL_FUNC void load_rows(short valid_rows) const thread {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRows; i++) {
      const bool row_valid = (row0 + i) < valid_rows;
      STEEL_PRAGMA_UNROLL
      for (short e = 0; e < kVec; e++) {
        dst[i * kLdv + e] = row_valid ? src[i * src_ld + e] : T(0);
      }
    }
  }

  METAL_FUNC void next() thread {
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

// One SIMD group's 8-row stripe of COLS 8x8 fragments. A thread holds two
// adjacent columns of one row of each fragment.
template <int COLS>
struct RowTile {
  STEEL_CONST int kFragCols = 8;

  mma_frag_t val_frags[COLS];

  METAL_FUNC constexpr void clear() thread {
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < COLS; ++j) {
      val_frags[j] = mma_frag_t(0);
    }
  }

  METAL_FUNC constexpr thread mma_frag_t& frag_at(const short j) thread {
    return val_frags[j];
  }

  METAL_FUNC void row_max(thread float* vals) const thread {
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

  METAL_FUNC void row_sum(thread float* vals) const thread {
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

  METAL_FUNC void scale_by(float scale) thread {
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < COLS; ++j) {
      val_frags[j][0] *= scale;
      val_frags[j][1] *= scale;
    }
  }

  METAL_FUNC void exp2_sub(thread float* vals) thread {
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < COLS; ++j) {
      val_frags[j][0] = fast::exp2(val_frags[j][0] - vals[0]);
      val_frags[j][1] = fast::exp2(val_frags[j][1] - vals[0]);
    }
  }

  METAL_FUNC void mul_by(thread float* vals) thread {
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < COLS; ++j) {
      val_frags[j][0] *= vals[0];
      val_frags[j][1] *= vals[0];
    }
  }

  METAL_FUNC void div_by(thread float* vals) thread {
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < COLS; ++j) {
      val_frags[j][0] /= vals[0];
      val_frags[j][1] /= vals[0];
    }
  }

  template <typename T>
  METAL_FUNC void load(const threadgroup T* src) thread {
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < COLS; ++j) {
      const short off = j * kFragCols;
      val_frags[j][0] = static_cast<float>(src[off]);
      val_frags[j][1] = static_cast<float>(src[off + 1]);
    }
  }

  template <typename T>
  METAL_FUNC void store(device T* dst) const thread {
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < COLS; ++j) {
      const short off = j * kFragCols;
      dst[off] = static_cast<T>(val_frags[j][0]);
      dst[off + 1] = static_cast<T>(val_frags[j][1]);
    }
  }
};

} // namespace sol

#pragma METAL internals : disable
