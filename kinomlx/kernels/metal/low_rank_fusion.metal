// FP32 block products on 8x8 SIMD-group fragments, and loaders that widen
// adapter factors to FP32 while staging them in threadgroup memory, for the
// low-rank fusion kernel.
// The fragment layout, the block product and the loaders' index scheme are
// derived from Apple MLX's STEEL GEMM sources (steel/gemm/mma.h, loader.h).
// Copyright (c) 2024 Apple Inc.
// SPDX-License-Identifier: MIT
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

#define LRF_CONST static constant constexpr const
#define LRF_UNROLL _Pragma("clang loop unroll(full)")

namespace lrf {

// One 8x8 FP32 fragment of a SIMD group: each thread holds two adjacent
// elements of one row.
struct Frag {
  typedef simdgroup_matrix<float, 8, 8> mat_type;
  typedef float2 frag_type;

  // The thread's (column, row) in a fragment.
  METAL_FUNC static constexpr short2 coord(ushort lane) {
    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;
    return short2{fn, fm};
  }

  // The thread's two elements, the second SY elements after the first.
  template <short SY, typename P>
  METAL_FUNC static constexpr void load(thread frag_type& dst, P src) {
    dst[0] = static_cast<float>(src[0]);
    dst[1] = static_cast<float>(src[SY]);
  }

  METAL_FUNC static void mma(
      thread frag_type& d,
      thread frag_type& a,
      thread frag_type& b,
      thread frag_type& c) {
    mat_type dm;
    mat_type am;
    mat_type bm;
    mat_type cm;
    reinterpret_cast<thread frag_type&>(am.thread_elements()) = a;
    reinterpret_cast<thread frag_type&>(bm.thread_elements()) = b;
    reinterpret_cast<thread frag_type&>(cm.thread_elements()) = c;
    simdgroup_multiply_accumulate(dm, am, bm, cm);
    d = reinterpret_cast<thread frag_type&>(dm.thread_elements());
  }
};

// TR x TC fragments, row-major.
template <short TR, short TC>
struct Tile {
  Frag::frag_type frags[TR * TC] = {Frag::frag_type(0)};

  METAL_FUNC thread Frag::frag_type& at(short i, short j) thread {
    return frags[i * TC + j];
  }

  // Fragment (i, j) starts (8 i WX) rows of stride SX and (8 j WY) columns of
  // stride SY into src.
  template <short WX, short WY, short SX, short SY>
  METAL_FUNC void load(const threadgroup float* src) thread {
    LRF_UNROLL
    for (short i = 0; i < TR; ++i) {
      LRF_UNROLL
      for (short j = 0; j < TC; ++j) {
        Frag::load<SY>(at(i, j), &src[(i * 8) * WX * SX + (j * 8) * WY * SY]);
      }
    }
  }
};

template <short M, short N, short K>
METAL_FUNC void tile_mma(
    thread Tile<M, N>& d,
    thread Tile<M, K>& a,
    thread Tile<K, N>& b,
    thread Tile<M, N>& c) {
  LRF_UNROLL
  for (short m = 0; m < M; ++m) {
    LRF_UNROLL
    for (short n = 0; n < N; ++n) {
      const short n_serp = (m % 2) ? (N - 1 - n) : n;
      LRF_UNROLL
      for (short k = 0; k < K; ++k) {
        Frag::mma(d.at(m, n_serp), a.at(m, k), b.at(k, n_serp), c.at(m, n_serp));
      }
    }
  }
}

// A SIMD group's share of a BM x BN FP32 product of threadgroup tiles: A is
// BM x BK (stored K-major with TA) with row stride LDA, B is BK x BN (stored
// N-major with TB) with row stride LDB. The SIMD groups form a WM x WN grid
// and each owns TM x TN fragments interleaved across it. Each call adds BK
// to the inner dimension in 8-wide steps.
template <short BM, short BN, short BK, short WM, short WN, bool TA, bool TB, short LDA, short LDB>
struct BlockMMA {
  LRF_CONST short TM = BM / (8 * WM);
  LRF_CONST short TN = BN / (8 * WN);
  LRF_CONST short A_str_m = TA ? 1 : LDA;
  LRF_CONST short A_str_k = TA ? LDA : 1;
  LRF_CONST short B_str_k = TB ? 1 : LDB;
  LRF_CONST short B_str_n = TB ? LDB : 1;
  LRF_CONST short tile_stride_a = 8 * A_str_k;
  LRF_CONST short tile_stride_b = 8 * B_str_k;

  Tile<TM, 1> Atile;
  Tile<1, TN> Btile;
  Tile<TM, TN> Ctile;

  // The thread's first element in the tile, and its fragment offsets.
  short sm;
  short sn;
  short As_offset;
  short Bs_offset;

  METAL_FUNC BlockMMA(ushort simd_group_id, ushort simd_lane_id) thread {
    const short tm = 8 * (simd_group_id / WN);
    const short tn = 8 * (simd_group_id % WN);
    const short2 coord = Frag::coord(simd_lane_id);
    sm = coord.y;
    sn = coord.x;
    As_offset = (tm + sm) * A_str_m + sn * A_str_k;
    Bs_offset = sm * B_str_k + (tn + sn) * B_str_n;
    sm += tm;
    sn += tn;
  }

  METAL_FUNC void mma(const threadgroup float* As, const threadgroup float* Bs) thread {
    As += As_offset;
    Bs += Bs_offset;
    LRF_UNROLL
    for (short kk = 0; kk < BK; kk += 8) {
      simdgroup_barrier(mem_flags::mem_none);
      Atile.template load<WM, 1, A_str_m, A_str_k>(As);
      simdgroup_barrier(mem_flags::mem_none);
      Btile.template load<1, WN, B_str_k, B_str_n>(Bs);
      simdgroup_barrier(mem_flags::mem_none);
      tile_mma(Ctile, Atile, Btile, Ctile);
      As += tile_stride_a;
      Bs += tile_stride_b;
    }
  }
};

// Copies a BROWS x BCOLS block of S from device rows (stride src_ld) into
// FP32 threadgroup rows (stride dst_ld), each thread one contiguous run of
// one row; next() steps along the reduction dimension.
template <
    typename S,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size,
    short n_reads = (BCOLS * BROWS) / (tgp_size),
    short TCOLS = BCOLS / n_reads,
    short TROWS = tgp_size / TCOLS>
struct WideningLoader {
  LRF_CONST short vec_size = n_reads;

  const int src_ld;
  const int tile_stride;
  const short bi;
  const short bj;
  threadgroup float* dst;
  const device S* src;

  struct alignas(sizeof(S)) ReadVector {
    S v[vec_size];
  };

  METAL_FUNC WideningLoader(
      const device S* src_,
      const int src_ld_,
      threadgroup float* dst_,
      ushort simd_group_id,
      ushort simd_lane_id) thread
      : src_ld(src_ld_),
        tile_stride(reduction_dim ? BCOLS : BROWS * src_ld_),
        bi((simd_group_id * 32 + simd_lane_id) / TCOLS),
        bj(vec_size * ((simd_group_id * 32 + simd_lane_id) % TCOLS)),
        dst(dst_ + bi * dst_ld + bj),
        src(src_ + bi * src_ld_ + bj) {}

  METAL_FUNC void load_unsafe() const thread {
    LRF_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      const ReadVector r = *((const device ReadVector*)(&src[i * src_ld]));
      LRF_UNROLL
      for (short j = 0; j < vec_size; j++) {
        dst[i * dst_ld + j] = static_cast<float>(r.v[j]);
      }
    }
  }

  METAL_FUNC void load_safe(short2 src_tile_dim) const thread {
    src_tile_dim = src_tile_dim - short2(bj, bi);
    LRF_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      LRF_UNROLL
      for (short j = 0; j < vec_size; j++) {
        const bool valid = (i < src_tile_dim.y) && (j < src_tile_dim.x);
        const S value = src[valid ? i * src_ld + j : 0];
        dst[i * dst_ld + j] = valid ? static_cast<float>(value) : 0.0f;
      }
    }
  }

  METAL_FUNC void next() thread {
    src += tile_stride;
  }
};

// The same copy for BF16 factors, writing only each FP32 word's high 16 bits:
// with the low 16 bits zeroed once per threadgroup, the word is the BF16 value
// widened exactly, and staging needs no instruction to build it.
template <
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size,
    short n_reads = (BCOLS * BROWS) / (tgp_size),
    short TCOLS = BCOLS / n_reads,
    short TROWS = tgp_size / TCOLS>
struct HighHalfLoader {
  LRF_CONST short vec_size = n_reads;

  const int src_ld;
  const int tile_stride;
  const short bi;
  const short bj;
  threadgroup ushort* dst;
  const device bfloat* src;

  struct alignas(sizeof(bfloat)) ReadVector {
    ushort v[vec_size];
  };

  METAL_FUNC HighHalfLoader(
      const device bfloat* src_,
      const int src_ld_,
      threadgroup float* dst_,
      ushort simd_group_id,
      ushort simd_lane_id) thread
      : src_ld(src_ld_),
        tile_stride(reduction_dim ? BCOLS : BROWS * src_ld_),
        bi((simd_group_id * 32 + simd_lane_id) / TCOLS),
        bj(vec_size * ((simd_group_id * 32 + simd_lane_id) % TCOLS)),
        dst(reinterpret_cast<threadgroup ushort*>(dst_ + bi * dst_ld + bj) + 1),
        src(src_ + bi * src_ld_ + bj) {}

  METAL_FUNC void load_unsafe() const thread {
    LRF_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      const ReadVector r = *((const device ReadVector*)(&src[i * src_ld]));
      LRF_UNROLL
      for (short j = 0; j < vec_size; j++) {
        dst[2 * (i * dst_ld + j)] = r.v[j];
      }
    }
  }

  METAL_FUNC void load_safe(short2 src_tile_dim) const thread {
    src_tile_dim = src_tile_dim - short2(bj, bi);
    LRF_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      LRF_UNROLL
      for (short j = 0; j < vec_size; j++) {
        const bool valid = (i < src_tile_dim.y) && (j < src_tile_dim.x);
        const ushort value = as_type<ushort>(src[valid ? i * src_ld + j : 0]);
        dst[2 * (i * dst_ld + j)] = valid ? value : ushort(0);
      }
    }
  }

  METAL_FUNC void next() thread {
    src += tile_stride;
  }
};

template <typename S, short BROWS, short BCOLS, short dst_ld, short reduction_dim, short tgp_size>
struct StagingLoader {
  using type = WideningLoader<S, BROWS, BCOLS, dst_ld, reduction_dim, tgp_size>;
};

template <short BROWS, short BCOLS, short dst_ld, short reduction_dim, short tgp_size>
struct StagingLoader<bfloat, BROWS, BCOLS, dst_ld, reduction_dim, tgp_size> {
  using type = HighHalfLoader<BROWS, BCOLS, dst_ld, reduction_dim, tgp_size>;
};

}  // namespace lrf
