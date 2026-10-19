// FP16 dense attention body fragment consumed by mx.fast.metal_kernel.
// MLX supplies the kernel wrapper and input/output pointer declarations.
// Built on Apple MLX STEEL attention sources.
// Copyright (c) 2024-25 Apple Inc.
// SPDX-License-Identifier: MIT
//
// One threadgroup of four SIMD groups per (query tile, head) over token-major
// inputs: queries (TOKENS_Q, heads * D) and keys and values (TOKENS_K,
// heads * D), D 64 or 128. Each SIMD group owns QF 8-row query fragments (1 or
// 2), so a query tile is 32 * QF rows. Every BK-key tile (16 or 32) is staged
// with two barriers, scored, and folded into one online softmax per row, then
// the partial tail tile with its missing keys masked. The first tile is a full
// one unless there are fewer keys than a tile, so no row meets a masked tile
// before its running maximum is finite. The full and partial tile stages repeat
// one sequence on purpose; register allocation was measured on this form.

  using namespace sol;
  constexpr int WM = 4;
  constexpr int BQ = WM * 8 * QF;
  constexpr int THREADS = WM * 32;
  constexpr int LDT = H * D;
  constexpr short LDQ = D + 8;
  constexpr short LDK = BK + 8;
  constexpr short LDV = D + 8;
  constexpr int K_OFF = 0;
  constexpr int V_OFF = D * LDK;
  constexpr int KV_SIZE = V_OFF + BK * LDV;
  constexpr int SMEM = BQ * LDQ > KV_SIZE ? BQ * LDQ : KV_SIZE;
  constexpr short TK = BK / 8;
  constexpr short TD = D / 8;
  constexpr int FULL_TILES = TOKENS_K / BK;
  constexpr int TAIL = TOKENS_K % BK;
  static_assert(D == 64 || D == 128, "head dimension 64 or 128");
  static_assert(QF == 1 || QF == 2, "one or two query fragments per SIMD group");

  const ushort lane = ushort(thread_index_in_simdgroup);
  const ushort sg = ushort(simdgroup_index_in_threadgroup);
  const int qt = int(threadgroup_position_in_grid.x);
  const int head = int(threadgroup_position_in_grid.y);

  // Q is staged once and read into registers; the same memory then holds the
  // K tile (transposed) followed by the V tile.
  threadgroup T smem[SMEM];

  using QLoader = BlockLoader<T, BQ, D, LDQ, 1, THREADS>;
  using KLoader = SpreadKLoader<T, D, BK, LDK>;
  using VLoader = RowVLoader<T, D, BK, LDV>;

  const int q_start = qt * BQ;
  const short q_valid = short(min(BQ, TOKENS_Q - q_start));
  QLoader loader_q(q + q_start * LDT + head * D, LDT, &smem[0], sg, lane);
  if (q_valid < BQ) {
    loader_q.load_rows(q_valid);
  } else {
    loader_q.load_unsafe();
  }

  // Fragment row i of this SIMD group is query row tm + 8 i + sm.
  const short qid = short(lane / 4);
  const short sm = (qid & 4) + ((lane / 2) % 4);
  const short sn = (qid & 2) * 2 + (lane % 2) * 2;
  const short tm = 8 * QF * short(sg);
  const short Qs_offset = (tm + sm) * LDQ + sn;
  const short Ks_offset = sm * LDK + sn;
  const short Vs_offset = sm * LDV + sn;

  vec<T, 2> q_frags[QF][TD];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < QF; i++) {
    STEEL_PRAGMA_UNROLL
    for (short dd = 0; dd < TD; dd++) {
      const short at = Qs_offset + i * 8 * LDQ + dd * 8;
      q_frags[i][dd] = vec<T, 2>(smem[at], smem[at + 1]);
    }
  }

  const float scale = M_LOG2E_F / sqrt(float(D));
  constexpr float neg_inf = -3.4028234663852886e+38F;

  RowTile<1> Qtile[QF];
  RowTile<TK> Ktile;
  RowTile<TK> Stile[QF];
  RowTile<1> Vtile;
  RowTile<TD> Otile[QF];
  float max_score[QF];
  float sum_score[QF];
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < QF; i++) {
    Otile[i].clear();
    max_score[i] = neg_inf;
    sum_score[i] = 0.0f;
  }

  {
    KLoader loader_k(k + head * D, LDT, &smem[K_OFF], sg, lane);
    VLoader loader_v(v + head * D, LDT, &smem[V_OFF], sg, lane);
    for (int t = 0; t < FULL_TILES; t++) {
      loader_k.fetch();
      loader_v.fetch();
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_k.commit();
      loader_v.commit();
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < QF; i++) {
        Stile[i].clear();
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      STEEL_PRAGMA_UNROLL
      for (short dd = 0; dd < TD; dd++) {
        simdgroup_barrier(mem_flags::mem_none);
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < QF; i++) {
          Qtile[i].frag_at(0) = float2(float(q_frags[i][dd].x), float(q_frags[i][dd].y));
        }
        Ktile.load(&smem[K_OFF + Ks_offset + dd * 8 * LDK]);
        simdgroup_barrier(mem_flags::mem_none);
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < QF; i++) {
          STEEL_PRAGMA_UNROLL
          for (short ik = 0; ik < TK; ik++) {
            mma_fragment(
                Stile[i].frag_at(ik),
                Qtile[i].frag_at(0),
                Ktile.frag_at(ik),
                Stile[i].frag_at(ik));
          }
        }
      }

      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < QF; i++) {
        Stile[i].scale_by(scale);
        float new_max[1] = {max_score[i]};
        Stile[i].row_max(new_max);
        Stile[i].exp2_sub(new_max);
        float tile_sum[1] = {0.0f};
        Stile[i].row_sum(tile_sum);
        if (simd_any(new_max[0] != max_score[i])) {
          simdgroup_barrier(mem_flags::mem_none);
          float factor[1] = {fast::exp2(max_score[i] - new_max[0])};
          sum_score[i] = sum_score[i] * factor[0] + tile_sum[0];
          Otile[i].mul_by(factor);
        } else {
          sum_score[i] = sum_score[i] + tile_sum[0];
        }
        max_score[i] = new_max[0];
      }

      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < TD; id++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          simdgroup_barrier(mem_flags::mem_none);
          Vtile.load(&smem[V_OFF + Vs_offset + ik * 8 * LDV + id * 8]);
          simdgroup_barrier(mem_flags::mem_none);
          STEEL_PRAGMA_UNROLL
          for (short i = 0; i < QF; i++) {
            mma_fragment(
                Otile[i].frag_at(id),
                Stile[i].frag_at(ik),
                Vtile.frag_at(0),
                Otile[i].frag_at(id));
          }
        }
      }

      loader_k.next();
      loader_v.next();
    }
    if (TAIL > 0) {
      const short valid = short(TAIL);
      loader_k.fetch_rows(valid);
      loader_v.fetch_rows(valid);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_k.commit();
      loader_v.commit();
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < QF; i++) {
        Stile[i].clear();
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      STEEL_PRAGMA_UNROLL
      for (short dd = 0; dd < TD; dd++) {
        simdgroup_barrier(mem_flags::mem_none);
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < QF; i++) {
          Qtile[i].frag_at(0) = float2(float(q_frags[i][dd].x), float(q_frags[i][dd].y));
        }
        Ktile.load(&smem[K_OFF + Ks_offset + dd * 8 * LDK]);
        simdgroup_barrier(mem_flags::mem_none);
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < QF; i++) {
          STEEL_PRAGMA_UNROLL
          for (short ik = 0; ik < TK; ik++) {
            mma_fragment(
                Stile[i].frag_at(ik),
                Qtile[i].frag_at(0),
                Ktile.frag_at(ik),
                Stile[i].frag_at(ik));
          }
        }
      }

      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < QF; i++) {
        Stile[i].scale_by(scale);

        // Keys past the end of the sequence take no weight.
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < TK; j++) {
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < 2; jj++) {
            if (sn + j * 8 + jj >= valid) {
              Stile[i].frag_at(j)[jj] = neg_inf;
            }
          }
        }

        float new_max[1] = {max_score[i]};
        Stile[i].row_max(new_max);
        Stile[i].exp2_sub(new_max);
        float tile_sum[1] = {0.0f};
        Stile[i].row_sum(tile_sum);
        if (simd_any(new_max[0] != max_score[i])) {
          simdgroup_barrier(mem_flags::mem_none);
          float factor[1] = {fast::exp2(max_score[i] - new_max[0])};
          sum_score[i] = sum_score[i] * factor[0] + tile_sum[0];
          Otile[i].mul_by(factor);
        } else {
          sum_score[i] = sum_score[i] + tile_sum[0];
        }
        max_score[i] = new_max[0];
      }

      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < TD; id++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          simdgroup_barrier(mem_flags::mem_none);
          Vtile.load(&smem[V_OFF + Vs_offset + ik * 8 * LDV + id * 8]);
          simdgroup_barrier(mem_flags::mem_none);
          STEEL_PRAGMA_UNROLL
          for (short i = 0; i < QF; i++) {
            mma_fragment(
                Otile[i].frag_at(id),
                Stile[i].frag_at(ik),
                Vtile.frag_at(0),
                Otile[i].frag_at(id));
          }
        }
      }
    }
  }

  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < QF; i++) {
    Otile[i].div_by(&sum_score[i]);
    const short out_row = tm + 8 * i + sm;
    if (out_row < q_valid) {
      Otile[i].store(out + (q_start + out_row) * LDT + head * D + sn);
    }
  }

// The wrapper's header redefines `kernel` to cap threads per threadgroup;
// MLX's explicit instantiation that follows must not carry the attribute.
#undef kernel
