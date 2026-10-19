// FP16 Sol-Attn body fragment consumed by mx.fast.metal_kernel.
// MLX supplies the kernel wrapper and input/output pointer declarations.
// Built on Apple MLX STEEL attention sources.
// Copyright (c) 2024-25 Apple Inc.
// SPDX-License-Identifier: MIT
//
// One threadgroup of four SIMD groups per (32-query tile, head). Inputs are in
// routing order; row t of the output is stored at row rows[t]. The exact pass
// walks the query block's ascending list of routed 64-token key blocks; the
// centroid pass then scores every block's mean key with the routed blocks
// masked out, adds log2 of the block length, and uses the block's mean value.
// Both feed one online softmax. The exact pass runs first: it always holds the
// query's own block, so no row meets a fully masked tile before its running
// maximum is finite.
//
// The four tile stages below (exact and centroid, full and partial) repeat one
// sequence on purpose: stage K and V with two barriers, Q K^T, a fix-up, the
// guarded online-softmax update, and P V. Each copy compiles to its own
// unrolled loop; register allocation was measured on this form.

  using namespace sol;
  constexpr int BQ = 32;
  constexpr int BK = 16;
  constexpr int WM = 4;
  constexpr int THREADS = WM * 32;
  constexpr int BLOCK = 64;
  constexpr int LDT = H * D;
  constexpr short LDQ = D + 8;
  constexpr short LDK = BK + 8;
  constexpr short LDV = D + 8;
  constexpr int K_OFF = 0;
  constexpr int V_OFF = D * LDK;
  constexpr short TK = BK / 8;
  constexpr short TD = D / 8;
  constexpr short TILES_PER_BLOCK = BLOCK / BK;
  constexpr int NCHUNK = (NB + BK - 1) / BK;
  static_assert(D == 128, "head dimension 128");
  static_assert(NW * 32 >= NB, "routing words cover every block");

  const ushort lane = ushort(thread_index_in_simdgroup);
  const ushort sg = ushort(simdgroup_index_in_threadgroup);
  const int qt = int(threadgroup_position_in_grid.x);
  const int head = int(threadgroup_position_in_grid.y);

  // Q is staged once and read into registers; the same memory then holds the
  // K tile (transposed) followed by the V tile.
  threadgroup T smem[V_OFF + BK * LDV];

  using QLoader = BlockLoader<T, BQ, D, LDQ, 1, THREADS>;
  using KLoader = SpreadKLoader<T, D, BK, LDK>;
  using VLoader = RowVLoader<T, D, BK, LDV>;

  const int q_start = qt * BQ;
  const short q_valid = short(min(BQ, TOKENS - q_start));
  const int route_row = head * NB + q_start / BLOCK;
  QLoader loader_q(q + q_start * LDT + head * D, LDT, &smem[0], sg, lane);
  if (q_valid < BQ) {
    loader_q.load_rows(q_valid);
  } else {
    loader_q.load_unsafe();
  }

  const short qid = short(lane / 4);
  const short sm = (qid & 4) + ((lane / 2) % 4);
  const short sn = (qid & 2) * 2 + (lane % 2) * 2;
  const short tm = 8 * short(sg);
  const short Qs_offset = (tm + sm) * LDQ + sn;
  const short Ks_offset = sm * LDK + sn;
  const short Vs_offset = sm * LDV + sn;

  vec<T, 2> q_frags[TD];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  STEEL_PRAGMA_UNROLL
  for (short dd = 0; dd < TD; dd++) {
    q_frags[dd] = vec<T, 2>(smem[Qs_offset + dd * 8], smem[Qs_offset + dd * 8 + 1]);
  }

  const float scale = M_LOG2E_F / sqrt(float(D));
  constexpr float neg_inf = -3.4028234663852886e+38F;

  RowTile<1> Qtile;
  RowTile<TK> Ktile;
  RowTile<TK> Stile;
  RowTile<1> Vtile;
  RowTile<TD> Otile;
  Otile.clear();
  float max_score[1] = {neg_inf};
  float sum_score[1] = {0.0f};

  // Exact pass over the routed key blocks.
  {
    const int count = min(counts[route_row], NB);
    const device ushort* row_list = lists + route_row * NB;
    for (int i = 0; i < count; i++) {
      const int kb = min(int(row_list[i]), NB - 1);
      const int k_start = kb * BLOCK;
      const int block_valid = min(BLOCK, TOKENS - k_start);
      KLoader loader_k(k + k_start * LDT + head * D, LDT, &smem[K_OFF], sg, lane);
      VLoader loader_v(v + k_start * LDT + head * D, LDT, &smem[V_OFF], sg, lane);
      if (block_valid == BLOCK) {
        for (short t = 0; t < TILES_PER_BLOCK; t++) {
          loader_k.fetch();
          loader_v.fetch();
          threadgroup_barrier(mem_flags::mem_threadgroup);
          loader_k.commit();
          loader_v.commit();
          Stile.clear();
          threadgroup_barrier(mem_flags::mem_threadgroup);

          STEEL_PRAGMA_UNROLL
          for (short dd = 0; dd < TD; dd++) {
            simdgroup_barrier(mem_flags::mem_none);
            Qtile.frag_at(0) = float2(float(q_frags[dd].x), float(q_frags[dd].y));
            Ktile.load(&smem[K_OFF + Ks_offset + dd * 8 * LDK]);
            simdgroup_barrier(mem_flags::mem_none);
            STEEL_PRAGMA_UNROLL
            for (short ik = 0; ik < TK; ik++) {
              mma_fragment(
                  Stile.frag_at(ik), Qtile.frag_at(0), Ktile.frag_at(ik), Stile.frag_at(ik));
            }
          }
          Stile.scale_by(scale);

          {
            float new_max[1] = {max_score[0]};
            Stile.row_max(new_max);
            Stile.exp2_sub(new_max);
            float tile_sum[1] = {0.0f};
            Stile.row_sum(tile_sum);
            if (simd_any(new_max[0] != max_score[0])) {
              simdgroup_barrier(mem_flags::mem_none);
              float factor[1] = {fast::exp2(max_score[0] - new_max[0])};
              sum_score[0] = sum_score[0] * factor[0] + tile_sum[0];
              Otile.mul_by(factor);
            } else {
              sum_score[0] = sum_score[0] + tile_sum[0];
            }
            max_score[0] = new_max[0];
          }

          STEEL_PRAGMA_UNROLL
          for (short id = 0; id < TD; id++) {
            STEEL_PRAGMA_UNROLL
            for (short ik = 0; ik < TK; ik++) {
              simdgroup_barrier(mem_flags::mem_none);
              Vtile.load(&smem[V_OFF + Vs_offset + ik * 8 * LDV + id * 8]);
              simdgroup_barrier(mem_flags::mem_none);
              mma_fragment(
                  Otile.frag_at(id), Stile.frag_at(ik), Vtile.frag_at(0), Otile.frag_at(id));
            }
          }

          loader_k.next();
          loader_v.next();
        }
      } else {
        for (short t = 0; t < TILES_PER_BLOCK; t++) {
          const short valid = short(min(int(BK), block_valid - t * BK));
          if (valid <= 0) {
            break;
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          loader_k.load_rows(valid);
          loader_v.load_rows(valid);
          Stile.clear();
          threadgroup_barrier(mem_flags::mem_threadgroup);

          STEEL_PRAGMA_UNROLL
          for (short dd = 0; dd < TD; dd++) {
            simdgroup_barrier(mem_flags::mem_none);
            Qtile.frag_at(0) = float2(float(q_frags[dd].x), float(q_frags[dd].y));
            Ktile.load(&smem[K_OFF + Ks_offset + dd * 8 * LDK]);
            simdgroup_barrier(mem_flags::mem_none);
            STEEL_PRAGMA_UNROLL
            for (short ik = 0; ik < TK; ik++) {
              mma_fragment(
                  Stile.frag_at(ik), Qtile.frag_at(0), Ktile.frag_at(ik), Stile.frag_at(ik));
            }
          }
          Stile.scale_by(scale);

          // Keys past the end of the sequence take no weight.
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < TK; j++) {
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < 2; jj++) {
              if (sn + j * 8 + jj >= valid) {
                Stile.frag_at(j)[jj] = neg_inf;
              }
            }
          }

          {
            float new_max[1] = {max_score[0]};
            Stile.row_max(new_max);
            Stile.exp2_sub(new_max);
            float tile_sum[1] = {0.0f};
            Stile.row_sum(tile_sum);
            if (simd_any(new_max[0] != max_score[0])) {
              simdgroup_barrier(mem_flags::mem_none);
              float factor[1] = {fast::exp2(max_score[0] - new_max[0])};
              sum_score[0] = sum_score[0] * factor[0] + tile_sum[0];
              Otile.mul_by(factor);
            } else {
              sum_score[0] = sum_score[0] + tile_sum[0];
            }
            max_score[0] = new_max[0];
          }

          STEEL_PRAGMA_UNROLL
          for (short id = 0; id < TD; id++) {
            STEEL_PRAGMA_UNROLL
            for (short ik = 0; ik < TK; ik++) {
              simdgroup_barrier(mem_flags::mem_none);
              Vtile.load(&smem[V_OFF + Vs_offset + ik * 8 * LDV + id * 8]);
              simdgroup_barrier(mem_flags::mem_none);
              mma_fragment(
                  Otile.frag_at(id), Stile.frag_at(ik), Vtile.frag_at(0), Otile.frag_at(id));
            }
          }

          loader_k.next();
          loader_v.next();
        }
      }
    }
  }

  // Centroid pass over every block's mean key, 16 blocks per tile.
  {
    KLoader loader_k(kmean + head * NB * D, D, &smem[K_OFF], sg, lane);
    VLoader loader_v(vmean + head * NB * D, D, &smem[V_OFF], sg, lane);
    const device uint* words = routed_bits + route_row * NW;
    const float full_bias = log2(float(BLOCK));
    const float last_bias = log2(float(LAST_LEN));
    for (int c = 0; c < NCHUNK; c++) {
      const int c0 = c * BK;
      const short valid = short(min(int(BK), NB - c0));
      const uint word = words[c0 / 32] >> uint(c0 % 32);
      if (valid == BK) {
        loader_k.fetch();
        loader_v.fetch();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_k.commit();
        loader_v.commit();
        Stile.clear();
        threadgroup_barrier(mem_flags::mem_threadgroup);

        STEEL_PRAGMA_UNROLL
        for (short dd = 0; dd < TD; dd++) {
          simdgroup_barrier(mem_flags::mem_none);
          Qtile.frag_at(0) = float2(float(q_frags[dd].x), float(q_frags[dd].y));
          Ktile.load(&smem[K_OFF + Ks_offset + dd * 8 * LDK]);
          simdgroup_barrier(mem_flags::mem_none);
          STEEL_PRAGMA_UNROLL
          for (short ik = 0; ik < TK; ik++) {
            mma_fragment(
                Stile.frag_at(ik), Qtile.frag_at(0), Ktile.frag_at(ik), Stile.frag_at(ik));
          }
        }
        Stile.scale_by(scale);

        // Routed blocks were computed exactly; every other mean key stands in
        // for its whole block.
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < TK; j++) {
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < 2; jj++) {
            const short col = sn + j * 8 + jj;
            const bool skip = ((word >> uint(col)) & 1u) != 0u;
            const float bias = (c0 + col == NB - 1) ? last_bias : full_bias;
            Stile.frag_at(j)[jj] = skip ? neg_inf : Stile.frag_at(j)[jj] + bias;
          }
        }

        {
          float new_max[1] = {max_score[0]};
          Stile.row_max(new_max);
          Stile.exp2_sub(new_max);
          float tile_sum[1] = {0.0f};
          Stile.row_sum(tile_sum);
          if (simd_any(new_max[0] != max_score[0])) {
            simdgroup_barrier(mem_flags::mem_none);
            float factor[1] = {fast::exp2(max_score[0] - new_max[0])};
            sum_score[0] = sum_score[0] * factor[0] + tile_sum[0];
            Otile.mul_by(factor);
          } else {
            sum_score[0] = sum_score[0] + tile_sum[0];
          }
          max_score[0] = new_max[0];
        }

        STEEL_PRAGMA_UNROLL
        for (short id = 0; id < TD; id++) {
          STEEL_PRAGMA_UNROLL
          for (short ik = 0; ik < TK; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            Vtile.load(&smem[V_OFF + Vs_offset + ik * 8 * LDV + id * 8]);
            simdgroup_barrier(mem_flags::mem_none);
            mma_fragment(
                Otile.frag_at(id), Stile.frag_at(ik), Vtile.frag_at(0), Otile.frag_at(id));
          }
        }
      } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_k.load_rows(valid);
        loader_v.load_rows(valid);
        Stile.clear();
        threadgroup_barrier(mem_flags::mem_threadgroup);

        STEEL_PRAGMA_UNROLL
        for (short dd = 0; dd < TD; dd++) {
          simdgroup_barrier(mem_flags::mem_none);
          Qtile.frag_at(0) = float2(float(q_frags[dd].x), float(q_frags[dd].y));
          Ktile.load(&smem[K_OFF + Ks_offset + dd * 8 * LDK]);
          simdgroup_barrier(mem_flags::mem_none);
          STEEL_PRAGMA_UNROLL
          for (short ik = 0; ik < TK; ik++) {
            mma_fragment(
                Stile.frag_at(ik), Qtile.frag_at(0), Ktile.frag_at(ik), Stile.frag_at(ik));
          }
        }
        Stile.scale_by(scale);

        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < TK; j++) {
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < 2; jj++) {
            const short col = sn + j * 8 + jj;
            const bool skip = col >= valid || ((word >> uint(col)) & 1u) != 0u;
            const float bias = (c0 + col == NB - 1) ? last_bias : full_bias;
            Stile.frag_at(j)[jj] = skip ? neg_inf : Stile.frag_at(j)[jj] + bias;
          }
        }

        {
          float new_max[1] = {max_score[0]};
          Stile.row_max(new_max);
          Stile.exp2_sub(new_max);
          float tile_sum[1] = {0.0f};
          Stile.row_sum(tile_sum);
          if (simd_any(new_max[0] != max_score[0])) {
            simdgroup_barrier(mem_flags::mem_none);
            float factor[1] = {fast::exp2(max_score[0] - new_max[0])};
            sum_score[0] = sum_score[0] * factor[0] + tile_sum[0];
            Otile.mul_by(factor);
          } else {
            sum_score[0] = sum_score[0] + tile_sum[0];
          }
          max_score[0] = new_max[0];
        }

        STEEL_PRAGMA_UNROLL
        for (short id = 0; id < TD; id++) {
          STEEL_PRAGMA_UNROLL
          for (short ik = 0; ik < TK; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            Vtile.load(&smem[V_OFF + Vs_offset + ik * 8 * LDV + id * 8]);
            simdgroup_barrier(mem_flags::mem_none);
            mma_fragment(
                Otile.frag_at(id), Stile.frag_at(ik), Vtile.frag_at(0), Otile.frag_at(id));
          }
        }
      }
      loader_k.next();
      loader_v.next();
    }
  }

  Otile.div_by(sum_score);
  const short out_row = tm + sm;
  if (out_row < q_valid) {
    Otile.store(out + int(rows[q_start + out_row]) * LDT + head * D + sn);
  }

// The wrapper's header redefines `kernel` to cap threads per threadgroup;
// MLX's explicit instantiation that follows must not carry the attribute.
#undef kernel
