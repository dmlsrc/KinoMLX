// Exact 3D neighborhood attention over 16-key runs along W, a body fragment for
// mx.fast.metal_kernel on the FP16 STEEL attention primitives of
// fp16_attention_header.metal (derived from Apple MLX STEEL attention sources,
// Copyright (c) 2024-25 Apple Inc., SPDX-License-Identifier: MIT).
//
// Inputs are token-major: q, k and v are (B, GT * GH * GW, HEADS * D) with the
// query already scaled, so a score is a plain dot product. Every query attends
// to the KT x KH x KW keys of its window, which keeps its full size at the grid
// edges by shifting inward: on an axis of length L the window of position i
// starts at clamp(i - K / 2, 0, L - K).
//
// One threadgroup of four SIMD groups computes a QT x QH x QW block of 32
// queries of one head: query row r is (r / (QH QW), (r / QW) % QH, r % QW)
// inside the block, and SIMD group s owns rows 8 s .. 8 s + 7. The union of the
// block's windows lies inside a key box of UT = QT + KT - 1 planes by
// UH = QH + KH - 1 rows by one BK-key run of W, positioned inside the grid. The
// box is walked one (plane, row) run at a time: the run's BK keys are
// consecutive tokens, staged with two barriers like a dense tile, scored, and
// masked to each query's window. The W part of a query's window is fixed over
// the walk, so each thread decides once which of its score columns it can see;
// the plane and row parts are one compare each per run. A SIMD group none of
// whose rows sees a run skips its products: every score would be masked, so the
// skip is exact.
//
// Masked scores are -FLT_MAX, the finite stand-in the dense kernel uses (fast
// math may assume no infinities). A row that has seen no visible key yet keeps
// that running maximum, and its exponents are taken against 0 instead of it, so
// a masked score's weight is exp2(-FLT_MAX) = 0 rather than exp2(0) = 1. No
// real score equals -FLT_MAX, so the test is exact.

  using namespace sol;
  constexpr int WM = 4;
  constexpr int BQ = WM * 8;
  constexpr int LDT = HEADS * D;
  constexpr short LDQ = D + 8;
  constexpr short LDK = BK + 8;
  constexpr short LDV = D + 8;
  constexpr int K_OFF = 0;
  constexpr int V_OFF = D * LDK;
  constexpr int KV_SIZE = V_OFF + BK * LDV;
  constexpr int SMEM = BQ * LDQ > KV_SIZE ? BQ * LDQ : KV_SIZE;
  constexpr short TK = BK / 8;
  constexpr short TD = D / 8;
  constexpr int UT = QT + KT - 1;
  constexpr int UH = QH + KH - 1;
  constexpr int TILES_W = (GW + QW - 1) / QW;
  constexpr int TILES_H = (GH + QH - 1) / QH;
  constexpr int TOKENS = GT * GH * GW;
  static_assert(D == 64, "64-wide heads");
  static_assert(QT * QH * QW == BQ, "the query block fills the threadgroup's rows");
  static_assert(QW + KW - 1 <= BK, "one key run spans the block's W union");
  static_assert(GT >= UT && GH >= UH && GW >= BK, "the grid holds the key box");
  static_assert(KT % 2 == 1 && KH % 2 == 1 && KW % 2 == 1, "odd windows");

  const ushort lane = ushort(thread_index_in_simdgroup);
  const ushort sg = ushort(simdgroup_index_in_threadgroup);
  const ushort tid = ushort(sg * 32 + lane);
  const int tile = int(threadgroup_position_in_grid.x);
  const int head = int(threadgroup_position_in_grid.y);
  const int batch = int(threadgroup_position_in_grid.z);
  const int t0 = (tile / (TILES_W * TILES_H)) * QT;
  const int h0 = ((tile / TILES_W) % TILES_H) * QH;
  const int w0 = (tile % TILES_W) * QW;
  // A 64-bit base once per threadgroup and a 64-bit run address once per run; the offsets
  // inside a run stay 32-bit, so the volume and the key box may exceed 2^31 elements.
  const size_t base = size_t(batch) * size_t(TOKENS) * LDT + size_t(head) * D;

  threadgroup T smem[SMEM];

  // Stage Q: two threads per row, D / 2 elements each, so the first 2 BQ threads stage
  // the BQ rows (rows past the grid edge read the edge query and are never stored). The
  // other threads must not write: rows past BQ would land outside this threadgroup's memory.
  {
    constexpr short HALF = D / 2;
    const short r = short(tid / 2);
    const short c0 = short((tid % 2) * HALF);
    if (r < BQ) {
      const int t = min(t0 + r / (QH * QW), GT - 1);
      const int h = min(h0 + (r / QW) % QH, GH - 1);
      const int w = min(w0 + r % QW, GW - 1);
      const device T* src = q + base + size_t((t * GH + h) * GW + w) * LDT + c0;
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < HALF; j += 4) {
        *reinterpret_cast<threadgroup vec<T, 4>*>(&smem[r * LDQ + c0 + j]) =
            *reinterpret_cast<const device vec<T, 4>*>(src + j);
      }
    }
  }

  // This thread's fragment row is query row 8 sg + sm.
  const short qid = short(lane / 4);
  const short sm = (qid & 4) + ((lane / 2) % 4);
  const short sn = (qid & 2) * 2 + (lane % 2) * 2;
  const short Qs_offset = (8 * short(sg) + sm) * LDQ + sn;
  const short Ks_offset = sm * LDK + sn;
  const short Vs_offset = sm * LDV + sn;

  vec<T, 2> q_frags[TD];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  STEEL_PRAGMA_UNROLL
  for (short dd = 0; dd < TD; dd++) {
    const short at = Qs_offset + dd * 8;
    q_frags[dd] = vec<T, 2>(smem[at], smem[at + 1]);
  }

  // The key box: plane tu0 .. tu0 + UT, row hu0 .. hu0 + UH, run wu0 .. wu0 + BK.
  const int tu0 = min(clamp(t0 - KT / 2, 0, GT - KT), GT - UT);
  const int hu0 = min(clamp(h0 - KH / 2, 0, GH - KH), GH - UH);
  const int wu0 = min(clamp(w0 - KW / 2, 0, GW - KW), GW - BK);

  // The fragment row's window starts, and which of the thread's score columns
  // fall inside its W window.
  const int r = 8 * int(sg) + sm;
  const bool row_valid =
      (t0 + r / (QH * QW) < GT) && (h0 + (r / QW) % QH < GH) && (w0 + r % QW < GW);
  const int t_c = min(t0 + r / (QH * QW), GT - 1);
  const int h_c = min(h0 + (r / QW) % QH, GH - 1);
  const int w_c = min(w0 + r % QW, GW - 1);
  const int out_token = (t_c * GH + h_c) * GW + w_c;
  const int ts = clamp(t_c - KT / 2, 0, GT - KT);
  const int hs = clamp(h_c - KH / 2, 0, GH - KH);
  const int ws = clamp(w_c - KW / 2, 0, GW - KW);
  bool wvis[TK][2];
  STEEL_PRAGMA_UNROLL
  for (short j = 0; j < TK; j++) {
    STEEL_PRAGMA_UNROLL
    for (short jj = 0; jj < 2; jj++) {
      wvis[j][jj] = uint(wu0 + sn + 8 * j + jj - ws) < uint(KW);
    }
  }

  const float scale = M_LOG2E_F;
  constexpr float neg_inf = -3.4028234663852886e+38F;

  RowTile<1> Qtile;
  RowTile<TK> Ktile;
  RowTile<TK> Stile;
  RowTile<1> Vtile;
  RowTile<TD> Otile;
  float max_score = neg_inf;
  float sum_score = 0.0f;
  Otile.clear();

  using KLoader = SpreadKLoader<T, D, BK, LDK>;
  using VLoader = RowVLoader<T, D, BK, LDV>;
  const device T* kbox = k + base + size_t(tu0) * GH * GW * LDT;
  const device T* vbox = v + base + size_t(tu0) * GH * GW * LDT;

  for (int ti = 0; ti < UT; ti++) {
    const int t = tu0 + ti;
    const bool tvis = uint(t - ts) < uint(KT);
    for (int hi = 0; hi < UH; hi++) {
      const int h = hu0 + hi;
      const int run_token = ((t - tu0) * GH + h) * GW + wu0;
      KLoader loader_k(kbox + size_t(run_token) * LDT, LDT, &smem[K_OFF], sg, lane);
      VLoader loader_v(vbox + size_t(run_token) * LDT, LDT, &smem[V_OFF], sg, lane);
      loader_k.fetch();
      loader_v.fetch();
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_k.commit();
      loader_v.commit();
      Stile.clear();
      threadgroup_barrier(mem_flags::mem_threadgroup);

      const bool rowvis = tvis && (uint(h - hs) < uint(KH));
      // The barrier keeps the skipped region a real branch.
      if (simd_any(rowvis)) {
        simdgroup_barrier(mem_flags::mem_none);
        STEEL_PRAGMA_UNROLL
        for (short dd = 0; dd < TD; dd++) {
          simdgroup_barrier(mem_flags::mem_none);
          Qtile.frag_at(0) = float2(float(q_frags[dd].x), float(q_frags[dd].y));
          Ktile.load(&smem[K_OFF + Ks_offset + dd * 8 * LDK]);
          simdgroup_barrier(mem_flags::mem_none);
          STEEL_PRAGMA_UNROLL
          for (short ik = 0; ik < TK; ik++) {
            mma_fragment(Stile.frag_at(ik), Qtile.frag_at(0), Ktile.frag_at(ik), Stile.frag_at(ik));
          }
        }

        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < TK; j++) {
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < 2; jj++) {
            const float s = Stile.frag_at(j)[jj] * scale;
            Stile.frag_at(j)[jj] = (rowvis && wvis[j][jj]) ? s : neg_inf;
          }
        }
        float new_max[1] = {max_score};
        Stile.row_max(new_max);
        float shift[1] = {new_max[0] == neg_inf ? 0.0f : new_max[0]};
        Stile.exp2_sub(shift);
        float tile_sum[1] = {0.0f};
        Stile.row_sum(tile_sum);
        if (simd_any(new_max[0] != max_score)) {
          simdgroup_barrier(mem_flags::mem_none);
          float factor[1] = {fast::exp2(max_score - new_max[0])};
          sum_score = sum_score * factor[0] + tile_sum[0];
          Otile.mul_by(factor);
        } else {
          sum_score = sum_score + tile_sum[0];
        }
        max_score = new_max[0];

        STEEL_PRAGMA_UNROLL
        for (short id = 0; id < TD; id++) {
          STEEL_PRAGMA_UNROLL
          for (short ik = 0; ik < TK; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            Vtile.load(&smem[V_OFF + Vs_offset + ik * 8 * LDV + id * 8]);
            simdgroup_barrier(mem_flags::mem_none);
            mma_fragment(Otile.frag_at(id), Stile.frag_at(ik), Vtile.frag_at(0), Otile.frag_at(id));
          }
        }
      }
    }
  }

  Otile.div_by(&sum_score);
  if (row_valid) {
    Otile.store(out + base + size_t(out_token) * LDT + sn);
  }

// The wrapper's header redefines `kernel` to cap threads per threadgroup;
// MLX's explicit instantiation that follows must not carry the attribute.
#undef kernel
