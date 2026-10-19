// Exact 3D neighborhood attention over 4 x 4 key patches, a body fragment for
// mx.fast.metal_kernel on the FP16 STEEL attention primitives of
// fp16_attention_header.metal (derived from Apple MLX STEEL attention sources,
// Copyright (c) 2024-25 Apple Inc., SPDX-License-Identifier: MIT). Inputs, the
// window rule and the masking are those of neighborhood_attention_runs.metal.
//
// Each SIMD group owns 8 queries, two frames by a 2 x 2 patch: fragment row r is
// frame r / 4, row (r / 2) % 2, column r % 2. A threadgroup stacks four SIMD
// groups along time, an 8 x 2 x 2 block. The spatial union of a 2 x 2 patch's
// windows is (KH + 1) x (KW + 1), 12 x 12 for the 11 x 11 window, which 4 x 4 key
// patches tile exactly: RH x RW runs of 16 keys per plane, with no key that every
// query of the run misses, where a 16-key run along W wastes 5 of every query's
// 16 columns. The cost is a W mask that changes per run; the plane test is per
// row and run, and a SIMD group skips the planes neither of its frames sees
// (exact: every score it would compute is masked).

  using namespace sol;
  constexpr int LDT = HEADS * D;
  constexpr short LDQ = D + 8;
  constexpr short LDK = 16 + 8;
  constexpr short LDV = D + 8;
  constexpr int K_OFF = 0;
  constexpr int V_OFF = D * LDK;
  constexpr int KV_SIZE = V_OFF + 16 * LDV;
  constexpr int SMEM = 32 * LDQ > KV_SIZE ? 32 * LDQ : KV_SIZE;
  constexpr short TK = 2;
  constexpr short TD = D / 8;
  constexpr int QT = 8;
  constexpr int RH = (KH + 1 + 3) / 4;
  constexpr int RW = (KW + 1 + 3) / 4;
  constexpr int UT = QT + KT - 1;
  constexpr int TILES_W = (GW + 1) / 2;
  constexpr int TILES_H = (GH + 1) / 2;
  constexpr int TOKENS = GT * GH * GW;
  static_assert(D == 64, "64-wide heads");
  static_assert(GT >= UT && GH >= 4 * RH && GW >= 4 * RW, "the grid holds the key box");
  static_assert(RH * RW <= 16, "the run masks fit two words");
  static_assert(KT % 2 == 1 && KH % 2 == 1 && KW % 2 == 1, "odd windows");

  const ushort lane = ushort(thread_index_in_simdgroup);
  const ushort sg = ushort(simdgroup_index_in_threadgroup);
  const ushort tid = ushort(sg * 32 + lane);
  const int tile = int(threadgroup_position_in_grid.x);
  const int head = int(threadgroup_position_in_grid.y);
  const int batch = int(threadgroup_position_in_grid.z);
  const int t0 = (tile / (TILES_W * TILES_H)) * QT;
  const int h0 = ((tile / TILES_W) % TILES_H) * 2;
  const int w0 = (tile % TILES_W) * 2;
  // A 64-bit base once per threadgroup and a 64-bit run address once per run; the offsets
  // inside a run stay 32-bit, so the volume and the key box may exceed 2^31 elements.
  const size_t base = size_t(batch) * size_t(TOKENS) * LDT + size_t(head) * D;

  threadgroup T smem[SMEM];

  // Stage Q: two threads per row, so the first 64 threads stage the 32 rows; threadgroup
  // row r is SIMD group r / 8's fragment row r % 8. The other threads must not write: rows
  // past 32 would land outside this threadgroup's memory.
  {
    const short r = short(tid / 2);
    const short c0 = short((tid % 2) * (D / 2));
    if (r < 32) {
      const int fr = r % 8;
      const int t = min(t0 + 2 * (r / 8) + fr / 4, GT - 1);
      const int h = min(h0 + (fr / 2) % 2, GH - 1);
      const int w = min(w0 + fr % 2, GW - 1);
      const device T* src = q + base + size_t((t * GH + h) * GW + w) * LDT + c0;
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < D / 2; j += 4) {
        *reinterpret_cast<threadgroup vec<T, 4>*>(&smem[r * LDQ + c0 + j]) =
            *reinterpret_cast<const device vec<T, 4>*>(src + j);
      }
    }
  }

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

  // The key box: rows hu0 .. hu0 + 4 RH and columns wu0 .. wu0 + 4 RW of the planes below.
  const int hu0 = min(clamp(h0 - KH / 2, 0, GH - KH), GH - 4 * RH);
  const int wu0 = min(clamp(w0 - KW / 2, 0, GW - KW), GW - 4 * RW);

  // This thread's fragment row and its window starts.
  const int qt = t0 + 2 * int(sg) + sm / 4;
  const int qh = h0 + (sm / 2) % 2;
  const int qw = w0 + sm % 2;
  const bool row_valid = qt < GT && qh < GH && qw < GW;
  const int t_c = min(qt, GT - 1);
  const int h_c = min(qh, GH - 1);
  const int w_c = min(qw, GW - 1);
  const int out_token = (t_c * GH + h_c) * GW + w_c;
  const int ts = clamp(t_c - KT / 2, 0, GT - KT);
  // Column (j, jj) of a run is key 8 j + sn + jj of its 4 x 4 patch: row
  // 2 j + (sn + jj) / 4, column (sn + jj) % 4. Its offsets from this row's window
  // start, before adding the run's origin 4 rh, 4 rw.
  const int hs = clamp(h_c - KH / 2, 0, GH - KH);
  const int ws = clamp(w_c - KW / 2, 0, GW - KW);
  // Which of the thread's four columns this row sees in each run of a plane: the
  // pattern is the same on every plane, so it is decided once, as four bits per
  // run (runs 0 to 7 in the low word, 8 to 15 in the high one).
  uint vis_lo = 0;
  uint vis_hi = 0;
  for (int rh = 0; rh < RH; rh++) {
    for (int rw = 0; rw < RW; rw++) {
      const int run = rh * RW + rw;
      uint bits = 0;
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < TK; j++) {
        STEEL_PRAGMA_UNROLL
        for (short jj = 0; jj < 2; jj++) {
          const int dh = hu0 + 4 * rh + 2 * j + (sn + jj) / 4 - hs;
          const int dw = wu0 + 4 * rw + (sn + jj) % 4 - ws;
          const bool seen = uint(dh) < uint(KH) && uint(dw) < uint(KW);
          bits |= uint(seen) << (2 * j + jj);
        }
      }
      if (run < 8) {
        vis_lo |= bits << (4 * run);
      } else {
        vis_hi |= bits << (4 * (run - 8));
      }
    }
  }

  // Loader offsets: K lanes take key lane % 16 of the patch, V SIMD group s the
  // patch's row s (keys 4 s .. 4 s + 3, consecutive tokens).
  const int k_key = int(lane % 16);
  const int k_d0 = 16 * int(sg) + 4 * int(lane / 16);
  const int k_lane = ((k_key / 4) * GW + k_key % 4) * LDT + k_d0;
  // Staging stores go through threadgroup pointers made once: indexing the array
  // with a run-invariant offset plus a constant made the compiler rebuild all
  // eight K store addresses every run (about 30 integer adds).
  threadgroup T* k_store = &smem[K_OFF + k_key + k_d0 * LDK];
  // V: SIMD group s stages its patch row s, keys 4 s .. 4 s + 3 (consecutive tokens);
  // half-warp l / 16 takes keys 4 s + l / 16 and 4 s + 2 + l / 16, four elements a lane, so
  // each half-warp writes one whole padded row.
  const int v_lane = (int(sg) * GW + int(lane / 16)) * LDT + 4 * int(lane % 16);
  threadgroup T* v_store =
      &smem[V_OFF + (4 * int(sg) + int(lane / 16)) * LDV + 4 * int(lane % 16)];

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

  // The first key of the current run, as a 32-bit token index inside the key box advanced
  // in place: 4 per run along W, a row of patches per run along H, the rest per plane. The
  // run's address is formed from it in 64 bits, so the box may exceed 2^31 elements.
  // Only the planes some valid row of this block sees: frames t0 .. last are valid, and
  // their windows span ts(t0) .. ts(last) + KT. A row past the grid sees nothing, so a SIMD
  // group whose rows all lie past it skips its products (those rows are never stored).
  const int t_last = min(t0 + QT - 1, GT - 1);
  const int plane_lo = clamp(t0 - KT / 2, 0, GT - KT);
  const int plane_hi = clamp(t_last - KT / 2, 0, GT - KT) + KT;
  const device T* kbox = k + base + size_t(plane_lo) * GH * GW * LDT;
  const device T* vbox = v + base + size_t(plane_lo) * GH * GW * LDT;
  int run_token = hu0 * GW + wu0;
  for (int t = plane_lo; t < plane_hi; t++) {
    const bool tvis = row_valid && uint(t - ts) < uint(KT);
    const bool plane_needed = simd_any(tvis);
    uint pending_lo = tvis ? vis_lo : 0u;
    uint pending_hi = tvis ? vis_hi : 0u;
    for (int rh = 0; rh < RH; rh++) {
      for (int rw = 0; rw < RW; rw++) {
        const uint bits = pending_lo & 0xfu;
        pending_lo = (pending_lo >> 4) | (pending_hi << 28);
        pending_hi >>= 4;
        vec<T, 4> k_regs[2];
        vec<T, 4> v_regs[2];
        const device T* k_run = kbox + size_t(run_token) * LDT;
        const device T* v_run = vbox + size_t(run_token) * LDT;
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 2; i++) {
          k_regs[i] = *reinterpret_cast<const device vec<T, 4>*>(k_run + k_lane + 8 * i);
        }
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 2; i++) {
          v_regs[i] = *reinterpret_cast<const device vec<T, 4>*>(v_run + v_lane + 2 * i * LDT);
        }
        run_token += 4;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 2; i++) {
          STEEL_PRAGMA_UNROLL
          for (short e = 0; e < 4; e++) {
            k_store[(8 * i + e) * LDK] = k_regs[i][e];
          }
        }
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 2; i++) {
          *reinterpret_cast<threadgroup vec<T, 4>*>(v_store + 2 * i * LDV) = v_regs[i];
        }
        Stile.clear();
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (plane_needed) {
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
              const bool visible = (bits & (1u << (2 * j + jj))) != 0u;
              const float s = Stile.frag_at(j)[jj] * scale;
              Stile.frag_at(j)[jj] = visible ? s : neg_inf;
            }
          }
          // One maximum over both fragments in the thread, then the row's two shuffles:
          // the same value as RowTile::row_max, with half the shuffles.
          float thread_max = max_score;
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < TK; j++) {
            thread_max = metal::max(thread_max, metal::max(Stile.frag_at(j)[0], Stile.frag_at(j)[1]));
          }
          thread_max = metal::max(thread_max, simd_shuffle_xor(thread_max, ushort(1)));
          thread_max = metal::max(thread_max, simd_shuffle_xor(thread_max, ushort(8)));
          float new_max[1] = {thread_max};
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
      run_token += 4 * GW - 4 * RW;
    }
    run_token += GH * GW - 4 * RH * GW;
  }

  Otile.div_by(&sum_score);
  if (row_valid) {
    Otile.store(out + base + size_t(out_token) * LDT + sn);
  }

// The wrapper's header redefines `kernel` to cap threads per threadgroup;
// MLX's explicit instantiation that follows must not carry the attribute.
#undef kernel
