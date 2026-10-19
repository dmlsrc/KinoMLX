// Sol-Attn preparation body fragment consumed by mx.fast.metal_kernel.
// MLX supplies the kernel wrapper and input/output pointer declarations.
//
// One thread per (8 columns, 64-token block, tensor) of q, k or v. It reads
// the block's rows in routing order (routing row t is input row rows[t]),
// writes the kernel-dtype copy, and sums each column in FP32 in row order for
// the block mean. Key and value means are rounded to the input dtype before
// the kernel dtype, so routing is the input dtype's whichever kernel dtype
// runs. Reads coalesce across a row's columns.

  using namespace sol;
  constexpr int LDT = H * D;
  const int c0 = int(thread_position_in_grid.x) * 8;
  const int b = int(thread_position_in_grid.y);
  const uint w = thread_position_in_grid.z;
  if (c0 >= LDT) {
    return;
  }
  const device I* src = w == 0 ? q : (w == 1 ? k : v);
  device T* dst = w == 0 ? qo : (w == 1 ? ko : vo);
  const int start = b * 64;
  const int len = min(64, TOKENS - start);
  float s[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  for (int r = 0; r < len; r++) {
    const int t = start + r;
    const device I* p = src + int(rows[t]) * LDT + c0;
    const vec<I, 4> lo = *reinterpret_cast<const device vec<I, 4>*>(p);
    const vec<I, 4> hi = *reinterpret_cast<const device vec<I, 4>*>(p + 4);
    for (int i = 0; i < 4; i++) {
      s[i] += static_cast<float>(lo[i]);
      s[i + 4] += static_cast<float>(hi[i]);
    }
    vec<T, 4> lo_out;
    vec<T, 4> hi_out;
    for (int i = 0; i < 4; i++) {
      lo_out[i] = sol_convert<T>(lo[i]);
      hi_out[i] = sol_convert<T>(hi[i]);
    }
    device T* d = dst + t * LDT + c0;
    *reinterpret_cast<device vec<T, 4>*>(d) = lo_out;
    *reinterpret_cast<device vec<T, 4>*>(d + 4) = hi_out;
  }
  const int o = ((c0 / D) * NB + b) * D + (c0 % D);
  for (int i = 0; i < 8; i++) {
    const float mean = s[i] / float(len);
    if (w == 0) {
      qmean[o + i] = mean;
    } else if (w == 1) {
      kmean[o + i] = sol_convert<T>(static_cast<I>(mean));
    } else {
      vmean[o + i] = sol_convert<T>(static_cast<I>(mean));
    }
  }
