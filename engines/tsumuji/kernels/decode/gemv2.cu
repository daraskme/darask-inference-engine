// M2 decode GEMV: y = W x, batch 1, memory-bound. W is bf16 [O][K] or MXFP8 (e4m3 [O][K] plus
// ue8m0 scales [O][K/32]; see quant.cu), row-major.
//
// A warp owns R consecutive rows over one K slice (grid.y = S slices). Lanes stride the slice in
// 16-byte vectors; the first P vectors per row (and their scales) are loaded into registers before
// `griddepcontrol.wait` (weights never depend on the previous kernel), so with PDL the weight
// stream starts while the previous kernel finishes. With S > 1 the warp writes a partial sum to
// y[s][row]; the consumer adds the S partials in slice order (deterministic), or `gemv_reduce`.
//
// The input slice is staged in shared memory, either as given or as the per-stream (1 + w)
// RMSNorm of the residual (the GR read-out), whose sum-of-squares partials the kernel that last
// wrote the residual left in `ss` [n][hc].
#include "decode.cuh"

struct Stage {
  const float* x;   // input (plain) or residual R [HC][H] (norm)
  const float* ss;  // norm: sum-of-squares partials [n][hc]
  const float* w;   // norm: RMSNorm weights [HC * H]
  int n, h, hc;
  float eps;
};

#ifndef TSJ_GEMV_ROW_REUSE
#define TSJ_GEMV_ROW_REUSE 1
#endif

// One pair of a vector, with exactly Fmt::dq's conversion and scale arithmetic. Keeping only
// two values per row live lets gemv_rows share x across R rows without R whole decoded vectors.
template <int FMT, int J>
TSJ_DEVICE float2 gemv_pair(uint4 code, float scale) {
  if constexpr (FMT == 0) {
    const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&code);
    return __bfloat1622float2(p[J / 2]);
  } else if constexpr (FMT == 1) {
    const __nv_fp8x2_storage_t* p = reinterpret_cast<const __nv_fp8x2_storage_t*>(&code);
    __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(p[J / 2], __NV_E4M3);
    const float2 f = __half22float2(*reinterpret_cast<__half2*>(&h));
    return {f.x * scale, f.y * scale};
  } else {
    const uint32_t words[4] = {code.x, code.y, code.z, code.w};
    const uint32_t word = words[J / 4] ^ 0x80808080u;
    const float bias = -8388736.f * scale;
    return {fmaf(__int_as_float(__byte_perm(word, 0x4B000000u, 0x7540 + J % 4)), scale, bias),
            fmaf(__int_as_float(__byte_perm(word, 0x4B000000u, 0x7541 + J % 4)), scale, bias)};
  }
}

template <int FMT, int R, int NT, int J = 0>
TSJ_DEVICE void gemv_rows(const uint4 (&code)[R], const float (&scale)[R], const float* xs,
                          int len, int v, int N, int row0, int O, float (&acc)[R][NT]) {
  constexpr int VW = Fmt<FMT>::VW;
  float2 f[R] = {};
#pragma unroll
  for (int r = 0; r < R; ++r)
    if (row0 + r < O) f[r] = gemv_pair<FMT, J>(code[r], scale[r]);
#pragma unroll
  for (int n = 0; n < NT; ++n) {
    if (n >= N) break;
    const float a = xs[n * len + xvec<VW>(v, J)], b = xs[n * len + xvec<VW>(v, J + 1)];
#pragma unroll
    for (int r = 0; r < R; ++r) {
      acc[r][n] = fmaf(f[r].x, a, acc[r][n]);
      acc[r][n] = fmaf(f[r].y, b, acc[r][n]);
    }
  }
  if constexpr (J + 2 < VW) gemv_rows<FMT, R, NT, J + 2>(code, scale, xs, len, v, N, row0, O, acc);
}

template <int VW, int NT, int R>
TSJ_DEVICE void gemv_fma(const float (&f)[VW], const float* xs, int len, int v, int N, float (&acc)[NT]) {
  if constexpr (R == 1) {
    // Interleave the independent token chains; release each weight after all tokens use it.
#pragma unroll
    for (int j = 0; j < VW; ++j) {
      const int at = xvec<VW>(v, j);
#pragma unroll
      for (int n = 0; n < NT; ++n) acc[n] = fmaf(f[j], xs[n * len + at], acc[n]);
    }
  } else {
    // Several rows already supply independent chains. Bound the token-major loop by N to
    // keep ptxas from hoisting every column's shared loads across the prefetched rows.
#pragma unroll
    for (int n = 0; n < NT; ++n) {
      if constexpr (NT > 1)
        if (n >= N) break;
#pragma unroll
      for (int j = 0; j < VW; ++j) acc[n] = fmaf(f[j], xs[n * len + xvec<VW>(v, j)], acc[n]);
    }
  }
}

// Windows (NT > 1) stage the input slice with a 16-byte swizzle: vector v's 4-float chunk c lives
// at v * VW + 4 * (c ^ g(v)), g chosen so the 8 lanes of a 16-byte shared-load phase hit 8
// different bank groups. A lane reads a chunk with one 16-byte load instead of four 4-byte ones;
// the values and every FMA order are those of the scalar xvec layout (bit-identical results).
template <int VW>
TSJ_DEVICE int xchunk(int v, int c) {
  static_assert(VW == 16 || VW == 8);
  const int g = VW == 16 ? (v >> 1) & 3 : (v >> 2) & 1;
  return v * VW + 4 * (c ^ g);
}
template <int VW>
TSJ_DEVICE int xelem(int i) {
  return xchunk<VW>(i / VW, (i % VW) >> 2) + (i & 3);
}
TSJ_DEVICE float f4at(const float4& a, int j) { return j == 0 ? a.x : j == 1 ? a.y : j == 2 ? a.z : a.w; }

// gemv_fma for the 16-byte layout: acc[n] += f[j] x[n][j], j in order, for each token.
template <int VW, int NT>
TSJ_DEVICE void gemv_fma4(const float (&f)[VW], const float* xs, int len, int v, int N, float (&acc)[NT]) {
#pragma unroll
  for (int c = 0; c < VW / 4; ++c) {
    float4 x[NT];
#pragma unroll
    for (int n = 0; n < NT; ++n)
      x[n] = (NT == 1 || n < N) ? *reinterpret_cast<const float4*>(xs + n * len + xchunk<VW>(v, c)) : make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int n = 0; n < NT; ++n) acc[n] = fmaf(f[4 * c + j], f4at(x[n], j), acc[n]);
  }
}

// gemv_pair with a J that is constant after unrolling.
template <int FMT>
TSJ_DEVICE float2 gemv_pair_at(uint4 code, float scale, int J) {
  switch (J) {
    case 0: return gemv_pair<FMT, 0>(code, scale);
    case 2: return gemv_pair<FMT, 2>(code, scale);
    case 4: return gemv_pair<FMT, 4>(code, scale);
    case 6: return gemv_pair<FMT, 6>(code, scale);
    case 8: return gemv_pair<FMT, 8>(code, scale);
    case 10: return gemv_pair<FMT, 10>(code, scale);
    case 12: return gemv_pair<FMT, 12>(code, scale);
    default: return gemv_pair<FMT, 14>(code, scale);
  }
}

// gemv_rows for the 16-byte layout: each chunk of every token is loaded once for all R rows.
template <int FMT, int R, int NT>
TSJ_DEVICE void gemv_rows4(const uint4 (&code)[R], const float (&scale)[R], const float* xs, int len, int v, int N,
                           int row0, int O, float (&acc)[R][NT]) {
  constexpr int VW = Fmt<FMT>::VW;
#pragma unroll
  for (int c = 0; c < VW / 4; ++c) {
    float4 x[NT];
#pragma unroll
    for (int n = 0; n < NT; ++n)
      x[n] = n < N ? *reinterpret_cast<const float4*>(xs + n * len + xchunk<VW>(v, c)) : make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      float2 f[R] = {};
#pragma unroll
      for (int r = 0; r < R; ++r)
        if (row0 + r < O) f[r] = gemv_pair_at<FMT>(code[r], scale[r], 4 * c + 2 * h);
#pragma unroll
      for (int n = 0; n < NT; ++n) {
        if (n >= N) break;
#pragma unroll
        for (int r = 0; r < R; ++r) {
          acc[r][n] = fmaf(f[r].x, f4at(x[n], 2 * h), acc[r][n]);
          acc[r][n] = fmaf(f[r].y, f4at(x[n], 2 * h + 1), acc[r][n]);
        }
      }
    }
  }
}

// N tokens share one pass over the weights: x is [N][K] (the residual [N][HC*H] when NORM, with
// stats ss [N][n][hc]), y is [S][N][O]. NT is the compiled token capacity (1, 2, 4, 8; N <= NT);
// each token's accumulation order is the same for every NT, so a token's result does not depend
// on how many tokens share the pass.
template <int FMT, int R, int P, bool NORM, int NT>
__device__ __forceinline__ void gemv_impl(const void* __restrict__ Wv, const uint8_t* __restrict__ SC, const Stage st,
                                          float* __restrict__ y, int O, int K, int N) {
  using F = Fmt<FMT>;
  constexpr int VW = F::VW;
  extern __shared__ float xs[];  // [NT][slice]
  __shared__ float rs[NT][8];
  const int S = gridDim.y, s = blockIdx.y;
  const int kv = K / VW;             // 16-byte vectors per row
  const int per = (kv + S - 1) / S;  // vectors per slice
  const int v0 = s * per, v1 = min(kv, v0 + per);
  const int len = (v1 - v0) * VW;    // slice length in elements
  const int lane = threadIdx.x & 31;
  const int row0 = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * R;
  const uint4* w = reinterpret_cast<const uint4*>(Wv);
  auto scale = [&](int row, int v) { return FMT != 0 ? ue8m0(__ldcs(SC + size_t(row) * (K / 32) + v * VW / 32)) : 1.f; };

  uint4 pre[P][R];
  float psc[P][R];
#pragma unroll
  for (int i = 0; i < P; ++i) {
    const int v = v0 + lane + 32 * i;
#pragma unroll
    for (int r = 0; r < R; ++r)
      if (v < v1 && row0 + r < O) {
        pre[i][r] = __ldcs(w + size_t(row0 + r) * kv + v);
        psc[i][r] = scale(row0 + r, v);
      }
  }
  griddep_wait();
  griddep_launch_dependents();
  if constexpr (NORM) {
    if (threadIdx.x < NT * st.hc) {
      const int n = threadIdx.x / st.hc, c = threadIdx.x % st.hc;
      float a = 0.f;
      if (n < N)
        for (int p = 0; p < st.n; ++p) a += st.ss[(n * st.n + p) * st.hc + c];
      rs[n][c] = rsqrtf(a / float(st.h) + st.eps);
    }
    __syncthreads();
  }
  // All tokens' loads of an element are issued together (and two elements per round), so a window
  // waits for about len / 512 L2 round trips instead of NT times that. Same values as before.
#pragma unroll 2
  for (int i = threadIdx.x; i < len; i += blockDim.x) {
    const int k = v0 * VW + i, at = xvec<VW>(i / VW, i % VW);
    float x[NT];
#pragma unroll
    for (int n = 0; n < NT; ++n) x[n] = n < N ? st.x[size_t(n) * K + k] : 0.f;
    if constexpr (NORM) {
      const int c = k / st.h;
      const float g = 1.f + st.w[k];
#pragma unroll
      for (int n = 0; n < NT; ++n)
        if (n < N) x[n] = x[n] * rs[n][c] * g;
    }
#pragma unroll
    for (int n = 0; n < NT; ++n) xs[n * len + at] = x[n];
  }
  __syncthreads();
  if (row0 >= O) return;

  float acc[R][NT] = {};
#pragma unroll
  for (int i = 0; i < P; ++i) {
    const int v = v0 + lane + 32 * i;
    if (v < v1) {
      if constexpr (TSJ_GEMV_ROW_REUSE && NT > 1 && R > 1) {
        // Each input pair serves all rows before moving on. Per-row/token FMA order stays
        // j=0..VW-1, then the next vector; R, P, S and the lane -> K map are unchanged.
        gemv_rows<FMT, R, NT>(pre[i], psc[i], xs, len, v - v0, N, row0, O, acc);
      } else {
#pragma unroll
        for (int r = 0; r < R; ++r) {
          if (row0 + r < O) {
            float f[VW];
            F::dq(pre[i][r], psc[i][r], f);
            gemv_fma<VW, NT, R>(f, xs, len, v - v0, N, acc[r]);
          }
        }
      }
    }
  }
  constexpr int TAIL_UNROLL = TSJ_GEMV_ROW_REUSE && NT > 1 && R > 1 ? 1 : 4;
#pragma unroll TAIL_UNROLL
  for (int v = v0 + lane + 32 * P; v < v1; v += 32) {
    if constexpr (TSJ_GEMV_ROW_REUSE && NT > 1 && R > 1) {
      uint4 code[R];
      float sc[R];
#pragma unroll
      for (int r = 0; r < R; ++r)
        if (row0 + r < O) {
          code[r] = __ldcs(w + size_t(row0 + r) * kv + v);
          sc[r] = scale(row0 + r, v);
        }
      gemv_rows<FMT, R, NT>(code, sc, xs, len, v - v0, N, row0, O, acc);
    } else {
#pragma unroll
      for (int r = 0; r < R; ++r) {
        if (row0 + r < O) {
          float f[VW];
          F::dq(__ldcs(w + size_t(row0 + r) * kv + v), scale(row0 + r, v), f);
          gemv_fma<VW, NT, R>(f, xs, len, v - v0, N, acc[r]);
        }
      }
    }
  }
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const float a = warp_sum_scatter<NT>(acc[r]);
    if (lane < N && lane < NT && row0 + r < O)
      y[(size_t(s) * N + lane) * O + row0 + r] = a;
  }
}

// gemv_impl for windows (NT > 1): the same weights, lanes, slices and per-token FMA order (so a
// token's result is the same for every NT), with cheaper staging. The slice is staged in batches of
// B elements per thread whose loads are all in flight before the first store; the first batch is
// loaded before the norm statistics are reduced (one L2 round trip instead of two), the norm's
// stream index is computed once per element instead of once per token, and the slice uses the
// 16-byte layout. B = 5 covers the GR down slice (1,280 elements) in one batch.
template <int FMT, int R, int P, bool NORM, int NT>
__device__ __forceinline__ void gemv_win(const void* __restrict__ Wv, const uint8_t* __restrict__ SC, const Stage st,
                                         float* __restrict__ y, int O, int K, int N) {
  using F = Fmt<FMT>;
  constexpr int VW = F::VW, B = NORM ? 5 : 4;
  extern __shared__ float xs[];  // [NT][slice], 16-byte layout
  __shared__ float rs[NT][8];
  const int S = gridDim.y, s = blockIdx.y;
  const int kv = K / VW;
  const int per = (kv + S - 1) / S;
  const int v0 = s * per, v1 = min(kv, v0 + per);
  const int len = (v1 - v0) * VW;
  const int lane = threadIdx.x & 31;
  const int row0 = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * R;
  const uint4* w = reinterpret_cast<const uint4*>(Wv);
  auto scale = [&](int row, int v) { return FMT != 0 ? ue8m0(__ldcs(SC + size_t(row) * (K / 32) + v * VW / 32)) : 1.f; };

  uint4 pre[P][R];
  float psc[P][R];
#pragma unroll
  for (int i = 0; i < P; ++i) {
    const int v = v0 + lane + 32 * i;
#pragma unroll
    for (int r = 0; r < R; ++r)
      if (v < v1 && row0 + r < O) {
        pre[i][r] = __ldcs(w + size_t(row0 + r) * kv + v);
        psc[i][r] = scale(row0 + r, v);
      }
  }
  // The norm weights of the first batch do not depend on earlier kernels.
  float nw[B];
  int cs[B];
  if constexpr (NORM) {
#pragma unroll
    for (int b = 0; b < B; ++b) {
      const int i = threadIdx.x + b * blockDim.x;
      nw[b] = i < len ? 1.f + st.w[v0 * VW + i] : 0.f;
      cs[b] = (v0 * VW + i) / st.h;
    }
  }
  griddep_wait();
  griddep_launch_dependents();
  float val[NT][B];
  auto load = [&](int i0) {
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
      for (int b = 0; b < B; ++b) {
        const int i = i0 + threadIdx.x + b * blockDim.x;
        val[n][b] = (n < N && i < len) ? st.x[size_t(n) * K + v0 * VW + i] : 0.f;
      }
  };
  load(0);
  if constexpr (NORM) {
    if (threadIdx.x < NT * st.hc) {
      const int n = threadIdx.x / st.hc, c = threadIdx.x % st.hc;
      constexpr int PM = 16;  // stat partials loaded together, then added in order
      float part[PM];
#pragma unroll
      for (int p = 0; p < PM; ++p) part[p] = (n < N && p < st.n) ? st.ss[(n * st.n + p) * st.hc + c] : 0.f;
      float a = 0.f;
      if (n < N) {
#pragma unroll
        for (int p = 0; p < PM; ++p)
          if (p < st.n) a += part[p];
        for (int p = PM; p < st.n; ++p) a += st.ss[(n * st.n + p) * st.hc + c];
      }
      rs[n][c] = rsqrtf(a / float(st.h) + st.eps);
    }
    __syncthreads();
  }
  for (int i0 = 0;;) {
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
      for (int b = 0; b < B; ++b) {
        const int i = i0 + threadIdx.x + b * blockDim.x;
        if (i < len) {
          float x = val[n][b];
          if constexpr (NORM) {
            const int k = v0 * VW + i;
            const int c = i0 == 0 ? cs[b] : k / st.h;
            const float wn = i0 == 0 ? nw[b] : 1.f + st.w[k];
            x = n < N ? x * rs[n][c] * wn : 0.f;
          }
          xs[n * len + xelem<VW>(i)] = x;
        }
      }
    i0 += B * blockDim.x;
    if (i0 >= len) break;
    load(i0);
  }
  __syncthreads();
  if (row0 >= O) return;

  float acc[R][NT] = {};
#pragma unroll
  for (int i = 0; i < P; ++i) {
    const int v = v0 + lane + 32 * i;
    if (v < v1) {
      if constexpr (R > 1) {
        gemv_rows4<FMT, R, NT>(pre[i], psc[i], xs, len, v - v0, N, row0, O, acc);
      } else {
        float f[VW];
        F::dq(pre[i][0], psc[i][0], f);
        gemv_fma4<VW, NT>(f, xs, len, v - v0, N, acc[0]);
      }
    }
  }
#pragma unroll 1
  for (int v = v0 + lane + 32 * P; v < v1; v += 32) {
    uint4 code[R];
    float sc[R];
#pragma unroll
    for (int r = 0; r < R; ++r)
      if (row0 + r < O) {
        code[r] = __ldcs(w + size_t(row0 + r) * kv + v);
        sc[r] = scale(row0 + r, v);
      }
    if constexpr (R > 1) {
      gemv_rows4<FMT, R, NT>(code, sc, xs, len, v - v0, N, row0, O, acc);
    } else {
      float f[VW];
      F::dq(code[0], sc[0], f);
      gemv_fma4<VW, NT>(f, xs, len, v - v0, N, acc[0]);
    }
  }
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const float a = warp_sum_scatter<NT>(acc[r]);
    if (lane < N && lane < NT && row0 + r < O)
      y[(size_t(s) * N + lane) * O + row0 + r] = a;
  }
}

// One token: gemv_impl. Windows: gemv_win (same results, cheaper staging and shared loads).
#define GEMV(NAME, FMT, R, P, NORM, NT)                                                                \
  extern "C" __global__ void __launch_bounds__(256) NAME##_r##R##_p##P##_n##NT(                         \
      const void* __restrict__ W, const uint8_t* __restrict__ SC, const Stage st, float* __restrict__ y, \
      int O, int K, int N) {                                                                          \
    if constexpr (NT == 1)                                                                            \
      gemv_impl<FMT, R, P, NORM, NT>(W, SC, st, y, O, K, N);                                          \
    else                                                                                              \
      gemv_win<FMT, R, P, NORM, NT>(W, SC, st, y, O, K, N);                                           \
  }
// The (R, P) pairs gemv_plan uses, for 1, 2, 4 and 8 tokens.
#define GEMV_NT(NAME, FMT, NORM, NT) \
  GEMV(NAME, FMT, 1, 4, NORM, NT)    \
  GEMV(NAME, FMT, 1, 8, NORM, NT)    \
  GEMV(NAME, FMT, 2, 8, NORM, NT)    \
  GEMV(NAME, FMT, 4, 2, NORM, NT)
#define GEMV_ALL(NAME, FMT, NORM) \
  GEMV_NT(NAME, FMT, NORM, 1)     \
  GEMV_NT(NAME, FMT, NORM, 2)     \
  GEMV_NT(NAME, FMT, NORM, 4)     \
  GEMV_NT(NAME, FMT, NORM, 8)
GEMV_ALL(gemv_bf16, 0, false)
GEMV_ALL(gemv_norm, 0, true)
GEMV_ALL(gemv_fp8, 1, false)
GEMV_ALL(gemv_fp8norm, 1, true)
GEMV_ALL(gemv_i8, 2, false)
GEMV_ALL(gemv_i8norm, 2, true)

// y[o] = sum_s part[s][o], s in order.
extern "C" __global__ void gemv_reduce(const float* __restrict__ part, float* __restrict__ y, int O, int S) {
  griddep_wait();
  griddep_launch_dependents();
  for (int o = blockIdx.x * blockDim.x + threadIdx.x; o < O; o += gridDim.x * blockDim.x) {
    float a = 0.f;
    for (int s = 0; s < S; ++s) a += part[size_t(s) * O + o];
    y[o] = a;
  }
}

// GR up-projection and read-out in one pass (warp per position k), for N tokens:
//   hmid[n][j] = silu(sum_s di[s][n][j] / HC)          (j < rank; di = split-K partials of down)
//   u_i        = up[i * H + k] . hmid[n]                (i < HC streams)
//   inp[n][k]  = (1 / HC) sum_i sigmoid(u_i) xn_i[k],   xn_i[k] = R[n][i H + k] r[n][i] (1 + w[i H + k])
// up is [HC * H][rank] (bf16 or MXFP8 / MXINT8), loaded once for all tokens; rank / VW <= 64
// vectors per row, HC <= 4. stats ss [N][P][HC]. grid = ceil(H / 8), 256 threads, smem NT rank floats
// (16-byte layout, see xchunk).
template <int FMT, int NT>
__device__ __forceinline__ void gr_up_mix_impl(const void* __restrict__ up, const uint8_t* __restrict__ SC,
                                               const float* __restrict__ di, int S, int ld, const float* __restrict__ R,
                                               const float* __restrict__ ss, int P, const float* __restrict__ w,
                                               float* __restrict__ inp, int H, int HC, int rank, float eps, int N) {
  using F = Fmt<FMT>;
  constexpr int VW = F::VW;
  extern __shared__ float hs[];  // [NT][rank]
  __shared__ float rs[NT][8];
  const int lane = threadIdx.x & 31;
  const int k = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  const int kk = k < H ? k : 0;  // every lane reaches the warp sums
  const int rv = rank / VW;
  const uint4* u4 = reinterpret_cast<const uint4*>(up);
  uint4 pre[4][2];
  float psc[4][2];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j)
      if (i < HC && lane + 32 * j < rv) {
        const size_t row = size_t(i * H + kk);
        pre[i][j] = __ldcs(u4 + row * rv + lane + 32 * j);
        psc[i][j] = FMT != 0 ? ue8m0(__ldcs(SC + row * (rank / 32) + (lane + 32 * j) * VW / 32)) : 1.f;
      }
  griddep_wait();
  griddep_launch_dependents();
  // Slice-major with the tokens inside: NT independent loads per slice instead of one chain per
  // token. Each token's sum keeps the slice order.
  for (int j = threadIdx.x; j < rank; j += blockDim.x) {
    float a[NT] = {};
#pragma unroll 4
    for (int s = 0; s < S; ++s)
#pragma unroll
      for (int n = 0; n < NT; ++n)
        if (n < N) a[n] += di[(s * N + n) * ld + j];
#pragma unroll
    for (int n = 0; n < NT; ++n) hs[n * rank + xelem<VW>(j)] = siluf(a[n] / float(HC));
  }
  if (threadIdx.x < NT * HC) {
    const int n = threadIdx.x / HC, c = threadIdx.x % HC;
    float a = 0.f;
    if (n < N)
      for (int p = 0; p < P; ++p) a += ss[(n * P + p) * HC + c];
    rs[n][c] = rsqrtf(a / float(H) + eps);
  }
  __syncthreads();
  float acc[4 * NT] = {};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    if (i >= HC) break;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int v = lane + 32 * j;
      if (v < rv) {
        float f[VW];
        F::dq(pre[i][j], psc[i][j], f);  // dequantize once, apply to every token
        float a[NT];
#pragma unroll
        for (int n = 0; n < NT; ++n) a[n] = acc[i * NT + n];
        gemv_fma4<VW, NT>(f, hs, rank, v, NT, a);  // 16-byte loads; same order (q, then token)
#pragma unroll
        for (int n = 0; n < NT; ++n) acc[i * NT + n] = a[n];
      }
    }
  }
  const float u = warp_sum_scatter<4 * NT>(acc);
  const int ni = lane % NT, ci = lane / NT;
  float sg = 0.f, xn = 0.f;
  if (ci < HC && ni < N) {
    const size_t c = size_t(ni) * HC * H + ci * H + kk;
    sg = sigmoidf(u);
    xn = R[c] * rs[ni][ci] * (1.f + w[ci * H + kk]);
  }
  float mix = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    if (i >= HC) break;
    // Shuffle the factors separately to preserve the original multiply-add contraction.
    const float a = __shfl_sync(0xffffffffu, sg, i * NT + ni);
    const float b = __shfl_sync(0xffffffffu, xn, i * NT + ni);
    mix += a * b;
  }
  if (lane < NT && lane < N && k < H) inp[size_t(lane) * H + k] = mix / float(HC);
}

#define GR_UP(NAME, FMT, NT)                                                                                  \
  extern "C" __global__ void __launch_bounds__(256)                                                           \
      NAME##_n##NT(const void* __restrict__ up, const uint8_t* __restrict__ SC, const float* __restrict__ di,  \
                   int S, int ld, const float* __restrict__ R, const float* __restrict__ ss, int P,            \
                   const float* __restrict__ w, float* __restrict__ inp, int H, int HC, int rank, float eps, int N) { \
    gr_up_mix_impl<FMT, NT>(up, SC, di, S, ld, R, ss, P, w, inp, H, HC, rank, eps, N);                         \
  }
#define GR_UP_ALL(NAME, FMT) GR_UP(NAME, FMT, 1) GR_UP(NAME, FMT, 2) GR_UP(NAME, FMT, 4) GR_UP(NAME, FMT, 8)
GR_UP_ALL(gr_up_mix, 0)
GR_UP_ALL(gr_up_mix_fp8, 1)
GR_UP_ALL(gr_up_mix_i8, 2)

// MTP draft head (docs/m3.md §12): the greedy token of an NVFP4 copy of lm_head, for one token.
// Only drafts use it, so its rounding changes the acceptance rate, never the output. It reads
// 0.56 bytes per weight instead of MXINT8's 1.03.
//
// W: codes [O][K/2] (e2m1, low nibble first) then e4m3 scales [O][K/16] (`so` bytes in), one fp32
// global scale *gs (weights.rs `DraftHead`, quantized like the MTP experts). 16 lanes per row, two
// rows per warp, DH_RG row pairs per warp; every weight load is issued before griddepcontrol.wait.
// Each block leaves its best (logit, row), lowest row on ties, in part[2 * block]; draft_pick
// reduces those. grid = ceil(O / (16 DH_RG)), block 256, smem (K / 32) * 33 floats.
constexpr int DH_N = 5;  // vectors of 32 codes per lane: K = 2560 in one pass
template <int DH_RG>
__device__ __forceinline__ void draft_head_impl(const uint8_t* __restrict__ W, size_t so, const float* __restrict__ gs,
                                                const float* __restrict__ x, float* __restrict__ part, int O, int K) {
  extern __shared__ float xs[];
  __shared__ float bv[8];
  __shared__ int bi[8];
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, j = lane & 15;
  const int kv = K / 32;
  const int row0 = (blockIdx.x * 8 + w) * 2 * DH_RG + (lane >> 4);  // rows row0 + 2 r
  const uint4* codes = reinterpret_cast<const uint4*>(W);
  const uint16_t* sf = reinterpret_cast<const uint16_t*>(W + so);
  uint4 c[DH_RG][DH_N];
  uint16_t s[DH_RG][DH_N];
#pragma unroll
  for (int r = 0; r < DH_RG; ++r)
#pragma unroll
    for (int i = 0; i < DH_N; ++i) {
      const int row = row0 + 2 * r, v = j + 16 * i;
      if (row < O && v < kv) {
        c[r][i] = __ldcs(codes + size_t(row) * kv + v);
        s[r][i] = __ldcs(sf + size_t(row) * kv + v);
      }
    }
  griddep_wait();
  griddep_launch_dependents();
  for (int i = threadIdx.x; i < K; i += blockDim.x) xs[x32(i / 32, i % 32)] = x[i];
  __syncthreads();
  const float g = *gs;
  float best = -INFINITY;
  int arg = 0x7fffffff;
#pragma unroll
  for (int r = 0; r < DH_RG; ++r) {
    const int row = row0 + 2 * r;
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < DH_N; ++i) {
      const int v = j + 16 * i;
      if (v >= kv) break;
      const uint32_t words[4] = {c[r][i].x, c[r][i].y, c[r][i].z, c[r][i].w};
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        float p = 0.f;
#pragma unroll
        for (int q = 0; q < 2; ++q)
#pragma unroll
          for (int b = 0; b < 4; ++b) {
            const float2 wf = __half22float2(e2m1x2_to_half2(uint8_t(words[h * 2 + q] >> (8 * b))));
            const int k = x32(v, h * 16 + q * 8 + b * 2);
            p = fmaf(wf.x, xs[k], p);
            p = fmaf(wf.y, xs[k + 1], p);
          }
        acc = fmaf(p, e4m3_to_float(uint8_t(s[r][i] >> (8 * h))), acc);
      }
    }
    if (row < O) acc += nvfp4_dot_part(codes + size_t(row) * kv, sf + size_t(row) * kv, xs, K, j + 16 * DH_N, 16);
    acc = group_sum<16>(acc) * g;
    if (row < O && acc > best) best = acc, arg = row;  // rows ascend with r: ties keep the lower
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, best, o);
    const int oi = __shfl_xor_sync(0xffffffffu, arg, o);
    if (ov > best || (ov == best && oi < arg)) best = ov, arg = oi;
  }
  if (lane == 0) bv[w] = best, bi[w] = arg;
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int i = 1; i < 8; ++i)
      if (bv[i] > best || (bv[i] == best && bi[i] < arg)) best = bv[i], arg = bi[i];
    part[2 * blockIdx.x] = best;
    reinterpret_cast<int*>(part)[2 * blockIdx.x + 1] = arg;
  }
}

// The draft: the DP_C best rows of draft_head's n block results (each block's best; lowest row on
// ties) are rescored with the main lm_head (format FMT: 0 bf16, 1 MXFP8, 2 MXINT8; one warp per
// candidate, exact dequantization, fp32 sums), and out[0] = the best of those. NVFP4 then only
// has to rank the right token among its top few; the final pick uses the main head's rounding.
// 1 block of 1024 threads.
constexpr int DP_C = 4;
template <int FMT>
__device__ __forceinline__ void draft_pick_impl(const float* __restrict__ part, int n, const uint4* __restrict__ lm,
                                                const uint8_t* __restrict__ lsc, const float* __restrict__ x, int K,
                                                int* __restrict__ out) {
  using F = Fmt<FMT>;
  constexpr int VW = F::VW;
  __shared__ float rv[32];
  __shared__ int ri[32], cand[DP_C];
  __shared__ float score[DP_C];
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
  for (int c = 0; c < DP_C; ++c) {
    float best = -INFINITY;
    int arg = 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
      const float v = part[2 * i];
      const int a = reinterpret_cast<const int*>(part)[2 * i + 1];
      bool taken = false;
      for (int e = 0; e < c; ++e) taken |= cand[e] == a;
      if (!taken && (v > best || (v == best && a < arg))) best = v, arg = a;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, best, o);
      const int oi = __shfl_xor_sync(0xffffffffu, arg, o);
      if (ov > best || (ov == best && oi < arg)) best = ov, arg = oi;
    }
    if (lane == 0) rv[w] = best, ri[w] = arg;
    __syncthreads();
    if (threadIdx.x == 0) {
      for (int i = 1; i < nw; ++i)
        if (rv[i] > best || (rv[i] == best && ri[i] < arg)) best = rv[i], arg = ri[i];
      cand[c] = arg;  // 0x7fffffff when fewer than c + 1 blocks
    }
    __syncthreads();
  }
  if (w < DP_C) {
    const int row = cand[w];
    float acc = 0.f;
    if (row != 0x7fffffff) {
      const int kv = K / VW;
      for (int v = lane; v < kv; v += 32) {
        float f[VW];
        F::dq(lm[size_t(row) * kv + v], FMT != 0 ? ue8m0(lsc[size_t(row) * (K / 32) + v * VW / 32]) : 1.f, f);
#pragma unroll
        for (int j = 0; j < VW; ++j) acc = fmaf(f[j], x[v * VW + j], acc);
      }
    }
    acc = warp_sum(acc);
    if (lane == 0) score[w] = row != 0x7fffffff ? acc : -INFINITY;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float best = score[0];
    int arg = cand[0];
    for (int c = 1; c < DP_C; ++c)
      if (score[c] > best || (score[c] == best && cand[c] < arg)) best = score[c], arg = cand[c];
    *out = arg;
  }
}

#define DRAFT_PICK(SUFFIX, FMT)                                                                                   \
  extern "C" __global__ void __launch_bounds__(1024) draft_pick##SUFFIX(                                          \
      const float* __restrict__ part, int n, const uint4* __restrict__ lm, const uint8_t* __restrict__ lsc,       \
      const float* __restrict__ x, int K, int* __restrict__ out) {                                                \
    TSJ_PDL_ENTRY;                                                                                                \
    draft_pick_impl<FMT>(part, n, lm, lsc, x, K, out);                                                            \
  }
DRAFT_PICK(_bf16, 0)
DRAFT_PICK(_fp8, 1)
DRAFT_PICK(_i8, 2)

extern "C" __global__ void __launch_bounds__(256) draft_head(const uint8_t* __restrict__ W, size_t so, const float* __restrict__ gs,
                                                             const float* __restrict__ x, float* __restrict__ part, int O, int K) {
  draft_head_impl<4>(W, so, gs, x, part, O, K);
}
extern "C" __global__ void __launch_bounds__(256, 2) draft_head_rg2(const uint8_t* __restrict__ W, size_t so, const float* __restrict__ gs,
                                                                    const float* __restrict__ x, float* __restrict__ part, int O, int K) {
  draft_head_impl<2>(W, so, gs, x, part, O, K);
}
