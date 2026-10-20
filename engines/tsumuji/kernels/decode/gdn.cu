// Gated DeltaNet decode for N tokens. Matches transformers `Qwen4ExpTextGatedDeltaNet` with a
// cache: causal conv update + SiLU, L2-normalized q/k (q also scaled by D^-1/2), beta =
// sigmoid(b), decay = exp(-exp(A_log) * softplus(a + dt_bias)), the delta rule on an FP32 state,
// and the sigmoid-gated RMSNorm before out_proj.
#include "decode.cuh"

// Most tokens a step takes (engine.rs N_MAX): the capacity of the window records.
constexpr int NMAX = 8;

// The causal conv of one channel, with the association spelled out: left to the compiler, the
// same expression contracted as fma(w.y, h1, w.x h0) in gdn_fused but fma(w.x, h0, w.y h1) in
// gdn_window, so the two kernels differed in the last bits. This is gdn_fused's order.
TSJ_DEVICE float conv4(float4 w, const float (&h)[3], float x) {
  return fmaf(w.w, x, fmaf(w.z, h[2], fmaf(w.y, h[1], __fmul_rn(w.x, h[0]))));
}

// The GDN step of N consecutive tokens for one value head h in one block (N <= 8).
//
// Delayed commit (docs/m3.md §4.1): a step does not know yet how many of its tokens will be
// committed (speculative verification decides that at the end of the step). So it leaves records of
// its window and never writes the state past it; the next step first applies the first `*napply`
// records of the previous window to the committed state, writes that back, then runs its own window.
// Records and conv histories alternate by step parity `par`: a step reads [par], writes [par ^ 1].
//   rec [2][RS], RS = NMAX C + NMAX V (2 D + 1):  in [NMAX][C] the conv inputs of the q/k/v
//   channels, then kd [NMAX][V][2][D] (normalized k, delta), dec [NMAX][V] (decay).
//
// Per block:
//   * conv + SiLU of this head's q, k (key head kh = h / (V / QK)) and v channels from the input
//     projection's split-K partials proj[s][n][ld], row [q | k | v | z | b | a]; the conv window of
//     token n is the committed history (the three inputs before the window) followed by the window's
//     own inputs. The first value head of each key head writes the q/k history and records, every
//     block its own v channels;
//   * L2 norm of q and k (q also scaled by D^-1/2), beta = sigmoid(b), decay;
//   * the delta rule, token after token, on the D x D state held in registers (warp (wr, wc): key
//     rows 32 wr.., value columns 32 wc.., lane = column: coalesced 128-byte lines);
//   * y[n][h] = o * rsqrt(mean o^2 + eps) * w * sigmoid(z).
// A record is applied with exactly the operations its token's own step ran (s *= decay, then
// s += k delta), so every token sees the same state bits as in single-token decoding.
//
// Window latency (docs/m3.md §14): the split-K partials of every token are loaded together, the
// q/k norms and the output norms of all tokens share one pair of barriers each, and the token loop
// keeps 3 barriers per token: the next token's decay and k.S run in the same pass over the state as
// this token's update and q.S (two separate dot chains, each in the original order), and the
// output partials have their own buffer. Same bits as gdn_fused_ref.
// grid = V, blockDim = 4 D (D <= 128), smem: (4 N + 1) D + 2 (D / 32) D + 2 NMAX D floats.
extern "C" __global__ void __launch_bounds__(512) gdn_fused(
    const float* __restrict__ proj, int S, int ld, float* __restrict__ conv_state, const int* __restrict__ par,
    const int* __restrict__ napply, float* __restrict__ rec, const float* __restrict__ conv_w,
    const float* __restrict__ a_log, const float* __restrict__ dt_bias, float* __restrict__ state,
    const float* __restrict__ norm_w, float* __restrict__ y, int QK, int V, int D, float eps, int N) {
  TSJ_PDL_ENTRY;
  extern __shared__ float sm[];
  const int nw = D / 32;
  float* vs = sm;             // [N][D] v after conv; after token n's delta, its output o
  float* qs = vs + N * D;     // [N][D] normalized, scaled q
  float* ks = qs + N * D;     // [N][D] normalized k
  float* zs = ks + N * D;     // [N][D]
  float* os = zs + N * D;     // [D] delta
  float* part = os + D;       // [D / 32][D] k.S column partials
  float* part2 = part + nw * D;  // [D / 32][D] q.S column partials
  float* rk = part2 + nw * D;    // [A][2][D] the previous window's (k, delta) records to apply
  __shared__ float rdec[NMAX];    // and their decays
  __shared__ float red[NMAX][16];
  __shared__ float bd[NMAX][2];  // beta, decay per token
  const int h = blockIdx.x, t = threadIdx.x, lane = t & 31, w = t >> 5;
  const int kh = h / (V / QK), C = (2 * QK + V) * D;
  const int parity = *par, A = *napply;
  const float* old = conv_state + size_t(parity) * 3 * C;
  float* neu = conv_state + size_t(parity ^ 1) * 3 * C;
  const size_t RS = size_t(NMAX) * C + size_t(NMAX) * V * (2 * D + 1);
  const float* in_rd = rec + parity * RS;
  const float* kd_rd = in_rd + size_t(NMAX) * C;
  const float* dec_rd = kd_rd + size_t(NMAX) * V * 2 * D;
  float* in_wr = rec + (parity ^ 1) * RS;
  float* kd_wr = in_wr + size_t(NMAX) * C;
  float* dec_wr = kd_wr + size_t(NMAX) * V * 2 * D;
  const bool qk_writer = h % (V / QK) == 0;
  auto psum = [&](int n, int i) {
    float a = 0.f;
    for (int s = 0; s < S; ++s) a += proj[(s * N + n) * ld + i];
    return a;
  };
  // Every token's sum over the S slices of row i, slice-major: all tokens' loads of a slice are
  // independent (each token still adds its slices in order from 0).
  auto psum_all = [&](int i, float (&a)[NMAX]) {
#pragma unroll
    for (int n = 0; n < NMAX; ++n) a[n] = 0.f;
    for (int s = 0; s < S; ++s)
#pragma unroll
      for (int n = 0; n < NMAX; ++n)
        if (n < N) a[n] += proj[(s * N + n) * ld + i];
  };
  // Thread groups of D: [0, D) q, [D, 2D) k, [2D, 3D) v, [3D, 4D) z.
  const int g = t / D, j = t % D;
  const int c = g == 0 ? kh * D + j : g == 1 ? (QK + kh) * D + j : g == 2 ? 2 * QK * D + h * D + j : C + h * D + j;
  float conv[NMAX];  // this thread's conv output per token (q, k, v groups)
  float xin[NMAX];
  psum_all(c, xin);
  // Stage the records the state commits first: one round trip for all of them (they were read
  // one record after the other in the state loop).
  for (int i = t; i < A * 2 * D; i += blockDim.x) rk[i] = kd_rd[(size_t(i / (2 * D)) * V + h) * 2 * D + i % (2 * D)];
  if (t < A) rdec[t] = dec_rd[t * V + h];
  if (g < 3) {
    const bool writer = g == 2 || qk_writer;
    const float4 wc = reinterpret_cast<const float4*>(conv_w)[c];
    float hist[3] = {old[c], old[C + c], old[2 * C + c]};  // inputs at -3, -2, -1
#pragma unroll
    for (int e = 0; e < NMAX; ++e) {  // commit the previous window's accepted inputs
      if (e >= A) break;
      hist[0] = hist[1];
      hist[1] = hist[2];
      hist[2] = in_rd[e * C + c];
    }
    if (writer) {
      neu[c] = hist[0];
      neu[C + c] = hist[1];
      neu[2 * C + c] = hist[2];
    }
#pragma unroll
    for (int n = 0; n < NMAX; ++n) {
      if (n >= N) break;
      const float x = xin[n];
      if (writer) in_wr[n * C + c] = x;
      conv[n] = siluf(conv4(wc, hist, x));
      hist[0] = hist[1];
      hist[1] = hist[2];
      hist[2] = x;
    }
  } else {
#pragma unroll
    for (int n = 0; n < NMAX; ++n)
      if (n < N) zs[n * D + j] = xin[n];
  }
  if (g == 2) {
#pragma unroll
    for (int n = 0; n < NMAX; ++n)
      if (n < N) vs[n * D + j] = conv[n];
  }
  if (t < N) {
    const int ab = C + V * D;
    bd[t][0] = sigmoidf(psum(t, ab + h));
    bd[t][1] = expf(-expf(a_log[h]) * softplusf(psum(t, ab + V + h) + dt_bias[h]));
    dec_wr[t * V + h] = bd[t][1];
  }
  // L2 norms of q (group 0) and k (group 1) for all tokens: warp sums of the (uncontracted)
  // squares, then the group's warps in order.
  const int wpg = D / 32;
#pragma unroll
  for (int n = 0; n < NMAX; ++n) {
    if (n >= N) break;
    const float val = g < 3 ? conv[n] : 0.f;
    const float sq = warp_sum(g < 2 ? __fmul_rn(val, val) : 0.f);
    if (lane == 0) red[n][w] = sq;
  }
  __syncthreads();
  if (g < 2) {
#pragma unroll
    for (int n = 0; n < NMAX; ++n) {
      if (n >= N) break;
      float tot = 0.f;
      for (int i = 0; i < wpg; ++i) tot += red[n][i + g * wpg];
      float r = rsqrtf(tot + 1e-6f);
      if (g == 0) r *= rsqrtf(float(D));
      const float nv = conv[n] * r;
      (g == 0 ? qs : ks)[n * D + j] = nv;
      if (g == 1) kd_wr[(size_t(n) * V + h) * 2 * D + j] = nv;
    }
  }
  __syncthreads();

  // Delta rule: warps (wr, wc) cover the D x D state; nwr = nwc = D / 32.
  const bool active = w < nw * nw;
  const int wr = w / nw, wc = w % nw, col = wc * 32 + lane;
  float s[32];
  float* rows = state + (size_t(h) * D + wr * 32) * D + col;
  float m = 0.f;  // k.S of the next token
  if (active) {
#pragma unroll
    for (int r = 0; r < 32; ++r) s[r] = rows[r * D];
    // Commit the previous window's accepted tokens, then store the committed state.
    for (int e = 0; e < A; ++e) {
      const float dec = rdec[e];
      const float* kr = rk + e * 2 * D + wr * 32;
      const float dj = rk[e * 2 * D + D + col];
#pragma unroll
      for (int r = 0; r < 32; ++r) {
        s[r] *= dec;
        s[r] = fmaf(kr[r], dj, s[r]);
      }
    }
#pragma unroll
    for (int r = 0; r < 32; ++r) rows[r * D] = s[r];
    const float dec = bd[0][1];
#pragma unroll
    for (int r = 0; r < 32; ++r) {
      s[r] *= dec;
      m = fmaf(s[r], ks[wr * 32 + r], m);
    }
  }
  for (int n = 0; n < N; ++n) {
    const float* kn = ks + n * D;
    const float* qn = qs + n * D;
    if (active) part[wr * D + col] = m;
    __syncthreads();
    if (t < D) {
      float tot = 0.f;
      for (int i = 0; i < nw; ++i) tot += part[i * D + t];
      const float delta = (vs[n * D + t] - tot) * bd[n][0];
      os[t] = delta;
      kd_wr[(size_t(n) * V + h) * 2 * D + D + t] = delta;
    }
    __syncthreads();
    if (active) {
      const float dj = os[col];
      float acc = 0.f, next = 0.f;
      if (n + 1 < N) {
        const float dec = bd[n + 1][1];
        const float* kx = kn + D;
#pragma unroll
        for (int r = 0; r < 32; ++r) {
          s[r] = fmaf(kn[wr * 32 + r], dj, s[r]);
          acc = fmaf(s[r], qn[wr * 32 + r], acc);
          s[r] *= dec;
          next = fmaf(s[r], kx[wr * 32 + r], next);
        }
      } else {
#pragma unroll
        for (int r = 0; r < 32; ++r) {
          s[r] = fmaf(kn[wr * 32 + r], dj, s[r]);
          acc = fmaf(s[r], qn[wr * 32 + r], acc);
        }
      }
      part2[wr * D + col] = acc;
      m = next;
    }
    __syncthreads();
    if (t < D) {
      float o = 0.f;
      for (int i = 0; i < nw; ++i) o += part2[i * D + t];
      vs[n * D + t] = o;  // this thread read v[n] above; the slot now holds the output
    }
  }
  __syncthreads();
  // Gated RMSNorm over each token's D outputs (threads [0, D)), all tokens together.
#pragma unroll
  for (int n = 0; n < NMAX; ++n) {
    if (n >= N) break;
    const float o = t < D ? vs[n * D + t] : 0.f;
    const float oo = warp_sum(t < D ? __fmul_rn(o, o) : 0.f);
    if (lane == 0) red[n][w] = oo;
  }
  __syncthreads();
  if (t < D) {
#pragma unroll
    for (int n = 0; n < NMAX; ++n) {
      if (n >= N) break;
      float tot = 0.f;
      for (int i = 0; i < D / 32; ++i) tot += red[n][i];
      const float o = vs[n * D + t];
      y[(size_t(n) * V + h) * D + t] = o * rsqrtf(tot / float(D) + eps) * norm_w[t] * sigmoidf(zs[n * D + t]);
    }
  }
}

// The original gdn_fused (one token after the other, 5 barriers per token), kept as the reference
// for tests/gdn_window.rs.
// The GDN step of N consecutive tokens for one value head h in one block (N <= 8).
//
// Delayed commit (docs/m3.md §4.1): a step does not know yet how many of its tokens will be
// committed (speculative verification decides that at the end of the step). So it leaves records of
// its window and never writes the state past it; the next step first applies the first `*napply`
// records of the previous window to the committed state, writes that back, then runs its own window.
// Records and conv histories alternate by step parity `par`: a step reads [par], writes [par ^ 1].
//   rec [2][RS], RS = NMAX C + NMAX V (2 D + 1):  in [NMAX][C] the conv inputs of the q/k/v
//   channels, then kd [NMAX][V][2][D] (normalized k, delta), dec [NMAX][V] (decay).
//
// Per block:
//   * conv + SiLU of this head's q, k (key head kh = h / (V / QK)) and v channels from the input
//     projection's split-K partials proj[s][n][ld], row [q | k | v | z | b | a]; the conv window of
//     token n is the committed history (the three inputs before the window) followed by the window's
//     own inputs. The first value head of each key head writes the q/k history and records, every
//     block its own v channels;
//   * L2 norm of q and k (q also scaled by D^-1/2), beta = sigmoid(b), decay;
//   * the delta rule, token after token, on the D x D state held in registers (warp (wr, wc): key
//     rows 32 wr.., value columns 32 wc.., lane = column: coalesced 128-byte lines);
//   * y[n][h] = o * rsqrt(mean o^2 + eps) * w * sigmoid(z).
// A record is applied with exactly the operations its token's own step ran (s *= decay, then
// s += k delta), so every token sees the same state bits as in single-token decoding.
// grid = V, blockDim = 4 D (D <= 128), smem: (4 N + 1) D + (D / 32) D floats.
extern "C" __global__ void __launch_bounds__(512) gdn_fused_ref(
    const float* __restrict__ proj, int S, int ld, float* __restrict__ conv_state, const int* __restrict__ par,
    const int* __restrict__ napply, float* __restrict__ rec, const float* __restrict__ conv_w,
    const float* __restrict__ a_log, const float* __restrict__ dt_bias, float* __restrict__ state,
    const float* __restrict__ norm_w, float* __restrict__ y, int QK, int V, int D, float eps, int N) {
  TSJ_PDL_ENTRY;
  extern __shared__ float sm[];
  float* vs = sm;             // [N][D] v after conv
  float* qs = vs + N * D;     // [N][D] normalized, scaled q
  float* ks = qs + N * D;     // [N][D] normalized k
  float* zs = ks + N * D;     // [N][D]
  float* os = zs + N * D;     // [D] delta, then output partials
  float* part = os + D;       // [D / 32][D] column partials
  __shared__ float red[32];
  __shared__ float bd[8][2];  // beta, decay per token
  const int h = blockIdx.x, t = threadIdx.x, lane = t & 31, w = t >> 5;
  const int kh = h / (V / QK), C = (2 * QK + V) * D;
  const int parity = *par, A = *napply;
  const float* old = conv_state + size_t(parity) * 3 * C;
  float* neu = conv_state + size_t(parity ^ 1) * 3 * C;
  const size_t RS = size_t(NMAX) * C + size_t(NMAX) * V * (2 * D + 1);
  const float* in_rd = rec + parity * RS;
  const float* kd_rd = in_rd + size_t(NMAX) * C;
  const float* dec_rd = kd_rd + size_t(NMAX) * V * 2 * D;
  float* in_wr = rec + (parity ^ 1) * RS;
  float* kd_wr = in_wr + size_t(NMAX) * C;
  float* dec_wr = kd_wr + size_t(NMAX) * V * 2 * D;
  const bool qk_writer = h % (V / QK) == 0;
  auto psum = [&](int n, int i) {
    float a = 0.f;
    for (int s = 0; s < S; ++s) a += proj[(s * N + n) * ld + i];
    return a;
  };
  // Thread groups of D: [0, D) q, [D, 2D) k, [2D, 3D) v, [3D, 4D) z.
  const int g = t / D, j = t % D;
  const int c = g == 0 ? kh * D + j : g == 1 ? (QK + kh) * D + j : 2 * QK * D + h * D + j;
  float conv[8];  // this thread's conv output per token (q, k, v groups)
  if (g < 3) {
    const bool writer = g == 2 || qk_writer;
    const float4 wc = reinterpret_cast<const float4*>(conv_w)[c];
    float hist[3] = {old[c], old[C + c], old[2 * C + c]};  // inputs at -3, -2, -1
    for (int e = 0; e < A; ++e) {  // commit the previous window's accepted inputs
      hist[0] = hist[1];
      hist[1] = hist[2];
      hist[2] = in_rd[e * C + c];
    }
    if (writer) {
      neu[c] = hist[0];
      neu[C + c] = hist[1];
      neu[2 * C + c] = hist[2];
    }
#pragma unroll
    for (int n = 0; n < 8; ++n) {
      if (n >= N) break;
      const float x = psum(n, c);
      if (writer) in_wr[n * C + c] = x;
      conv[n] = siluf(conv4(wc, hist, x));
      hist[0] = hist[1];
      hist[1] = hist[2];
      hist[2] = x;
    }
  } else {
    for (int n = 0; n < N; ++n) zs[n * D + j] = psum(n, C + h * D + j);
  }
  if (t < N) {
    const int ab = C + V * D;
    bd[t][0] = sigmoidf(psum(t, ab + h));
    bd[t][1] = expf(-expf(a_log[h]) * softplusf(psum(t, ab + V + h) + dt_bias[h]));
    dec_wr[t * V + h] = bd[t][1];
  }
  // L2 norms of q (group 0) and k (group 1) per token: warp sums, then the group's warps in order.
  const int wpg = D / 32;
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    if (n >= N) break;  // N is uniform: the barriers below stay matched
    const float val = g < 3 ? conv[n] : 0.f;
    const float sq = warp_sum(g < 2 ? val * val : 0.f);
    __syncthreads();
    if (lane == 0) red[w] = sq;
    __syncthreads();
    if (g < 2) {
      float tot = 0.f;
      for (int i = 0; i < wpg; ++i) tot += red[i + g * wpg];
      float r = rsqrtf(tot + 1e-6f);
      if (g == 0) r *= rsqrtf(float(D));
      const float nv = val * r;
      (g == 0 ? qs : ks)[n * D + j] = nv;
      if (g == 1) kd_wr[(size_t(n) * V + h) * 2 * D + j] = nv;
    }
  }
  __syncthreads();
  // v per token.
  if (g == 2) {
#pragma unroll
    for (int n = 0; n < 8; ++n)
      if (n < N) vs[n * D + j] = conv[n];
  }
  __syncthreads();

  // Delta rule: warps (wr, wc) cover the D x D state; nwr = nwc = D / 32.
  const int nw = D / 32;
  const bool active = w < nw * nw;
  const int wr = w / nw, wc = w % nw, col = wc * 32 + lane;
  float s[32];
  float* rows = state + (size_t(h) * D + wr * 32) * D + col;
  if (active) {
#pragma unroll
    for (int r = 0; r < 32; ++r) s[r] = rows[r * D];
    // Commit the previous window's accepted tokens, then store the committed state.
    for (int e = 0; e < A; ++e) {
      const float dec = dec_rd[e * V + h];
      const float* kr = kd_rd + (size_t(e) * V + h) * 2 * D + wr * 32;
      const float dj = kd_rd[(size_t(e) * V + h) * 2 * D + D + col];
#pragma unroll
      for (int r = 0; r < 32; ++r) {
        s[r] *= dec;
        s[r] = fmaf(kr[r], dj, s[r]);
      }
    }
#pragma unroll
    for (int r = 0; r < 32; ++r) rows[r * D] = s[r];
  }
  for (int n = 0; n < N; ++n) {
    const float* kn = ks + n * D;
    const float* qn = qs + n * D;
    if (active) {
      const float dec = bd[n][1];
      float m = 0.f;
#pragma unroll
      for (int r = 0; r < 32; ++r) {
        s[r] *= dec;
        m = fmaf(s[r], kn[wr * 32 + r], m);
      }
      part[wr * D + col] = m;
    }
    __syncthreads();
    if (t < D) {
      float tot = 0.f;
      for (int i = 0; i < nw; ++i) tot += part[i * D + t];
      const float delta = (vs[n * D + t] - tot) * bd[n][0];
      os[t] = delta;
      kd_wr[(size_t(n) * V + h) * 2 * D + D + t] = delta;
    }
    __syncthreads();
    if (active) {
      const float dj = os[col];
      float acc = 0.f;
#pragma unroll
      for (int r = 0; r < 32; ++r) {
        s[r] = fmaf(kn[wr * 32 + r], dj, s[r]);
        acc = fmaf(s[r], qn[wr * 32 + r], acc);
      }
      part[wr * D + col] = acc;
    }
    __syncthreads();
    float o = 0.f;
    if (t < D)
      for (int i = 0; i < nw; ++i) o += part[i * D + t];
    // Gated RMSNorm over the head's D outputs (threads [0, D) hold them).
    const float oo = warp_sum(t < D ? o * o : 0.f);
    __syncthreads();
    if (lane == 0) red[w] = oo;
    __syncthreads();
    if (t < D) {
      float tot = 0.f;
      for (int i = 0; i < D / 32; ++i) tot += red[i];
      y[(size_t(n) * V + h) * D + t] = o * rsqrtf(tot / float(D) + eps) * norm_w[t] * sigmoidf(zs[n * D + t]);
    }
  }
}

// Window path: one CTA per 32 value columns, with all D key rows (D threads, D/32 warps).
// A column has no dependency on the other columns until the final RMSNorm, so the recurrence
// runs on V * D/32 CTAs (192 for the real model) instead of V (48). Each lane still owns the
// same 32 consecutive key rows, and the column partials are added in the same row-group order.
// gdn_window_norm does the cross-column norm after this grid, in parallel for every token.
// N=1 continues to use gdn_fused. The record layout and the delayed-commit arithmetic are shared.
// grid = (V, D/32), block = D, smem: (2 N D + N*32 + 32 + 2 D) floats; D % 32 == 0, D <= 128.
extern "C" __global__ void __launch_bounds__(128) gdn_window(
    const float* __restrict__ proj, int S, int ld, float* __restrict__ conv_state, const int* __restrict__ par,
    const int* __restrict__ napply, float* __restrict__ rec, const float* __restrict__ conv_w,
    const float* __restrict__ a_log, const float* __restrict__ dt_bias, float* __restrict__ state,
    const float* __restrict__ norm_w, float* __restrict__ y, int QK, int V, int D, float eps, int N) {
  TSJ_PDL_ENTRY;
  extern __shared__ float sm[];
  float* qs = sm;                 // [N][D]
  float* ks = qs + N * D;         // [N][D]
  float* vs = ks + N * D;         // [N][32], this CTA's value columns
  float* ds = vs + N * 32;        // [32] delta
  float* part = ds + 32;          // [D/32][32] k.S column partials
  float* out_part = part + D;     // [D/32][32] output column partials
  __shared__ float red[2][4];
  __shared__ float bd[NMAX][2];
  const int h = blockIdx.x, chunk = blockIdx.y, t = threadIdx.x;
  const int lane = t & 31, wr = t >> 5, col = chunk * 32 + lane, nw = D / 32;
  const int kh = h / (V / QK), C = (2 * QK + V) * D;
  const int parity = *par, A = *napply;
  const float* old = conv_state + size_t(parity) * 3 * C;
  float* neu = conv_state + size_t(parity ^ 1) * 3 * C;
  const size_t RS = size_t(NMAX) * C + size_t(NMAX) * V * (2 * D + 1);
  const float* in_rd = rec + parity * RS;
  const float* kd_rd = in_rd + size_t(NMAX) * C;
  const float* dec_rd = kd_rd + size_t(NMAX) * V * 2 * D;
  float* in_wr = rec + (parity ^ 1) * RS;
  float* kd_wr = in_wr + size_t(NMAX) * C;
  float* dec_wr = kd_wr + size_t(NMAX) * V * 2 * D;
  auto psum = [&](int n, int i) {
    float a = 0.f;
    for (int s = 0; s < S; ++s) a += proj[(s * N + n) * ld + i];
    return a;
  };
  auto conv = [&](int c, bool writer, float (&out)[NMAX]) {
    const float4 wc = reinterpret_cast<const float4*>(conv_w)[c];
    float hist[3] = {old[c], old[C + c], old[2 * C + c]};
    for (int e = 0; e < A; ++e) {
      hist[0] = hist[1];
      hist[1] = hist[2];
      hist[2] = in_rd[e * C + c];
    }
    if (writer) {
      neu[c] = hist[0];
      neu[C + c] = hist[1];
      neu[2 * C + c] = hist[2];
    }
#pragma unroll
    for (int n = 0; n < NMAX; ++n) {
      if (n >= N) break;
      const float x = psum(n, c);
      if (writer) in_wr[n * C + c] = x;
      out[n] = siluf(conv4(wc, hist, x));
      hist[0] = hist[1];
      hist[1] = hist[2];
      hist[2] = x;
    }
  };
  float qv[NMAX], kv[NMAX];
  const bool qk_writer = chunk == 0 && h % (V / QK) == 0;
  conv(kh * D + t, qk_writer, qv);
  conv((QK + kh) * D + t, qk_writer, kv);
  if (wr == 0) {
    float vv[NMAX];
    conv(2 * QK * D + h * D + col, true, vv);
#pragma unroll
    for (int n = 0; n < NMAX; ++n)
      if (n < N) vs[n * 32 + lane] = vv[n];
  }
  if (t < N) {
    const int ab = C + V * D;
    bd[t][0] = sigmoidf(psum(t, ab + h));
    bd[t][1] = expf(-expf(a_log[h]) * softplusf(psum(t, ab + V + h) + dt_bias[h]));
    if (chunk == 0) dec_wr[t * V + h] = bd[t][1];
  }
#pragma unroll
  for (int n = 0; n < NMAX; ++n) {
    if (n >= N) break;
    // __fmul_rn: a plain square feeding the first shuffle add may be contracted into an FFMA by
    // ptxas (gdn_fused squares through a select, which it does not contract).
    const float sq = warp_sum(__fmul_rn(qv[n], qv[n])), sk = warp_sum(__fmul_rn(kv[n], kv[n]));
    __syncthreads();
    if (lane == 0) red[0][wr] = sq, red[1][wr] = sk;
    __syncthreads();
    float tq = 0.f, tk = 0.f;
    for (int i = 0; i < nw; ++i) tq += red[0][i], tk += red[1][i];
    float rq = rsqrtf(tq + 1e-6f);
    rq *= rsqrtf(float(D));
    qs[n * D + t] = qv[n] * rq;
    const float k = kv[n] * rsqrtf(tk + 1e-6f);
    ks[n * D + t] = k;
    if (chunk == 0) kd_wr[(size_t(n) * V + h) * 2 * D + t] = k;
  }
  __syncthreads();

  float s[32];
  float* rows = state + (size_t(h) * D + wr * 32) * D + col;
#pragma unroll
  for (int r = 0; r < 32; ++r) s[r] = rows[r * D];
  for (int e = 0; e < A; ++e) {
    const float dec = dec_rd[e * V + h];
    const float* kr = kd_rd + (size_t(e) * V + h) * 2 * D + wr * 32;
    const float dj = kd_rd[(size_t(e) * V + h) * 2 * D + D + col];
#pragma unroll
    for (int r = 0; r < 32; ++r) {
      s[r] *= dec;
      s[r] = fmaf(kr[r], dj, s[r]);
    }
  }
#pragma unroll
  for (int r = 0; r < 32; ++r) rows[r * D] = s[r];

  // Seed token 0's k.S. Later tokens compute it alongside the preceding output dot, after
  // that token has consumed the updated state element. Two independent dot chains overlap
  // without regrouping either sum; each element still sees update, next decay, next k.S.
  float m = 0.f;
#pragma unroll
  for (int r = 0; r < 32; ++r) {
    s[r] *= bd[0][1];
    m = fmaf(s[r], ks[wr * 32 + r], m);
  }
  for (int n = 0; n < N; ++n) {
    const float* kn = ks + n * D;
    const float* qn = qs + n * D;
    part[wr * 32 + lane] = m;
    __syncthreads();
    if (wr == 0) {
      float tot = 0.f;
      for (int i = 0; i < nw; ++i) tot += part[i * 32 + lane];
      const float delta = (vs[n * 32 + lane] - tot) * bd[n][0];
      ds[lane] = delta;
      kd_wr[(size_t(n) * V + h) * 2 * D + D + col] = delta;
    }
    __syncthreads();
    const float dj = ds[lane];
    float acc = 0.f, next_m = 0.f;
    const bool more = n + 1 < N;
    const float next_dec = more ? bd[n + 1][1] : 1.f;
#pragma unroll
    for (int r = 0; r < 32; ++r) {
      s[r] = fmaf(kn[wr * 32 + r], dj, s[r]);
      acc = fmaf(s[r], qn[wr * 32 + r], acc);
      if (more) {
        s[r] *= next_dec;
        next_m = fmaf(s[r], kn[D + wr * 32 + r], next_m);
      }
    }
    m = next_m;
    // Separate storage lets the next token write k.S while warp 0 still reads this output.
    out_part[wr * 32 + lane] = acc;
    __syncthreads();
    if (wr == 0) {
      float o = 0.f;
      for (int i = 0; i < nw; ++i) o += out_part[i * 32 + lane];
      y[(size_t(n) * V + h) * D + col] = o;
    }
  }
}

// Same warp squares and ordered warp sum as gdn_fused's final norm. Reading/writing y in place
// is safe: each (token, head) belongs to one CTA and block_sum finishes the reads before stores.
// grid = (V, N), block = D, no dynamic smem. z is reduced from the same split-K slices in order.
extern "C" __global__ void __launch_bounds__(128) gdn_window_norm(
    float* __restrict__ y, const float* __restrict__ proj, int S, int ld,
    const float* __restrict__ norm_w, int QK, int V, int D, float eps, int N) {
  TSJ_PDL_ENTRY;
  __shared__ float red[32];
  const int h = blockIdx.x, n = blockIdx.y, t = threadIdx.x;
  const size_t at = (size_t(n) * V + h) * D + t;
  const float o = y[at];
  const float tot = block_sum(__fmul_rn(o, o), red);  // uncontracted square, as in gdn_fused
  const int zrow = (2 * QK + V) * D + h * D + t;
  float z = 0.f;
  for (int s = 0; s < S; ++s) z += proj[(s * N + n) * ld + zrow];
  y[at] = o * rsqrtf(tot / float(D) + eps) * norm_w[t] * sigmoidf(z);
}
