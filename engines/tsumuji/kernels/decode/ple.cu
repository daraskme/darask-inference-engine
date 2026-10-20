// PLE (per-layer n-gram embedding) decode for N tokens. Matches transformers
// `Qwen4ExpTextPLELayer`: hashed 2/3-gram rows from an FP8 table in pinned host memory, key/value
// projections, a signed square-root gate per residual stream, and a dilated depthwise conv over
// the gated values.
#include "decode.cuh"

struct PleHash {
  unsigned long long mult[3];
  unsigned long long size[16];
  unsigned long long offset[16];
};

// e[n][h*W + j] = fp8(table[row_h][j]) * scale for the 16 heads of token n; row width W bytes
// (W % 16 == 0). tok3[n] = [t0, t1, t2] (current, previous, before that; EOS-reset already
// applied). grid = N blocks of 16 * 32 threads: warp h handles head h.
extern "C" __global__ void __launch_bounds__(512) ple_gather(const uint8_t* __restrict__ table,
                                                            const PleHash hash, const int* __restrict__ tok3,
                                                            const __nv_bfloat16* __restrict__ scale,
                                                            float* __restrict__ e, int W) {
  TSJ_PDL_ENTRY;
  const int h = threadIdx.x >> 5, lane = threadIdx.x & 31, n = blockIdx.x;
  const int* tok = tok3 + 3 * n;
  e += size_t(n) * 16 * W;
  const unsigned long long t0 = tok[0], t1 = tok[1], t2 = tok[2];
  unsigned long long m = (t0 * hash.mult[0]) ^ (t1 * hash.mult[1]);
  if (h >= 8) m ^= t2 * hash.mult[2];
  const unsigned long long row = m % hash.size[h] + hash.offset[h];
  const uint4* src = reinterpret_cast<const uint4*>(table + row * W);
  const float sc = bf16_to_f(*scale);
  for (int v = lane; v < W / 16; v += 32) {
    const uint4 q = src[v];
    const uint8_t* b = reinterpret_cast<const uint8_t*>(&q);
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      __half_raw r = __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(b[j]), __NV_E4M3);
      e[h * W + v * 16 + j] = __half2float(*reinterpret_cast<__half*>(&r)) * sc;
    }
  }
}

// ple_gather for NVFP4 tables (nvidia's FP4-Local): a row is row_bytes bytes, [W/2 e2m1 codes (low
// nibble first) | W/16 e4m3 block scales | pad]; value = code * block scale * global (f32).
// grid = N blocks of 16 * 32 threads: warp h handles head h, lane g < W/16 one 16-value block.
extern "C" __global__ void __launch_bounds__(512) ple_gather_nvfp4(const uint8_t* __restrict__ table,
                                                                  const PleHash hash, const int* __restrict__ tok3,
                                                                  const float* __restrict__ global,
                                                                  float* __restrict__ e, int W, int row_bytes) {
  TSJ_PDL_ENTRY;
  const int h = threadIdx.x >> 5, lane = threadIdx.x & 31, n = blockIdx.x;
  const int* tok = tok3 + 3 * n;
  e += size_t(n) * 16 * W;
  const unsigned long long t0 = tok[0], t1 = tok[1], t2 = tok[2];
  unsigned long long m = (t0 * hash.mult[0]) ^ (t1 * hash.mult[1]);
  if (h >= 8) m ^= t2 * hash.mult[2];
  const unsigned long long row = m % hash.size[h] + hash.offset[h];
  const uint8_t* src = table + row * row_bytes;
  const float g = *global;
  for (int b = lane; b < W / 16; b += 32) {
    const uint2 q = *reinterpret_cast<const uint2*>(src + b * 8);
    const float s = e4m3_to_float(src[W / 2 + b]) * g;
    const uint32_t words[2] = {q.x, q.y};
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float2 f = __half22float2(e2m1x2_to_half2(uint8_t(words[j / 4] >> (8 * (j % 4)))));
      e[h * W + b * 16 + 2 * j] = f.x * s;
      e[h * W + b * 16 + 2 * j + 1] = f.y * s;
    }
  }
}

// Most tokens a step takes (engine.rs N_MAX): the capacity of the window records.
constexpr int NMAX = 8;

// kv[n] = [key (HC*H) | value (H)] from the projections. Per stream i (block i) and token n in order:
//   kn = norm_key(key_i), qn = norm_query(R_i)            ((1 + w) RMSNorm over H)
//   g = sum(kn * qn) / sqrt(H);  g = sign(g) * sqrt(max(|g|, 1e-6));  gv = sigmoid(g) * value
//   v = norm_conv(gv);  conv over time per channel c = i*H + k, kernel 4, dilation 3:
//     out = silu(w[c][0] v[t-9] + w[c][1] v[t-6] + w[c][2] v[t-3] + w[c][3] v[t])
//   R_i += gv + out
// Conv history with delayed commit (docs/m3.md §4.1), alternating by step parity `par` (read [par],
// write [par ^ 1]): state [2][9][HC*H] holds the committed v[-9 .. -1] (oldest first) as of the
// previous step, win [2][NMAX][HC*H] that step's window values, of which the first `*napply` are
// committed now. This step's history is the last 9 of (state ++ committed window values); it is
// written to state[par ^ 1] and this window's values to win[par ^ 1]. Each thread only touches its
// own channels. grid = HC blocks, 512 threads.
// Stream statistics: block i writes row i of ss[n] [HC][HC] (stream i's sum of squares, zeros elsewhere).
extern "C" __global__ void __launch_bounds__(512) ple_mix(float* __restrict__ R, const float* __restrict__ kv,
                                                         const float* __restrict__ w_key,
                                                         const float* __restrict__ w_query,
                                                         const float* __restrict__ w_conv,
                                                         const float* __restrict__ conv_w,
                                                         float* __restrict__ state, float* __restrict__ win,
                                                         const int* __restrict__ par, const int* __restrict__ napply,
                                                         float* __restrict__ ss, int H, int HC, float eps, int N) {
  TSJ_PDL_ENTRY;
  __shared__ float red[32];
  const int i = blockIdx.x;
  const size_t CH = size_t(HC) * H;
  const int pr = *par, A = *napply;
  const float* old = state + pr * 9 * CH;
  float* neu = state + (pr ^ 1) * 9 * CH;
  const float* win_rd = win + pr * NMAX * CH;
  float* win_wr = win + (pr ^ 1) * NMAX * CH;
  // Committed history row r (0..8, oldest first): the last 9 of (old ++ win_rd[0..A)).
  auto committed = [&](int r, size_t c) { return A + r < 9 ? old[(A + r) * CH + c] : win_rd[(A + r - 9) * CH + c]; };
  // v at window-relative time m (m < 0: the committed history), channel c.
  auto hist = [&](int m, size_t c) { return m >= 0 ? win_wr[m * CH + c] : committed(9 + m, c); };
  for (int n = 0; n < N; ++n) {
    float* Ri = R + (size_t(n) * HC + i) * H;
    const float* key = kv + size_t(n) * (CH + H) + size_t(i) * H;
    const float* value = kv + size_t(n) * (CH + H) + CH;
    float sk = 0.f, sq = 0.f;
    for (int k = threadIdx.x; k < H; k += blockDim.x) {
      sk = fmaf(key[k], key[k], sk);
      sq = fmaf(Ri[k], Ri[k], sq);
    }
    const float rk = rsqrtf(block_sum(sk, red) / float(H) + eps);
    const float rq = rsqrtf(block_sum(sq, red) / float(H) + eps);
    float dot = 0.f;
    for (int k = threadIdx.x; k < H; k += blockDim.x) {
      const size_t c = size_t(i) * H + k;
      dot += (key[k] * rk * (1.f + w_key[c])) * (Ri[k] * rq * (1.f + w_query[c]));
    }
    float g = block_sum(dot, red) / sqrtf(float(H));
    g = copysignf(sqrtf(fmaxf(fabsf(g), 1e-6f)), g) * (g != 0.f);
    const float sg = sigmoidf(g);
    float sv = 0.f;
    for (int k = threadIdx.x; k < H; k += blockDim.x) sv = fmaf(sg * value[k], sg * value[k], sv);
    const float rv = rsqrtf(block_sum(sv, red) / float(H) + eps);
    float nsq = 0.f;
    for (int k = threadIdx.x; k < H; k += blockDim.x) {
      const size_t c = size_t(i) * H + k;
      const float gv = sg * value[k];
      const float v = gv * rv * (1.f + w_conv[c]);
      const float4 w = reinterpret_cast<const float4*>(conv_w)[c];
      const float out = siluf(w.x * hist(n - 9, c) + w.y * hist(n - 6, c) + w.z * hist(n - 3, c) + w.w * v);
      win_wr[n * CH + c] = v;
      Ri[k] += gv + out;
      nsq = fmaf(Ri[k], Ri[k], nsq);
    }
    const float nss = block_sum(nsq, red);
    if (threadIdx.x < HC) ss[(size_t(n) * HC + i) * HC + threadIdx.x] = threadIdx.x == i ? nss : 0.f;
  }
  for (int k = threadIdx.x; k < H; k += blockDim.x) {
    const size_t c = size_t(i) * H + k;
    float rows[9];
#pragma unroll
    for (int t = 0; t < 9; ++t) rows[t] = committed(t, c);
#pragma unroll
    for (int t = 0; t < 9; ++t) neu[t * CH + c] = rows[t];
  }
}
