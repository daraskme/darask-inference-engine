// QSA (indexed sparse attention) decode for N tokens at positions p0 .. p0 + N - 1. Matches
// transformers `Qwen4ExpTextAttention` + `Qwen4ExpTextQSAIndexer` with a cache:
//   * q_proj gives [query | gate] per head; q/k get (1 + w) RMSNorm, then RoPE on the first ROPE dims.
//   * The indexer keeps raw keys per token; every IR tokens form a block whose key is
//     mean(raw keys) -> k_layernorm -> RoPE at the block's first position. A block is complete as
//     soon as its last token (possibly the current one) is visible.
//   * Each block scores sum_h relu(iq_h . key_b) / sqrt(ID); the top TOPB complete blocks plus the
//     incomplete tail (up to IR - 1 tokens, including p) are attended to.
// Caches (kv_t, fp32 by default): K/V [ctx][AKV][AD], raw keys [ctx][ID], block keys [ctx/IR][ID].
//
// Token n of the window sees exactly what it would see if the window's tokens came one step at a
// time: positions up to its own, and the blocks completed by then. The window's K/V and raw keys
// are written to the caches at their positions (a verification that rejects a token later
// rewrites that position before anything reads it). N=1's fused query blocks recompute their
// own K/V; for N>1, attention reads the caches after the preparation grid has completed.
#include "decode.cuh"

// Cache element type: fp32 by default. Rounding K, V and the indexer keys to bf16 (as the bf16
// reference does) costs KL ~0.01 on this model (docs/m1.md); build with
// TSJ_NVCC_DEFINES=-DTSJ_KV_BF16 for half the cache size.
#ifdef TSJ_KV_BF16
using kv_t = __nv_bfloat16;
TSJ_DEVICE kv_t to_kv(float x) { return __float2bfloat16(x); }
TSJ_DEVICE float kv_f(kv_t x) { return __bfloat162float(x); }
#else
using kv_t = float;
TSJ_DEVICE kv_t to_kv(float x) { return x; }
TSJ_DEVICE float kv_f(kv_t x) { return x; }
#endif

struct QsaDims {
  int AQ, AKV, AD, ROPE, IH, ID, IR, TOPB;
  float theta, eps;
};

// Projections, attention of the dense regime, and cache writes.
// proj = [q_proj (AQ x 2AD: query | gate per head) | k (AKV x AD) | v (AKV x AD) | idx (IH x ID + ID)]
// per token, as S split-K partial slices: token m of slice s at proj[(s N + m) ld] (summed here).
// For N=1, the query attends to every visible token ((p + 1) / IR <= TOPB complete blocks: the
// first QSA_MAX_KEYS positions), the selection is all of 0..p, so its query-head block attends
// right away and `qsa_scores`, `qsa_attn` return at once for it; past that, it only leaves q and
// the gate for them.
// grid = (AQ + 2, N), blockDim = AD. Blocks x: [0, AQ) query heads (each also computes its KV
// head's k and v for N=1; for N>1 only the first query head of each KV group computes its own
// token's K/V and writes the cache). N>1 attends in qsa_attn after this grid completes, so each
// K/V is prepared once instead of repeating norm/RoPE AQ/AKV * N(N+1)/2 times.
// AQ indexer queries, AQ + 1 raw key and block key. smem: 4 AD + max tokens floats.
extern "C" __global__ void __launch_bounds__(256) qsa_prep_attn(
    const float* __restrict__ proj, int S, int ld, const QsaDims d, const int* __restrict__ pos,
    const float* __restrict__ q_norm, const float* __restrict__ k_norm, const float* __restrict__ iq_norm,
    const float* __restrict__ ik_norm, float* __restrict__ q, float* __restrict__ gate, kv_t* __restrict__ kc,
    kv_t* __restrict__ vc, float* __restrict__ iq, kv_t* __restrict__ raw, kv_t* __restrict__ bkeys,
    float* __restrict__ out, int N) {
  TSJ_PDL_ENTRY;
  const int tn = blockIdx.y, p0 = *pos, p = p0 + tn;
  extern __shared__ float sm[];
  __shared__ float red[32];
  float* buf = sm;
  float* qs = sm + d.AD;
  float* ks = qs + d.AD;  // N=1: current key and value for fused dense attention
  float* vs = ks + d.AD;
  float* sc = vs + d.AD;
  const int b = blockIdx.x, t = threadIdx.x, n = blockDim.x;
  const int kvo = d.AQ * 2 * d.AD, io = kvo + 2 * d.AKV * d.AD;
  // Projection output i of window token m.
  auto P = [&](int m, int i) {
    float a = 0.f;
    for (int s = 0; s < S; ++s) a += proj[size_t(s * N + m) * ld + i];
    return a;
  };
  auto norm_rope = [&](int len, const float* w, int at) {
    float ss = 0.f;
    for (int j = t; j < len; j += n) ss = fmaf(buf[j], buf[j], ss);
    const float r = rsqrtf(block_sum(ss, red) / float(len) + d.eps);
    for (int j = t; j < len; j += n) buf[j] = buf[j] * r * (1.f + w[j]);
    __syncthreads();
    rope_inplace(buf, d.ROPE, at, d.theta, t, n);
    __syncthreads();
  };
  const bool dense = (p + 1) / d.IR <= d.TOPB;

  if (b < d.AQ) {
    const int kvh = b / (d.AQ / d.AKV);
    const size_t qo = (size_t(tn) * d.AQ + b) * d.AD;
    float g[1] = {0.f};
    for (int j = t; j < d.AD; j += n) {
      buf[j] = P(tn, b * 2 * d.AD + j);
      g[0] = P(tn, b * 2 * d.AD + d.AD + j);  // blockDim == AD: one element per thread
      gate[qo + j] = g[0];
    }
    __syncthreads();
    norm_rope(d.AD, q_norm, p);
    for (int j = t; j < d.AD; j += n) {
      qs[j] = buf[j];
      q[qo + j] = buf[j];
    }
    __syncthreads();
    const bool writer = b % (d.AQ / d.AKV) == 0;
    if (N > 1 && !writer) return;
    {
      for (int j = t; j < d.AD; j += n) buf[j] = P(tn, kvo + kvh * d.AD + j);
      __syncthreads();
      norm_rope(d.AD, k_norm, p);
      const size_t at = (size_t(p) * d.AKV + kvh) * d.AD;
      for (int j = t; j < d.AD; j += n) {
        const kv_t kq = to_kv(buf[j]), vq = to_kv(P(tn, kvo + (d.AKV + kvh) * d.AD + j));
        if (N == 1) {
          ks[j] = kv_f(kq);
          vs[j] = kv_f(vq);
        }
        if (writer) {
          kc[at + j] = kq;
          vc[at + j] = vq;
        }
      }
      __syncthreads();
    }
    if (N > 1 || !dense) return;  // K/V are now ready for qsa_attn (also dense when N>1)

    // Dense attention over 0..p (the window positions from shared memory).
    const int lane = t & 31, w = t >> 5, nw = n >> 5, cnt = p + 1;
    const float scale = rsqrtf(float(d.AD));
    for (int i = w; i < cnt; i += nw) {
      float s = 0.f;
      if (i < p0) {
        const kv_t* k = kc + (size_t(i) * d.AKV + kvh) * d.AD;
        for (int j = lane; j < d.AD; j += 32) s = fmaf(qs[j], kv_f(k[j]), s);
      } else {
        const float* k = ks + (i - p0) * d.AD;
        for (int j = lane; j < d.AD; j += 32) s = fmaf(qs[j], k[j], s);
      }
      s = warp_sum(s);
      if (lane == 0) sc[i] = s * scale;
    }
    __syncthreads();
    float m = -INFINITY;
    for (int i = t; i < cnt; i += n) m = fmaxf(m, sc[i]);
    m = block_max(m, red);
    float sum = 0.f;
    for (int i = t; i < cnt; i += n) {
      sc[i] = expf(sc[i] - m);
      sum += sc[i];
    }
    sum = block_sum(sum, red);
    for (int j = t; j < d.AD; j += n) {
      float acc = 0.f;
      for (int i = 0; i < p0; ++i) acc = fmaf(sc[i], kv_f(vc[(size_t(i) * d.AKV + kvh) * d.AD + j]), acc);
      for (int i = p0; i <= p; ++i) acc = fmaf(sc[i], vs[(i - p0) * d.AD + j], acc);
      out[qo + j] = acc / sum * sigmoidf(g[0]);
    }
  } else if (b == d.AQ) {
    if (dense) return;  // the block scores are not used
    for (int h = 0; h < d.IH; ++h) {
      for (int j = t; j < d.ID; j += n) buf[j] = P(tn, io + h * d.ID + j);
      __syncthreads();
      norm_rope(d.ID, iq_norm, p);
      for (int j = t; j < d.ID; j += n) iq[(size_t(tn) * d.IH + h) * d.ID + j] = buf[j];
      __syncthreads();
    }
  } else {
    const int ko = io + d.IH * d.ID;
    for (int j = t; j < d.ID; j += n) raw[size_t(p) * d.ID + j] = to_kv(P(tn, ko + j));
    if ((p + 1) % d.IR != 0) return;
    const int first = p + 1 - d.IR;
    for (int j = t; j < d.ID; j += n) {
      float s = 0.f;
      for (int r = first; r < p; ++r) s += r < p0 ? kv_f(raw[size_t(r) * d.ID + j]) : kv_f(to_kv(P(r - p0, ko + j)));
      s += kv_f(to_kv(P(tn, ko + j)));
      // The reference averages in fp32 and stores the pooled key in the cache dtype.
      buf[j] = kv_f(to_kv(s / float(d.IR)));
    }
    __syncthreads();
    norm_rope(d.ID, ik_norm, first);
    for (int j = t; j < d.ID; j += n) bkeys[size_t(first / d.IR) * d.ID + j] = to_kv(buf[j]);
  }
}

// scores[n][b] = sum_h relu(iq[n]_h . bkeys[b]) / sqrt(ID) for token n's nb = (p + 1) / IR complete
// blocks, only when they exceed the budget TOPB (otherwise every block is taken). SB = scores per
// token. grid = (blocks, N), warp per block key, grid-stride.
extern "C" __global__ void __launch_bounds__(256) qsa_scores(const float* __restrict__ iq,
                                                            const kv_t* __restrict__ bkeys,
                                                            float* __restrict__ scores, const int* __restrict__ pos,
                                                            int IR, int TOPB, int IH, int ID, int SB) {
  TSJ_PDL_ENTRY;
  const int tn = blockIdx.y, nb = (*pos + tn + 1) / IR;
  if (nb <= TOPB) return;
  iq += size_t(tn) * IH * ID;
  scores += size_t(tn) * SB;
  const int lane = threadIdx.x & 31, wpb = blockDim.x >> 5;
  for (int b = blockIdx.x * wpb + (threadIdx.x >> 5); b < nb; b += gridDim.x * wpb) {
    float total = 0.f;
    for (int h = 0; h < IH; ++h) {
      float s = 0.f;
      for (int j = lane; j < ID; j += 32) s = fmaf(iq[h * ID + j], kv_f(bkeys[size_t(b) * ID + j]), s);
      total += fmaxf(warp_sum(s), 0.f);
    }
    if (lane == 0) scores[b] = total / sqrtf(float(ID));
  }
}

// Token list of token n to attend: the top `topb` of nb complete blocks (all of them when
// nb <= topb; ties at the threshold go to lower block ids), expanded to IR tokens each in block
// order, then the tail tokens nb*IR .. p. toks [N][MT], n_out [N]. grid = N blocks of 1024
// threads; deterministic.
extern "C" __global__ void __launch_bounds__(1024) qsa_select(const float* __restrict__ scores,
                                                             const int* __restrict__ pos, int topb, int IR,
                                                             int* __restrict__ toks, int* __restrict__ n_out, int SB,
                                                             int MT) {
  TSJ_PDL_ENTRY;
  const int tn = blockIdx.x, p = *pos + tn, nb = (p + 1) / IR;
  scores += size_t(tn) * SB;
  toks += size_t(tn) * MT;
  __shared__ int cnt;
  __shared__ int wsel[32], weq[32];
  __shared__ int base_sel, base_eq;
  const int t = threadIdx.x, lane = t & 31, w = t >> 5;
  const unsigned* bits = reinterpret_cast<const unsigned*>(scores);  // scores >= 0: bits are ordered
  unsigned thr = 0;
  int take_eq = 0;
  const bool all = nb <= topb;
  if (!all) {
    auto count_ge = [&](unsigned v) {
      if (t == 0) cnt = 0;
      __syncthreads();
      int c = 0;
      for (int b = t; b < nb; b += blockDim.x) c += bits[b] >= v;
      atomicAdd(&cnt, c);
      __syncthreads();
      const int r = cnt;
      __syncthreads();
      return r;
    };
    unsigned lo = 0, hi = 0x7f800001u;  // count(>= lo) >= topb > count(>= hi)
    while (hi - lo > 1) {
      const unsigned mid = lo + (hi - lo) / 2;
      if (count_ge(mid) >= topb) lo = mid; else hi = mid;
    }
    thr = lo;
    take_eq = topb - count_ge(thr + 1);
  }
  if (t == 0) base_sel = 0, base_eq = 0;
  __syncthreads();
  for (int c0 = 0; c0 < nb; c0 += blockDim.x) {
    const int b = c0 + t;
    const bool in = b < nb;
    const bool gt = in && (all || bits[b] > thr);
    const bool eq = in && !all && bits[b] == thr;
    const unsigned meq = __ballot_sync(0xffffffffu, eq);
    if (lane == 0) weq[w] = __popc(meq);
    __syncthreads();
    int eq_before = base_eq;
    for (int i = 0; i < w; ++i) eq_before += weq[i];
    eq_before += __popc(meq & ((1u << lane) - 1));
    const bool sel = gt || (eq && eq_before < take_eq);
    const unsigned msel = __ballot_sync(0xffffffffu, sel);
    if (lane == 0) wsel[w] = __popc(msel);
    __syncthreads();
    int at = base_sel;
    for (int i = 0; i < w; ++i) at += wsel[i];
    at += __popc(msel & ((1u << lane) - 1));
    if (sel)
      for (int r = 0; r < IR; ++r) toks[at * IR + r] = b * IR + r;
    __syncthreads();
    if (t == 0) {
      for (int i = 0; i < int(blockDim.x >> 5); ++i) base_sel += wsel[i], base_eq += weq[i];
    }
    __syncthreads();
  }
  const int nsel = base_sel;
  const int tail0 = nb * IR;
  for (int j = t; tail0 + j <= p; j += blockDim.x) toks[nsel * IR + j] = tail0 + j;
  if (t == 0) n_out[tn] = nsel * IR + (p + 1 - tail0);
}

// Attention of each query head of token n over its selected tokens, then the
// sigmoid output gate:
//   out[n][h] = sigmoid(gate[n][h]) * sum_t softmax_t(q[n]_h . K[t][kvh] / sqrt(AD)) V[t][kvh]
// grid = (AQ heads, N), blockDim = AD (256), smem: max tokens floats + AD floats.
extern "C" __global__ void __launch_bounds__(256) qsa_attn(const float* __restrict__ q, const float* __restrict__ gate,
                                                          const kv_t* __restrict__ kc,
                                                          const kv_t* __restrict__ vc,
                                                          const int* __restrict__ toks, const int* __restrict__ n_tok,
                                                          float* __restrict__ out, int AQ, int AKV, int AD,
                                                          const int* __restrict__ pos, int IR, int TOPB, int MT) {
  TSJ_PDL_ENTRY;
  const int tn = blockIdx.y;
  if (gridDim.y == 1 && (*pos + tn + 1) / IR <= TOPB) return;  // N=1 fused dense path
  extern __shared__ float sm[];
  __shared__ float red[32];
  float* qs = sm;
  float* sc = sm + AD;
  const int h = blockIdx.x, kvh = h / (AQ / AKV), t = threadIdx.x, lane = t & 31, w = t >> 5;
  const size_t qo = (size_t(tn) * AQ + h) * AD;
  toks += size_t(tn) * MT;
  const int n = n_tok[tn], nw = blockDim.x >> 5;
  for (int j = t; j < AD; j += blockDim.x) qs[j] = q[qo + j];
  __syncthreads();
  const float scale = rsqrtf(float(AD));
  for (int i = w; i < n; i += nw) {
    const kv_t* k = kc + (size_t(toks[i]) * AKV + kvh) * AD;
    float s = 0.f;
    for (int j = lane; j < AD; j += 32) s = fmaf(qs[j], kv_f(k[j]), s);
    s = warp_sum(s);
    if (lane == 0) sc[i] = s * scale;
  }
  __syncthreads();
  float m = -INFINITY;
  for (int i = t; i < n; i += blockDim.x) m = fmaxf(m, sc[i]);
  m = block_max(m, red);
  float sum = 0.f;
  for (int i = t; i < n; i += blockDim.x) {
    sc[i] = expf(sc[i] - m);
    sum += sc[i];
  }
  sum = block_sum(sum, red);
  for (int j = t; j < AD; j += blockDim.x) {
    float acc = 0.f;
    for (int i = 0; i < n; ++i) acc = fmaf(sc[i], kv_f(vc[(size_t(toks[i]) * AKV + kvh) * AD + j]), acc);
    out[qo + j] = acc / sum * sigmoidf(gate[qo + j]);
  }
}
