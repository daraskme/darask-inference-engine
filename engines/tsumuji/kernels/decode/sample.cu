// Token selection after the lm_head: greedy, or a sample from softmax(logits / T) restricted to
// the top_k largest logits and then to the smallest top set holding top_p of the mass (as
// transformers' TopK / TopP warpers; with the fast path, tokens tied with the k-th value may be
// cut by index).
//
// Sampling is Gumbel-max with noise from a counter-based hash of (seed, position, token id), so
// the draw at a position does not depend on which step or window computes it. With speculative
// decoding the window samples every position and a draft is accepted iff it equals the sample
// (`accept`): that accepts a draft d with probability p(d) and, on rejection, leaves a sample of p
// restricted to tokens != d, which is exact speculative sampling for deterministic drafts; and it
// gives the same tokens as one-token decoding with the same seed.
//
// Fast path (0 < top_k <= TOPK_MAX): `sample_candidates` splits the vocabulary over CHUNKS blocks
// per token, each keeping its top_k (value, index) pairs; `sample_tokens` merges them (the global
// top_k is among them), applies top_p and samples. Otherwise `sample_tokens` scans the whole
// vocabulary (greedy, or top_k off / large: slow, one block per token).
#include "decode.cuh"

struct SampleParams {
  float temperature;  // <= 0: greedy
  int top_k;          // 0: off
  float top_p;        // >= 1: off
  int pad;
  unsigned long long seed;
};

constexpr int CHUNKS = 128;    // candidate blocks per token
constexpr int TOPK_MAX = 64;   // fast path limit

// Orders floats like their values (for the threshold searches over bit patterns).
TSJ_DEVICE unsigned order_key(float x) {
  const unsigned b = __float_as_uint(x);
  return b & 0x80000000u ? ~b : b | 0x80000000u;
}

TSJ_DEVICE unsigned long long mix64(unsigned long long z) {
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}

// Gumbel noise -log(-log(u)), u in (0, 1) from 53 hashed bits.
TSJ_DEVICE float gumbel(unsigned long long seed, int pos, int token) {
  const unsigned long long h = mix64(seed ^ mix64((unsigned long long)(unsigned)pos * 0x9e3779b97f4a7c15ull + (unsigned)token));
  const double u = (double(h >> 11) + 0.5) * (1.0 / 9007199254740992.0);
  return float(-log(-log(u)));
}

// "a ranks before b": larger value, then lower index.
TSJ_DEVICE bool before(float av, int ai, float bv, int bi) { return av > bv || (av == bv && ai < bi); }

// Warp argmax (value, index) with the `before` order.
TSJ_DEVICE void warp_best(float& v, int& i) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, v, o);
    const int oi = __shfl_xor_sync(0xffffffffu, i, o);
    if (before(ov, oi, v, i)) v = ov, i = oi;
  }
}

// Block argmax of (v, i) over all threads; every thread gets the result. rv/ri: 32 entries.
TSJ_DEVICE void block_best(float& v, int& i, float* rv, int* ri) {
  warp_best(v, i);
  __syncthreads();
  if ((threadIdx.x & 31) == 0) rv[threadIdx.x >> 5] = v, ri[threadIdx.x >> 5] = i;
  __syncthreads();
  v = rv[0], i = ri[0];
  for (int w = 1; w < int(blockDim.x + 31) / 32; ++w)
    if (before(rv[w], ri[w], v, i)) v = rv[w], i = ri[w];
}

// Stage 1 of the fast path: block (c, n) keeps the top_k of its slice of token n's logits, in
// order, into cand [N][CHUNKS][TOPK_MAX] (value, index); slots past the slice's size get -inf.
// Does nothing unless the fast path applies. grid = (CHUNKS, N), 256 threads.
extern "C" __global__ void __launch_bounds__(256) sample_candidates(const float* __restrict__ logits, int V,
                                                                   const SampleParams* __restrict__ params,
                                                                   float* __restrict__ cv, int* __restrict__ ci) {
  TSJ_PDL_ENTRY;
  const SampleParams P = *params;
  if (!(P.temperature > 0.f) || P.top_k <= 0 || P.top_k > TOPK_MAX) return;
  __shared__ float rv[32];
  __shared__ int ri[32];
  constexpr int PER = 8;  // slice values per thread (CHUNKS * 256 * PER >= V)
  const int n = blockIdx.y, c = blockIdx.x, t = threadIdx.x;
  const int len = (V + CHUNKS - 1) / CHUNKS, i0 = c * len, i1 = min(V, i0 + len);
  const float* x = logits + size_t(n) * V;
  float v[PER];
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    const int i = i0 + t + j * blockDim.x;
    v[j] = i < i1 ? x[i] : -INFINITY;
  }
  float* outv = cv + (size_t(n) * CHUNKS + c) * TOPK_MAX;
  int* outi = ci + (size_t(n) * CHUNKS + c) * TOPK_MAX;
  for (int r = 0; r < P.top_k; ++r) {
    float bv = -INFINITY;
    int bi = 0x7fffffff;
#pragma unroll
    for (int j = 0; j < PER; ++j) {
      const int i = i0 + t + j * blockDim.x;
      if (before(v[j], i, bv, bi)) bv = v[j], bi = i;
    }
    block_best(bv, bi, rv, ri);
    if (t == 0) outv[r] = bv, outi[r] = bi;
#pragma unroll
    for (int j = 0; j < PER; ++j)
      if (i0 + t + j * blockDim.x == bi) v[j] = -INFINITY;  // taken
  }
}

// out[n] for token n (grid = N, 1024 threads), logits [N][V], positions *pos + n; on the fast
// path from the candidates of `sample_candidates`.
extern "C" __global__ void __launch_bounds__(1024) sample_tokens(const float* __restrict__ logits, int V,
                                                                const SampleParams* __restrict__ params,
                                                                const int* __restrict__ pos,
                                                                const float* __restrict__ cv, const int* __restrict__ ci,
                                                                int* __restrict__ out) {
  TSJ_PDL_ENTRY;
  __shared__ float red[32];
  __shared__ float rv[32];
  __shared__ int ri[32];
  __shared__ float kv[TOPK_MAX];
  __shared__ int ki[TOPK_MAX];
  const SampleParams P = *params;
  const int n = blockIdx.x, t = threadIdx.x;
  const float* x = logits + size_t(n) * V;
  const int p = *pos + n;

  if (P.temperature > 0.f && P.top_k > 0 && P.top_k <= TOPK_MAX) {
    // Global top_k from the CHUNKS * TOPK_MAX candidates: repeated block argmax (in order).
    const int nc = CHUNKS * TOPK_MAX;
    const float* v = cv + size_t(n) * nc;
    const int* id = ci + size_t(n) * nc;
    // Each thread holds nc / 1024 = 8 candidates; taken ones become -inf.
    float c[8];
    int cidx[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int s = t + j * blockDim.x;
      const int b = s / TOPK_MAX, r = s % TOPK_MAX;
      c[j] = r < P.top_k ? v[s] : -INFINITY;
      cidx[j] = r < P.top_k ? id[s] : 0x7fffffff;
      (void)b;
    }
    for (int r = 0; r < P.top_k; ++r) {
      float bv = -INFINITY;
      int bi = 0x7fffffff;
#pragma unroll
      for (int j = 0; j < 8; ++j)
        if (before(c[j], cidx[j], bv, bi)) bv = c[j], bi = cidx[j];
      block_best(bv, bi, rv, ri);
      if (t == 0) kv[r] = bv, ki[r] = bi;
#pragma unroll
      for (int j = 0; j < 8; ++j)
        if (cidx[j] == bi) c[j] = -INFINITY, cidx[j] = 0x7fffffff;
    }
    __syncthreads();
    if (t < 32) {
      // kv is sorted descending: softmax with temperature, top_p cut, Gumbel-max (one warp).
      const float inv_t = 1.f / P.temperature, m = kv[0];
      const int k = P.top_k;
      float w[2];
      float z = 0.f;
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int r = t + 32 * j;
        w[j] = r < k ? expf((kv[r] - m) * inv_t) : 0.f;
        z += w[j];
      }
      z = warp_sum(z);
      // Keep rank r while the mass of the ranks before it is < top_p (the crossing token stays).
      int keep = k;
      if (P.top_p < 1.f) {
        float before_mass = 0.f;
        keep = 0;
        for (int r = 0; r < k; ++r) {
          if (before_mass >= P.top_p * z) break;
          const float wr = __shfl_sync(0xffffffffu, w[r / 32], r % 32);
          before_mass += wr;
          keep = r + 1;
        }
      }
      float bv = -INFINITY;
      int bi = 0x7fffffff;
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int r = t + 32 * j;
        if (r < keep) {
          const float s = (kv[r] - m) * inv_t + gumbel(P.seed, p, ki[r]);
          if (before(s, ki[r], bv, bi)) bv = s, bi = ki[r];
        }
      }
      warp_best(bv, bi);
      if (t == 0) out[n] = bi;
    }
    return;
  }

  // Full scan. Greedy (and the max for the softmax): lowest index on ties, as `argmax`.
  float bv = -INFINITY;
  int bi = 0x7fffffff;
  for (int i = t; i < V; i += blockDim.x)
    if (x[i] > bv) bv = x[i], bi = i;
  block_best(bv, bi, rv, ri);
  const float m = bv;
  if (!(P.temperature > 0.f)) {
    if (t == 0) out[n] = bi;
    return;
  }
  const float inv_t = 1.f / P.temperature;

  // top_k: the largest key thr with count(key >= thr) >= k (ties at the k-th value stay).
  unsigned thr = 0;
  if (P.top_k > 0 && P.top_k < V) {
    unsigned lo = 0, hi = 0xffffffffu;  // count(>= lo) >= k > count(>= hi + 1)
    while (lo < hi) {
      const unsigned mid = lo + unsigned((1ull + hi - lo) / 2);
      float c = 0.f;
      for (int i = t; i < V; i += blockDim.x) c += order_key(x[i]) >= mid;
      if (block_sum(c, red) >= float(P.top_k)) lo = mid; else hi = mid - 1;
    }
    thr = lo;
  }
  // top_p: the largest key thr2 >= thr whose kept mass reaches top_p of the top_k set's mass.
  if (P.top_p < 1.f) {
    float z = 0.f;
    for (int i = t; i < V; i += blockDim.x)
      if (order_key(x[i]) >= thr) z += expf((x[i] - m) * inv_t);
    const float target = P.top_p * block_sum(z, red);
    unsigned lo = thr, hi = order_key(m);  // mass(>= lo) >= target; the max itself is always kept
    while (lo < hi) {
      const unsigned mid = lo + unsigned((1ull + hi - lo) / 2);
      float s = 0.f;
      for (int i = t; i < V; i += blockDim.x)
        if (order_key(x[i]) >= mid) s += expf((x[i] - m) * inv_t);
      if (block_sum(s, red) >= target) lo = mid; else hi = mid - 1;
    }
    thr = lo;
  }

  // Gumbel-max over the kept tokens.
  bv = -INFINITY;
  bi = 0x7fffffff;
  for (int i = t; i < V; i += blockDim.x) {
    if (order_key(x[i]) < thr) continue;
    const float s = (x[i] - m) * inv_t + gumbel(P.seed, p, i);
    if (before(s, i, bv, bi)) bv = s, bi = i;
  }
  block_best(bv, bi, rv, ri);
  if (t == 0) out[n] = bi;
}
