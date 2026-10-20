// Decode element-wise and small kernels for N tokens per step: step bookkeeping, embedding, GR
// write-back, router, MoE combine, greedy argmax. All fp32.
//
// Layouts: residual R [N][HC][H]. Every kernel that writes it also leaves per-stream
// sum-of-squares partials ss[n][block][HC] (one row per block along x, fixed grid, so the order is
// deterministic); the next GR read-out sums them for its RMSNorm instead of running a norm kernel.
// Per-token work never depends on N, so a window of N tokens computes the same bits as N steps.
#include "decode.cuh"
#include "rank1.cuh"

// Starts a step of N tokens on the device, so a captured graph can be replayed step after step:
// tok3[n] = [token, t1, t2] for the embedding and the PLE hash (the n-gram context runs through the
// window from the committed one; EOS resets it), pos = the position of token 0, par = this step's
// parity (state buffers alternate per step: a step reads [par] and writes [par ^ 1]).
// ctx = [t1, t2, next position, parity]; `accept` advances the first three at the end of the step.
extern "C" __global__ void begin_step(const int* __restrict__ next, int* __restrict__ ctx, int* __restrict__ tok3,
                                      int* __restrict__ pos, int* __restrict__ par, int eos, int N) {
  TSJ_PDL_ENTRY;
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    int t1 = ctx[0], t2 = ctx[1];
    for (int n = 0; n < N; ++n) {
      const int t = next[n];
      tok3[3 * n] = t;
      tok3[3 * n + 1] = t1;
      tok3[3 * n + 2] = t2;
      t2 = t == eos ? eos : t1;
      t1 = t;
    }
    *pos = ctx[2];
    *par = ctx[3];
    ctx[3] ^= 1;
  }
}

// Ends a step: how many of the window's tokens are committed. The window is next[0..N): next[0]
// the committed input, next[1..N) drafts. With `verify`, draft j is accepted while every draft
// before it was and it equals the greedy token after token j - 1 (out[j - 1]); otherwise all N
// tokens are committed (prompt chunks, plain decode). With a accepted drafts, a + 1 tokens are
// committed: the PLE context and the position advance past them, *napply = a + 1 tells the next
// step's state kernels how many of this window's records to apply (docs/m3.md §4.1), and next[0] =
// out[a] becomes the next step's committed input.
// For the MTP layer: mtp_tok[t] = the token after window token t (the MTP row of position p + t
// pairs h_{p+t} with it): the window's next token for t < a, out[a] for t = a (or, committing a
// whole prompt chunk, the lookahead next[N] when the host gave one, >= 0), anything after;
// chain_pos[i] = the position of draft step i (the next position + i). acc_out = [a + 1, the
// committed tokens] for the host.
extern "C" __global__ void accept(int* __restrict__ next, const int* __restrict__ out, int* __restrict__ ctx,
                                  int* __restrict__ napply, int* __restrict__ mtp_tok, int* __restrict__ chain_pos,
                                  int* __restrict__ acc_out, int eos, int N, int verify) {
  TSJ_PDL_ENTRY;
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    int a = N - 1;
    if (verify) {
      a = 0;
      while (a < N - 1 && next[a + 1] == out[a]) ++a;
    }
    for (int t = 0; t < N; ++t)
      mtp_tok[t] = t < a ? next[t + 1] : t > a ? out[t] : !verify && next[N] >= 0 ? next[N] : out[a];
    int t1 = ctx[0], t2 = ctx[1];
    for (int n = 0; n <= a; ++n) {
      const int t = next[n];
      t2 = t == eos ? eos : t1;
      t1 = t;
    }
    ctx[0] = t1;
    ctx[1] = t2;
    acc_out[0] = a + 1;
    for (int n = 0; n <= a; ++n) acc_out[1 + n] = next[n];
    ctx[2] += a + 1;
    for (int i = 0; i < 8; ++i) chain_pos[i] = ctx[2] + i;
    *napply = a + 1;
    next[0] = out[a];
  }
}

// Writes stream sums of squares of this block's part of token n's R: ss[n][blockIdx.x][i].
// `vals[i]` is this thread's new R[n][i][k] (0 when k >= H). blockDim = 256.
TSJ_DEVICE void write_stats(const float (&vals)[4], int HC, int n, float* __restrict__ ss, float* red) {
  for (int i = 0; i < HC; ++i) {
    const float sq = block_sum(vals[i] * vals[i], red);
    if (threadIdx.x == 0) ss[(size_t(n) * gridDim.x + blockIdx.x) * HC + i] = sq;
  }
}

// inj[i] = 2 sigmoid(sum_s di[s][n][rank + i] / HC) from the GR down partials, into shared memory.
TSJ_DEVICE void load_inject(const float* __restrict__ di, int S, int N, int n, int ld, int rank, int HC,
                            float* inj) {
  if (threadIdx.x < HC) {
    float a = 0.f;
    for (int s = 0; s < S; ++s) a += di[size_t(s * N + n) * ld + rank + threadIdx.x];
    inj[threadIdx.x] = 2.f * sigmoidf(a / float(HC));
  }
  __syncthreads();
}

// R[n][i][k] = embed[token n][k] for every stream i, plus stream statistics.
// grid = (ceil(H / 256), N), 256 threads, HC <= 4.
extern "C" __global__ void __launch_bounds__(256) embed_streams(const __nv_bfloat16* __restrict__ table,
                                                               const int* __restrict__ tok3, float* __restrict__ R,
                                                               float* __restrict__ ss, int H, int HC) {
  TSJ_PDL_ENTRY;
  __shared__ float red[32];
  const int n = blockIdx.y, k = blockIdx.x * blockDim.x + threadIdx.x;
  const float v = k < H ? bf16_to_f(table[size_t(tok3[3 * n]) * H + k]) : 0.f;
  float vals[4] = {};
  for (int i = 0; i < HC; ++i) {
    if (k < H) R[(size_t(n) * HC + i) * H + k] = v;
    vals[i] = v;
  }
  write_stats(vals, HC, n, ss, red);
}

// GR write-back of a mixer: R[n][i][k] += inj_i out[n][k], out = sum_s op[s][n][k] (So split-K
// partials of the output projection) + the projection's rank-1 adapter s (a . x[n]) b[k] (x [N][K]
// the projection's input), inj from the GR down partials; plus stream statistics.
// grid = (ceil(H / 256), N), 256 threads.
extern "C" __global__ void __launch_bounds__(256) inject_stats(float* __restrict__ R, const float* __restrict__ di,
                                                              int Sd, int ld, int rank, const float* __restrict__ op,
                                                              int So, float* __restrict__ ss, int H, int HC,
                                                              const float* __restrict__ x, int K, const Rank1 ad) {
  TSJ_PDL_ENTRY;
  __shared__ float red[32];
  __shared__ float inj[8];
  const int n = blockIdx.y, N = gridDim.y;
  load_inject(di, Sd, N, n, ld, rank, HC, inj);
  float lora = 0.f;
  if (ad.a != nullptr && *ad.scale != 0.f) {
    float p = 0.f;
    for (int i = threadIdx.x; i < K; i += blockDim.x) p = fmaf(ad.a[i], x[size_t(n) * K + i], p);
    lora = *ad.scale * block_sum(p, red);
  }
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  float vals[4] = {};
  if (k < H) {
    float o = 0.f;
    for (int s = 0; s < So; ++s) o += op[size_t(s * N + n) * H + k];
    if (lora != 0.f) o = fmaf(lora, ad.b[k], o);
    float* r = R + size_t(n) * HC * H + k;
    for (int i = 0; i < HC; ++i) {
      vals[i] = r[i * H] + inj[i] * o;
      r[i * H] = vals[i];
    }
  }
  write_stats(vals, HC, n, ss, red);
}

// Router: softmax over E logits (fp32), top-k (ties -> lower expert id), renormalized weights;
// shared-expert gate sigmoid(logits[E]). One warp per token (grid = N); each lane keeps E/32
// probabilities in registers. logits [N][E + 1], sel and w [N][topk], shared_gate [N].
extern "C" __global__ void __launch_bounds__(32) router_topk(const float* __restrict__ logits, int E, int topk,
                                                             int* __restrict__ sel, float* __restrict__ w,
                                                             float* __restrict__ shared_gate) {
  TSJ_PDL_ENTRY;
  constexpr int MAXV = 32;  // E <= 1024
  const int lane = threadIdx.x, n = blockIdx.x;
  logits += size_t(n) * (E + 1);
  sel += n * topk;
  w += n * topk;
  float v[MAXV];
  float m = -INFINITY;
#pragma unroll
  for (int i = 0; i < MAXV; ++i) {
    const int e = lane + 32 * i;
    v[i] = e < E ? logits[e] : -INFINITY;
    m = fmaxf(m, v[i]);
  }
  m = warp_max(m);
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < MAXV; ++i) {
    v[i] = lane + 32 * i < E ? expf(v[i] - m) : -1.f;  // -1: not a candidate
    s += fmaxf(v[i], 0.f);
  }
  s = warp_sum(s);
  float total = 0.f, chosen = 0.f;
  for (int t = 0; t < topk; ++t) {
    float bv = -1.f;
    int bi = 0x7fffffff;
#pragma unroll
    for (int i = 0; i < MAXV; ++i)
      if (v[i] > bv) bv = v[i], bi = lane + 32 * i;  // ascending index within a lane: first max wins
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
      const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
      if (ov > bv || (ov == bv && oi < bi)) bv = ov, bi = oi;
    }
    if ((bi & 31) == lane) {
#pragma unroll
      for (int i = 0; i < MAXV; ++i)
        if (lane + 32 * i == bi) v[i] = -2.f;  // taken
    }
    if (lane == t) chosen = bv / s;
    if (lane == 0) sel[t] = bi;
    total += bv / s;
  }
  if (lane < topk) w[lane] = chosen / total;  // norm_topk_prob
  if (lane == 0) shared_gate[n] = sigmoidf(logits[E]);
}

// MoE output and GR write-back: out[n][k] = sum_s w[n][s] y[n][s][k] + shared_gate[n] y[n][topk][k]
// (slot topk is the shared expert); R[n][i][k] += inj_i out[n][k]; plus stream statistics.
// y [N][topk + 1][H]. grid = (ceil(H / 256), N), 256 threads.
extern "C" __global__ void __launch_bounds__(256) moe_combine(float* __restrict__ R, const float* __restrict__ di,
                                                             int Sd, int ld, int rank, const float* __restrict__ y,
                                                             const float* __restrict__ w,
                                                             const float* __restrict__ shared_gate,
                                                             float* __restrict__ ss, int H, int HC, int topk) {
  TSJ_PDL_ENTRY;
  __shared__ float red[32];
  __shared__ float inj[8];
  const int n = blockIdx.y, N = gridDim.y;
  load_inject(di, Sd, N, n, ld, rank, HC, inj);
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  float vals[4] = {};
  if (k < H) {
    const float* yn = y + size_t(n) * (topk + 1) * H + k;
    const float* wn = w + n * topk;
    float o = 0.f;
    for (int s = 0; s < topk; ++s) o = fmaf(wn[s], yn[s * H], o);
    o = fmaf(shared_gate[n], yn[topk * H], o);
    float* r = R + size_t(n) * HC * H + k;
    for (int i = 0; i < HC; ++i) {
      vals[i] = r[i * H] + inj[i] * o;
      r[i * H] = vals[i];
    }
  }
  write_stats(vals, HC, n, ss, red);
}

// Greedy: for each token n (grid = N), the index of the largest of x[n][0..len) (lowest index on
// ties) into out[n]. 1024 threads.
extern "C" __global__ void __launch_bounds__(1024) argmax(const float* __restrict__ x, int len, int* __restrict__ out) {
  TSJ_PDL_ENTRY;
  __shared__ float rv[32];
  __shared__ int ri[32];
  x += size_t(blockIdx.x) * len;
  float bv = -INFINITY;
  int bi = 0x7fffffff;
  for (int i = threadIdx.x; i < len; i += blockDim.x)
    if (x[i] > bv) bv = x[i], bi = i;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
    const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
    if (ov > bv || (ov == bv && oi < bi)) bv = ov, bi = oi;
  }
  if ((threadIdx.x & 31) == 0) rv[threadIdx.x >> 5] = bv, ri[threadIdx.x >> 5] = bi;
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int i = 1; i < 32; ++i)
      if (rv[i] > rv[0] || (rv[i] == rv[0] && ri[i] < ri[0])) rv[0] = rv[i], ri[0] = ri[i];
    out[blockIdx.x] = ri[0];
  }
}
