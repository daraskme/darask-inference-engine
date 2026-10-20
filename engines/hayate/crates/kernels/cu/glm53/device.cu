// GLM-5.3 device decode (one token, graph-capturable). Every kernel keeps the
// arithmetic and reduction order of the per-op decode kernels it replaces
// (ops.cu, decode.cu), so logits stay bit-identical; only launches, host
// round trips and redundant passes are removed. Position, sequence numbers and
// the expert pointer table live in device memory.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include "dense.cuh"
namespace {
using namespace glm53;
__device__ __forceinline__ float f(B x) { return __bfloat162float(x); }
__device__ __forceinline__ B b(float x) { return __float2bfloat16_rn(x); }
__device__ __forceinline__ float rnd(float x) { return f(b(x)); }
// Same tree as ops.cu `reduce` (blockDim.x a power of two).
__device__ float reduce(float x, float* s) {
  __syncthreads(); s[threadIdx.x]=x; __syncthreads();
  for (unsigned i=blockDim.x/2;i;i/=2) { if(threadIdx.x<i)s[threadIdx.x]+=s[threadIdx.x+i]; __syncthreads(); }
  return s[0];
}
// Device control block written once per token (H2D) before the layers run.
struct Ctl { unsigned token, position, seq, error, next; };
// Last block to arrive returns true (counter reset for the next launch).
__device__ bool last_block(unsigned* counter) {
  __shared__ bool last;
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    last = atomicAdd(counter, 1u) == gridDim.x - 1;
    if (last) *counter = 0;
  }
  __syncthreads();
  return last;
}

// mHC map (24 logits) + collapse + RMSNorm. Block j computes logit j exactly
// as hc_map does; the last block runs hc_weights and hc_collapse_norm.
__global__ void __launch_bounds__(256) hc_norm(float* map, B* collapsed, B* out, const B* x, const B* fn,
    const float* base, const float* scale, const B* w, float* logits, unsigned* counter) {
  __shared__ float s[256], lg[24], mp[24];
  const unsigned cols = 4 * 4096, j = blockIdx.x;
  float sq = 0;
  for (unsigned i = threadIdx.x; i < cols; i += 256) { float v = f(x[i]); sq += v * v; }
  float inv = rsqrtf(reduce(sq, s) / cols + 1.e-5f);
  float sum = 0;
  for (unsigned i = threadIdx.x; i < cols; i += 256) sum += f(x[i]) * inv * f(fn[size_t(j) * cols + i]);
  sum = reduce(sum, s);
  if (threadIdx.x == 0) logits[j] = sum;
  if (!last_block(counter)) return;
  if (threadIdx.x < 24) lg[threadIdx.x] = __ldcg(logits + threadIdx.x);
  __syncthreads();
  if (threadIdx.x == 0) hc_weights(mp, lg, base, scale);
  __syncthreads();
  if (threadIdx.x < 24) map[threadIdx.x] = mp[threadIdx.x];
  const unsigned hidden = 4096;
  sq = 0;
  for (unsigned c = threadIdx.x; c < hidden; c += 256) {
    float v = 0; for (unsigned h = 0; h < 4; ++h) v += mp[h] * f(x[h * hidden + c]);
    collapsed[c] = b(v); sq += rnd(v) * rnd(v);
  }
  inv = rsqrtf(reduce(sq, s) / hidden + 1.e-5f);
  for (unsigned c = threadIdx.x; c < hidden; c += 256) out[c] = b(rnd(f(collapsed[c]) * inv) * f(w[c]));
}

// Several GEMVs sharing one BF16 input (rows of each segment, 8 per block).
struct Segment { DenseWeight w; B* out; unsigned rows, blocks; };
struct Segments { Segment s[6]; unsigned count; };
__global__ void __launch_bounds__(256) gemv_multi(Segments m, const B* x, unsigned k) {
  unsigned block = blockIdx.x, i = 0;
  while (i + 1 < m.count && block >= m.s[i].blocks) { block -= m.s[i].blocks; ++i; }
  const Segment& seg = m.s[i];
  unsigned row = block * 8 + threadIdx.x / 32;
  if (row >= seg.rows) return;
  float value = dot(x, seg.w, row, seg.rows, k);
  if (threadIdx.x % 32 == 0) seg.out[row] = b(value);
}

// KDA step for one head (block h, 256 threads; column j < 128): short
// convolutions, f_b/g_b projections of the 128 head rows (warp per row, dense
// dot), decay, beta sigmoid, recurrence and gated norm. Same arithmetic as
// conv_step, dense, decay, point(0) and kda_gated (the upper 128 threads add
// exact zeros to the 128-wide reductions).
__global__ void __launch_bounds__(256) kda_step(B* core, B* gated, float* state, B* hq, B* hk, B* hv,
    const B* q, const B* k, const B* v, const B* wq, const B* wk, const B* wv, const B* fa, DenseWeight fb,
    const B* ga, DenseWeight gb, const float* dt_bias, const float* alog, const B* beta_raw, const B* norm) {
  __shared__ float s[256], qs[128], ks[128], decay[128], gate[128];
  __shared__ float conv[3][128];
  const unsigned h = blockIdx.x, j = threadIdx.x, c = h * 128 + j;
  B* history[3] = {hq, hk, hv};
  const B* xs[3] = {q, k, v};
  const B* ws[3] = {wq, wk, wv};
  for (unsigned part = 0; part < 3 && j < 128; ++part) {
    float past[3], weight[4];
    for (unsigned t = 0; t < 3; ++t) past[t] = f(history[part][t * 8192 + c]);
    for (unsigned t = 0; t < 4; ++t) weight[t] = f(ws[part][c * 4 + t]);
    float sum = conv_sum(past, f(xs[part][c]), weight);
    for (unsigned t = 0; t < 2; ++t) history[part][t * 8192 + c] = history[part][(t + 1) * 8192 + c];
    history[part][2 * 8192 + c] = xs[part][c];
    sum = rnd(sum);
    conv[part][j] = f(b(sum * sigmoid(sum)));
  }
  const unsigned warp = j / 32, lane = j % 32;
  for (unsigned r = warp; r < 128; r += 8) {
    float a = dot(fa, fb, h * 128 + r, 8192, 128);
    float g = dot(ga, gb, h * 128 + r, 8192, 128);
    if (lane == 0) {
      decay[r] = forget(f(b(a)), dt_bias[h * 128 + r], alog[h]);
      gate[r] = f(b(g));
    }
  }
  float beta = f(b(sigmoid(f(beta_raw[h]))));
  __syncthreads();
  const bool col = j < 128;
  float qv = col ? conv[0][j] : 0.f, kv = col ? conv[1][j] : 0.f;
  float nq = reduce(qv * qv, s), nk = reduce(kv * kv, s);
  if (col) { qs[j] = qv / sqrtf(nq + 1.e-6f); ks[j] = kv / sqrtf(nk + 1.e-6f); }
  __syncthreads();
  float value = col ? rnd(kda_column(state + size_t(h) * 128 * 128, qs, ks, conv[2][j], decay, beta, 128, j)) : 0.f;
  float inv = rsqrtf(reduce(value * value, s) / 128 + 1.e-5f);
  if (!col) return;
  core[c] = b(value);
  gated[c] = b(value * inv * f(norm[j]) * sigmoid(gate[j]));
}

// ops.cu norm() for one row; blockDim 256.
__device__ void norm_row(B* out, const B* x, const B* w, const B* bias, unsigned cols, int mode, float* s) {
  float mean = 0, sq = 0;
  if (mode == 1) { for (unsigned i = threadIdx.x; i < cols; i += 256) mean += f(x[i]); mean = reduce(mean, s) / cols; }
  for (unsigned i = threadIdx.x; i < cols; i += 256) { float v = f(x[i]) - mean; sq += v * v; }
  float inv = rsqrtf(reduce(sq, s) / cols + (mode == 1 ? 1.e-6f : 1.e-5f));
  for (unsigned i = threadIdx.x; i < cols; i += 256) {
    float v = (f(x[i]) - mean) * inv;
    out[i] = mode == 1 ? b(v * f(w[i]) + f(bias[i])) : b(rnd(v) * f(w[i]));
  }
}
// Sparse layer bookkeeping at the device position: q_a norm (block 0), latent
// row (block 1), indexer key norm, pool tail and the completed pool (block 2).
__global__ void __launch_bounds__(256) sparse_prep(B* qr, const B* qa, const B* qa_norm, B* kvr, const B* kva,
    const B* kv_norm, B* latent, B* ikn, const B* ik, const B* kn_w, const B* kn_b, const B* ig, B* tail_keys,
    B* tail_gates, B* pools, const B* ape, const Ctl* ctl) {
  __shared__ float s[256];
  const unsigned pos = ctl->position;
  if (blockIdx.x == 0) { norm_row(qr, qa, qa_norm, nullptr, 1536, 0, s); return; }
  if (blockIdx.x == 1) {
    norm_row(kvr, kva, kv_norm, nullptr, 512, 0, s);
    __syncthreads();
    for (unsigned i = threadIdx.x; i < 512; i += 256) latent[size_t(pos) * 512 + i] = kvr[i];
    return;
  }
  norm_row(ikn, ik, kn_w, kn_b, 128, 1, s);
  __syncthreads();
  const unsigned c = threadIdx.x, tail = pos % 4;
  if (c < 128) { tail_keys[tail * 128 + c] = ikn[c]; tail_gates[tail * 128 + c] = ig[c]; }
  __syncthreads();
  if (tail != 3 || c >= 128) return;
  float m = -INFINITY, den = 0, lg[4];
  for (unsigned j = 0; j < 4; ++j) { lg[j] = f(tail_gates[j * 128 + c]) + f(ape[j * 128 + c]); m = fmaxf(m, lg[j]); }
  for (unsigned j = 0; j < 4; ++j) { lg[j] = expf(lg[j] - m); den += lg[j]; }
  float sum = 0; for (unsigned j = 0; j < 4; ++j) sum += rnd(rnd(lg[j] / den) * f(tail_keys[j * 128 + c]));
  pools[size_t(pos / 4) * 128 + c] = b(sum);
}

// Mailbox layout (async_decode.rs): READY 0, DONE 64, IDS 128, INPUT 256,
// OUTPUT INPUT + 4 * 8192 (slot a of row 0 at a * 8192 bytes).
// DONE2 (96) is rung after the deferred CPU experts of the layer.
constexpr unsigned MB_READY = 0, MB_DONE = 64, MB_DONE2 = 96, MB_IDS = 128, MB_INPUT = 256, MB_OUTPUT = 256 + 4 * 8192;
// Per-layer CPU-expert plan written by the router: counts of the experts the
// layer waits for (now) and the next layer waits for (later), and the masks of
// dropped (skip) and deferred experts.
struct Plan { unsigned now, later, skip, defer; };
// Router: 288 logits (warp per expert, same dot as router_project); the last
// block selects top-8 as router_select, resolves images from the device
// table, publishes ids + input to the CPU mailbox and rings READY = seq.
// The PCIe lane (`host` non-null): of the n chosen experts that miss the GPU and have a RAM-tier image (host[e],
// pinned), the first take[n] in route order stay on the CPU; the others are copied by the fetch kernel (exl3.cu)
// from host[e] into scratch[s] (src[s] = host[e]), computed by the layer's expert kernels there (images[j] =
// scratch[s]) and marked for the CPU worker to leave out (mailbox bit 31). take[n] is nibble n of `take`; at
// most 4 experts a layer take the lane. src[s] is null for unused slots.
// Prefetch (`pre_src` non-null): the previous MoE layer's `predict` picked pre_src[s] and `prefetch` copies them
// into pre_scratch[s] on a side stream, nobody waiting for it. A miss whose host image is one of them takes that
// slot only when the copy has finished for this token and layer (pre_flags[s] == tag, tag = seq << 8 | layer);
// the rest go through take[] above. The router publishes its tag in `progress` first, so a copy still pending
// for this layer stops (`prefetch`). `stats` (optional, u32 [4]): prefetched (by `predict`), prefetch used,
// serial lane, RAM-tier misses left to the CPU.
struct Pcie { const void* const* host; const void* const* scratch; const void** src; unsigned long long take;
              const void* const* pre_src; const void* const* pre_scratch; const volatile unsigned long long* pre_flags;
              unsigned long long* progress; unsigned layer; unsigned* stats; };
__global__ void __launch_bounds__(256) route(float* logits, float* weights, int* ids, const void** images,
    Plan* plan, const B* x, const B* w, const float* bias, const void* const* table, unsigned char* mailbox,
    const Ctl* ctl, unsigned* counter, float skip_below, float defer_below,
    const unsigned long long* format_table, unsigned long long* formats, Pcie pcie) {
  __shared__ float scores[288], choices[288], chosen_weight[8];
  __shared__ int chosen[8];
  __shared__ unsigned skip, defer, lane_mask;
  unsigned row = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x % 32;
  float value = dot(x, {w, nullptr, 0}, row, 288, 4096);
  if (lane == 0) logits[row] = value;
  if (!last_block(counter)) return;
  for (unsigned e = threadIdx.x; e < 288; e += 256) {
    scores[e] = sigmoid(__ldcg(logits + e));
    choices[e] = scores[e] + bias[e];
  }
  __syncthreads();
  if (threadIdx.x < 32) {
    for (unsigned j = 0; j < 8; ++j) {
      float best = -INFINITY; int arg = -1;
      for (unsigned e = lane; e < 288; e += 32)
        if (arg < 0 || choices[e] > best) { best = choices[e]; arg = e; }
      for (unsigned o = 16; o; o /= 2) {
        float ob = __shfl_xor_sync(~0u, best, o); int oa = __shfl_xor_sync(~0u, arg, o);
        if (ob > best || (ob == best && oa < arg)) { best = ob; arg = oa; }
      }
      if (lane == 0) { chosen[j] = arg; choices[arg] = -INFINITY; }
      __syncwarp();
    }
    if (lane == 0) {
      float sum = 0, wt[8];
      for (unsigned j = 0; j < 8; ++j) { wt[j] = scores[chosen[j]]; sum += wt[j]; }
      for (unsigned j = 0; j < 8; ++j) chosen_weight[j] = weights[j] = wt[j] / (sum + 1.e-20f) * 2.5f;
    }
  }
  __syncthreads();
  __shared__ const void* image[8];
  if (threadIdx.x < 8) {
    int e = chosen[threadIdx.x];
    ids[threadIdx.x] = e;
    image[threadIdx.x] = images[threadIdx.x] = table[e];
    // LOWBIT layers: the 8-byte ExpertFormat of each chosen expert, in route order.
    if (format_table) formats[threadIdx.x] = format_table[e];
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned mask = 0;
    if (pcie.host) {
      unsigned used = 0;
      const unsigned long long tag = (unsigned long long)ctl->seq << 8 | pcie.layer;
      if (pcie.progress) {
        *reinterpret_cast<volatile unsigned long long*>(pcie.progress) = tag;
        __threadfence();
      }
      if (pcie.pre_src)
        for (unsigned j = 0; j < 8; ++j) {
          const void* h = image[j] ? nullptr : pcie.host[chosen[j]];
          for (unsigned s = 0; h && s < 8; ++s)
            if (pcie.pre_src[s] == h && pcie.pre_flags[s] == tag) {
              image[j] = images[j] = pcie.pre_scratch[s];
              mask |= 1u << j;
              ++used;
              break;
            }
        }
      unsigned n = 0;
      for (unsigned j = 0; j < 8; ++j) n += !image[j] && pcie.host[chosen[j]];
      const unsigned cpu = unsigned(pcie.take >> (4 * n) & 15);
      unsigned seen = 0, slot = 0;
      for (unsigned j = 0; j < 8; ++j) {
        const void* h = image[j] ? nullptr : pcie.host[chosen[j]];
        if (!h || seen++ < cpu || slot == 4) continue;
        pcie.src[slot] = h;
        image[j] = images[j] = pcie.scratch[slot++];
        mask |= 1u << j;
      }
      if (pcie.stats) { pcie.stats[1] += used; pcie.stats[2] += slot; pcie.stats[3] += n - slot; }
      for (; slot < 4; ++slot) pcie.src[slot] = nullptr;
    }
    lane_mask = mask;
  }
  if (threadIdx.x == 0) {
    // CPU experts below `skip_below` are dropped, below `defer_below` join the
    // next layer (both approximations; 0 disables).
    Plan p{};
    for (unsigned j = 0; j < 8; ++j) {
      if (image[j]) continue;
      if (chosen_weight[j] < skip_below) p.skip |= 1u << j;
      else if (chosen_weight[j] < defer_below) { p.defer |= 1u << j; ++p.later; }
      else ++p.now;
    }
    *plan = p; skip = p.skip; defer = p.defer;
  }
  __syncthreads();
  if (!mailbox) return;
  if (threadIdx.x < 8)
    reinterpret_cast<unsigned*>(mailbox + MB_IDS)[threadIdx.x] =
        unsigned(chosen[threadIdx.x]) | ((skip | lane_mask) >> threadIdx.x & 1u) << 31 |
        (defer >> threadIdx.x & 1u) << 30;
  for (unsigned i = threadIdx.x; i < 4096 * 2 / 16; i += 256)
    reinterpret_cast<uint4*>(mailbox + MB_INPUT)[i] = reinterpret_cast<const uint4*>(x)[i];
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) *reinterpret_cast<volatile unsigned*>(mailbox + MB_READY) = ctl->seq;
}
// Prefetch pick (Pcie above): the next MoE layer's router on this layer's normalized input, the same top-8 as
// `route`; the predicted experts outside VRAM with a RAM-tier image, in score order, fill src[0..slots).
__global__ void __launch_bounds__(256) predict(float* logits, const B* x, const B* w, const float* bias,
    const void* const* table, const void* const* host, const void** src, unsigned slots, unsigned* counter,
    unsigned* stats) {
  __shared__ float choices[288];
  unsigned row = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x % 32;
  float value = dot(x, {w, nullptr, 0}, row, 288, 4096);
  if (lane == 0) logits[row] = value;
  if (!last_block(counter)) return;
  for (unsigned e = threadIdx.x; e < 288; e += 256) choices[e] = sigmoid(__ldcg(logits + e)) + bias[e];
  __syncthreads();
  if (threadIdx.x >= 32) return;
  unsigned taken = 0;
  for (unsigned j = 0; j < 8; ++j) {
    float best = -INFINITY; int arg = -1;
    for (unsigned e = lane; e < 288; e += 32)
      if (arg < 0 || choices[e] > best) { best = choices[e]; arg = e; }
    for (unsigned o = 16; o; o /= 2) {
      float ob = __shfl_xor_sync(~0u, best, o); int oa = __shfl_xor_sync(~0u, arg, o);
      if (ob > best || (ob == best && oa < arg)) { best = ob; arg = oa; }
    }
    if (lane == 0) {
      choices[arg] = -INFINITY;
      if (taken < slots && !table[arg] && host[arg]) src[taken++] = host[arg];
    }
    __syncwarp();
  }
  if (lane == 0) {
    if (stats) stats[0] += taken;
    for (; taken < 8; ++taken) src[taken] = nullptr;
  }
}
// The prefetch copy (Pcie above): slot blockIdx.y of dst from src (null: none), grid (blocks, 8). The blocks of a
// slot count in done[s]; the last one sets flags[s] = tag unless one of them found the target layer already
// routed (*progress >= tag) and skipped its part (bad[s]).
__global__ void __launch_bounds__(256) prefetch(void* const* dst, const void* const* src, size_t bytes,
    unsigned* done, unsigned* bad, unsigned long long* flags, const volatile unsigned long long* progress,
    const Ctl* ctl, unsigned layer) {
  const unsigned s = blockIdx.y;
  const uint4* from = static_cast<const uint4*>(src[s]);
  if (!from) return;
  const unsigned long long tag = (unsigned long long)ctl->seq << 8 | layer;
  __shared__ bool stale;
  if (threadIdx.x == 0) stale = *progress >= tag;
  __syncthreads();
  if (!stale) {
    uint4* to = static_cast<uint4*>(dst[s]);
    const size_t n = bytes / 16, step = size_t(gridDim.x) * 256;
    size_t i = size_t(blockIdx.x) * 256 + threadIdx.x;
    for (; i + 3 * step < n; i += 4 * step) {
      const uint4 a = from[i], b = from[i + step], c = from[i + 2 * step], e = from[i + 3 * step];
      to[i] = a; to[i + step] = b; to[i + 2 * step] = c; to[i + 3 * step] = e;
    }
    for (; i < n; i += step) to[i] = from[i];
  } else if (threadIdx.x == 0) {
    atomicOr(bad + s, 1u);
  }
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0 && atomicAdd(done + s, 1u) == gridDim.x - 1) {
    __threadfence();
    if (!atomicExch(bad + s, 0u)) *reinterpret_cast<volatile unsigned long long*>(flags + s) = tag;
    done[s] = 0;
  }
}
// Waits for the CPU misses of this layer (only when there are any). A stuck
// worker cannot hang the GPU forever: after `limit` ns the error flag is set.
__global__ void moe_wait(const unsigned* count, const unsigned char* mailbox, unsigned flag, Ctl* ctl,
    unsigned long long limit) {
  if (threadIdx.x || !*count) return;
  const unsigned seq = ctl->seq;
  auto done = reinterpret_cast<const volatile unsigned*>(mailbox + flag);
  unsigned long long start, now;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
  while (*done != seq) {
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    if (now - start > limit) { ctl->error = 1; break; }
    __nanosleep(256);
  }
  __threadfence_system();
}
// Expert reduce (sum kernel: route order, CPU outputs for misses) fused with
// the mHC expand of the FFN branch (hc_expand, one token).
// CPU experts of the previous MoE layer deferred to this one (expert deferral).
struct Deferred { const unsigned char* mailbox; const int* ids; const float* weights; const void* const* images;
                  const float* post; const Plan* plan; };
// Deferred and dropped CPU experts (plan masks) are left out of this layer.
// `prev.mailbox`: the previous layer's deferred outputs, carried through this
// layer's two mHC mixes with coefficients comb_f^T comb_a^T post_prev.
__global__ void __launch_bounds__(256) moe_finish(B* out, B* mlp, const B* expert, const B* residual,
    const float* map, const float* attn_map, const int* ids, const float* weights, const void* const* images,
    const unsigned char* mailbox, const Plan* plan, Deferred prev) {
  const unsigned absent = plan->skip | plan->defer, prev_defer = prev.mailbox ? prev.plan->defer : 0u;
  unsigned c = blockIdx.x * 256 + threadIdx.x;
  int order[8]; route_order(order, ids);
  float value = 0;
  auto host = reinterpret_cast<const B*>(mailbox + MB_OUTPUT);
  for (unsigned g = 0; g < 8; ++g) {
    unsigned a = order[g];
    if (!images[a] && (absent >> a & 1u)) continue;
    B e = images[a] ? expert[a * 4096 + c] : __ldcv(host + a * 4096 + c);
    value = expert_add(value, f(e), weights[a]);
  }
  B branch = b(value + f(expert[8 * 4096 + c]));
  mlp[c] = branch;
  float deferred = 0, coef[4] = {};
  if (prev.mailbox) {
    auto old = reinterpret_cast<const B*>(prev.mailbox + MB_OUTPUT);
    for (unsigned a = 0; a < 8; ++a)
      if (!prev.images[a] && (prev_defer >> a & 1u)) deferred += prev.weights[a] * f(__ldcv(old + a * 4096 + c));
    for (unsigned h = 0; h < 4; ++h) {
      float into = 0;
      for (unsigned g = 0; g < 4; ++g) into += rnd(attn_map[8 + g * 4 + h]) * rnd(prev.post[4 + g]);
      for (unsigned j = 0; j < 4; ++j) coef[j] += rnd(map[8 + h * 4 + j]) * into;
    }
  }
  for (unsigned j = 0; j < 4; ++j) {
    float mix = 0;
    for (unsigned h = 0; h < 4; ++h) mix += rnd(map[8 + h * 4 + j]) * f(residual[h * 4096 + c]);
    float product = rnd(rnd(map[4 + j]) * f(branch));
    float o = product + rnd(mix);
    if (prev.mailbox) o += coef[j] * deferred;
    out[j * 4096 + c] = b(o);
  }
}
// head_mean + final RMSNorm (ops.cu head_mean and norm, one block).
__global__ void __launch_bounds__(256) head_norm(B* mean, B* out, const B* x, const B* w) {
  __shared__ float s[256];
  for (unsigned c = threadIdx.x; c < 4096; c += 256) {
    float sum = 0; for (unsigned j = 0; j < 4; ++j) sum += f(x[j * 4096 + c]);
    mean[c] = b(sum / 4.f);
  }
  __syncthreads();
  norm_row(out, mean, w, nullptr, 4096, 0, s);
}
// Argmax with the lowest index on ties (as greedy), two levels.
__global__ void __launch_bounds__(256) greedy(float* part_v, unsigned* part_i, const B* logits, unsigned n,
    unsigned* counter, Ctl* ctl, volatile unsigned* host) {
  __shared__ float values[256]; __shared__ unsigned ids[256];
  const unsigned chunk = (n + gridDim.x - 1) / gridDim.x, start = blockIdx.x * chunk, end = min(n, start + chunk);
  float best = -INFINITY; unsigned arg = 0;
  for (unsigned i = start + threadIdx.x; i < end; i += 256) if (f(logits[i]) > best) { best = f(logits[i]); arg = i; }
  auto fold = [&]() {
    values[threadIdx.x] = best; ids[threadIdx.x] = arg; __syncthreads();
    for (unsigned s = 128; s; s /= 2) {
      if (threadIdx.x < s && (values[threadIdx.x + s] > values[threadIdx.x] ||
          (values[threadIdx.x + s] == values[threadIdx.x] && ids[threadIdx.x + s] < ids[threadIdx.x]))) {
        values[threadIdx.x] = values[threadIdx.x + s]; ids[threadIdx.x] = ids[threadIdx.x + s];
      }
      __syncthreads();
    }
  };
  fold();
  if (threadIdx.x == 0) { part_v[blockIdx.x] = values[0]; part_i[blockIdx.x] = ids[0]; }
  if (!last_block(counter)) return;
  best = -INFINITY; arg = 0;
  if (threadIdx.x < gridDim.x) { best = __ldcg(part_v + threadIdx.x); arg = __ldcg(part_i + threadIdx.x); }
  __syncthreads();
  fold();
  if (threadIdx.x == 0) {
    ctl->next = ids[0];
    host[0] = ids[0];
    host[1] = ctl->error;
    __threadfence_system();
  }
}
// Loads the host-written control block (mapped memory) into device memory
// and embeds its token into 4 replicated hidden streams.
__global__ void embed(B* out, const B* table, Ctl* ctl, const volatile Ctl* host, unsigned vocab) {
  __shared__ unsigned token;
  unsigned c = blockIdx.x * 256 + threadIdx.x;
  if (threadIdx.x == 0) {
    token = host->token;
    if (c == 0) { ctl->token = token; ctl->position = host->position; ctl->seq = host->seq; ctl->error = 0; ctl->next = 0; }
  }
  __syncthreads();
  unsigned id = token;
  if (c >= 4096) return;
  B value = b(0.f); if (id < vocab) value = table[size_t(id) * 4096 + c];
  for (unsigned h = 0; h < 4; ++h) out[h * 4096 + c] = value;
}
}  // namespace

#define END return cudaGetLastError()
extern "C" {
struct GlmSegment { glm53::DenseWeight w; void* out; unsigned rows; };
int hayate_glm_dev_hc_norm(void* map, void* collapsed, void* out, const void* x, const void* fn, const void* base,
    const void* scale, const void* w, void* logits, void* counter, cudaStream_t st) {
  if (!map || !collapsed || !out || !x || !fn || !base || !scale || !w || !logits || !counter) return cudaErrorInvalidValue;
  hc_norm<<<24, 256, 0, st>>>((float*)map, (B*)collapsed, (B*)out, (const B*)x, (const B*)fn, (const float*)base,
      (const float*)scale, (const B*)w, (float*)logits, (unsigned*)counter);
  END;
}
int hayate_glm_dev_gemv(const GlmSegment* segments, unsigned count, const void* x, unsigned k, cudaStream_t st) {
  if (!segments || !count || count > 6 || !x || !k || k % 16) return cudaErrorInvalidValue;
  Segments m{}; unsigned blocks = 0;
  for (unsigned i = 0; i < count; ++i) {
    auto& s = segments[i];
    if (!s.w.data || !s.out || !s.rows || s.w.format > 3 || (s.w.format == 1 && !s.w.scale)) return cudaErrorInvalidValue;
    m.s[i] = {s.w, (B*)s.out, s.rows, (s.rows + 7) / 8};
    blocks += m.s[i].blocks;
  }
  m.count = count;
  gemv_multi<<<blocks, 256, 0, st>>>(m, (const B*)x, k);
  END;
}
int hayate_glm_dev_kda(void* core, void* gated, void* state, void* hq, void* hk, void* hv, const void* q, const void* k,
    const void* v, const void* wq, const void* wk, const void* wv, const void* fa, glm53::DenseWeight fb, const void* ga,
    glm53::DenseWeight gb, const void* dt_bias, const void* alog, const void* beta_raw, const void* norm, cudaStream_t st) {
  if (!core || !gated || !state || !hq || !hk || !hv || !q || !k || !v || !wq || !wk || !wv || !fa || !fb.data || !ga ||
      !gb.data || !dt_bias || !alog || !beta_raw || !norm || fb.format > 3 || gb.format > 3)
    return cudaErrorInvalidValue;
  kda_step<<<64, 256, 0, st>>>((B*)core, (B*)gated, (float*)state, (B*)hq, (B*)hk, (B*)hv, (const B*)q, (const B*)k,
      (const B*)v, (const B*)wq, (const B*)wk, (const B*)wv, (const B*)fa, fb, (const B*)ga, gb, (const float*)dt_bias,
      (const float*)alog, (const B*)beta_raw, (const B*)norm);
  END;
}
int hayate_glm_dev_sparse_prep(void* qr, const void* qa, const void* qa_norm, void* kvr, const void* kva,
    const void* kv_norm, void* latent, void* ikn, const void* ik, const void* kn_w, const void* kn_b, const void* ig,
    void* tail_keys, void* tail_gates, void* pools, const void* ape, const void* ctl, cudaStream_t st) {
  if (!qr || !qa || !qa_norm || !kvr || !kva || !kv_norm || !latent || !ikn || !ik || !kn_w || !kn_b || !ig ||
      !tail_keys || !tail_gates || !pools || !ape || !ctl)
    return cudaErrorInvalidValue;
  sparse_prep<<<3, 256, 0, st>>>((B*)qr, (const B*)qa, (const B*)qa_norm, (B*)kvr, (const B*)kva, (const B*)kv_norm,
      (B*)latent, (B*)ikn, (const B*)ik, (const B*)kn_w, (const B*)kn_b, (const B*)ig, (B*)tail_keys,
      (B*)tail_gates, (B*)pools, (const B*)ape, (const Ctl*)ctl);
  END;
}
// `plan`: this layer's Plan (4 x u32), see `route`. `format_table`/`formats`: this layer's [288] ExpertFormat
// (8 bytes each) and the [8] chosen ones in route order (both null: no LOWBIT experts). `host_table`: this
// layer's [288] RAM-tier image pointers for the PCIe lane (null: no lane), with `scratch` [4] device slots,
// `src` [4] fetch sources and the nibble table `take` (see `Pcie`).
int hayate_glm_dev_route(void* logits, void* weights, void* ids, void* images, void* plan, const void* x,
    const void* w, const void* bias, const void* table, void* mailbox, const void* ctl, void* counter,
    float skip_below, float defer_below, const void* format_table, void* formats, const void* host_table,
    const void* scratch, void* src, unsigned long long take, const void* pre_src, const void* pre_scratch,
    const void* pre_flags, void* progress, unsigned layer, void* stats, cudaStream_t st) {
  if (!logits || !weights || !ids || !images || !plan || !x || !w || !bias || !table || !ctl || !counter ||
      !(skip_below >= 0.f) || !(defer_below >= 0.f) || (host_table && (!scratch || !src)) ||
      !pre_src != !pre_scratch || !pre_src != !pre_flags || (pre_src && (!host_table || !progress)) || layer > 255)
    return cudaErrorInvalidValue;
  route<<<36, 256, 0, st>>>((float*)logits, (float*)weights, (int*)ids, (const void**)images, (Plan*)plan,
      (const B*)x, (const B*)w, (const float*)bias, (const void* const*)table, (unsigned char*)mailbox,
      (const Ctl*)ctl, (unsigned*)counter, skip_below, defer_below, (const unsigned long long*)format_table,
      (unsigned long long*)formats,
      Pcie{(const void* const*)host_table, (const void* const*)scratch, (const void**)src, take,
           (const void* const*)pre_src, (const void* const*)pre_scratch,
           (const volatile unsigned long long*)pre_flags, (unsigned long long*)progress, layer, (unsigned*)stats});
  END;
}
int hayate_glm_dev_prefetch(const void* dst, const void* src, size_t bytes, unsigned blocks, void* done, void* bad,
    void* flags, const void* progress, const void* ctl, unsigned layer, cudaStream_t st) {
  if (!dst || !src || !bytes || bytes % 16 || !blocks || !done || !bad || !flags || !progress || !ctl || layer > 255)
    return cudaErrorInvalidValue;
  prefetch<<<dim3(blocks, 8), 256, 0, st>>>((void* const*)dst, (const void* const*)src, bytes, (unsigned*)done,
      (unsigned*)bad, (unsigned long long*)flags, (const volatile unsigned long long*)progress, (const Ctl*)ctl,
      layer);
  END;
}
int hayate_glm_dev_predict(void* logits, const void* x, const void* w, const void* bias, const void* table,
    const void* host, void* src, unsigned slots, void* counter, void* stats, cudaStream_t st) {
  if (!logits || !x || !w || !bias || !table || !host || !src || !counter || slots > 8) return cudaErrorInvalidValue;
  predict<<<36, 256, 0, st>>>((float*)logits, (const B*)x, (const B*)w, (const float*)bias,
      (const void* const*)table, (const void* const*)host, (const void**)src, slots, (unsigned*)counter,
      (unsigned*)stats);
  END;
}
// Waits for mailbox flag `flag` (64: DONE, 96: DONE2) when *count != 0.
int hayate_glm_dev_moe_wait(const void* count, const void* mailbox, unsigned flag, void* ctl,
    unsigned long long limit_ns, cudaStream_t st) {
  if (!count || !mailbox || !ctl || (flag != MB_DONE && flag != MB_DONE2)) return cudaErrorInvalidValue;
  moe_wait<<<1, 32, 0, st>>>((const unsigned*)count, (const unsigned char*)mailbox, flag, (Ctl*)ctl, limit_ns);
  END;
}
struct GlmDeferred { const void* mailbox; const void* ids; const void* weights; const void* images; const void* post;
                     const void* plan; };
int hayate_glm_dev_moe_finish(void* out, void* mlp, const void* expert, const void* residual, const void* map,
    const void* attn_map, const void* ids, const void* weights, const void* images, const void* mailbox,
    const void* plan, GlmDeferred prev, cudaStream_t st) {
  if (!out || !mlp || !expert || !residual || !map || !attn_map || !ids || !weights || !images || !mailbox || !plan ||
      (prev.mailbox && (!prev.ids || !prev.weights || !prev.images || !prev.post || !prev.plan)))
    return cudaErrorInvalidValue;
  moe_finish<<<16, 256, 0, st>>>((B*)out, (B*)mlp, (const B*)expert, (const B*)residual, (const float*)map,
      (const float*)attn_map, (const int*)ids, (const float*)weights, (const void* const*)images,
      (const unsigned char*)mailbox, (const Plan*)plan,
      Deferred{(const unsigned char*)prev.mailbox, (const int*)prev.ids, (const float*)prev.weights,
               (const void* const*)prev.images, (const float*)prev.post, (const Plan*)prev.plan});
  END;
}
int hayate_glm_dev_head_norm(void* mean, void* out, const void* x, const void* w, cudaStream_t st) {
  if (!mean || !out || !x || !w) return cudaErrorInvalidValue;
  head_norm<<<1, 256, 0, st>>>((B*)mean, (B*)out, (const B*)x, (const B*)w);
  END;
}
int hayate_glm_dev_greedy(void* part_v, void* part_i, const void* logits, unsigned n, void* counter, void* ctl,
    void* host, cudaStream_t st) {
  if (!part_v || !part_i || !logits || !n || !counter || !ctl || !host) return cudaErrorInvalidValue;
  greedy<<<128, 256, 0, st>>>((float*)part_v, (unsigned*)part_i, (const B*)logits, n, (unsigned*)counter, (Ctl*)ctl,
      (volatile unsigned*)host);
  END;
}
int hayate_glm_dev_embed(void* out, const void* table, void* ctl, const void* host, unsigned vocab, cudaStream_t st) {
  if (!out || !table || !ctl || !host || !vocab) return cudaErrorInvalidValue;
  embed<<<16, 256, 0, st>>>((B*)out, (const B*)table, (Ctl*)ctl, (const volatile Ctl*)host, vocab);
  END;
}
}
namespace {
// Bandwidth probe: XOR of every 16-byte word (device or mapped host memory).
__global__ void read_probe(const uint4* src, size_t words, unsigned* sink) {
  unsigned acc = 0;
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < words; i += size_t(gridDim.x) * blockDim.x) {
    uint4 v = src[i]; acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x9e3779b9u) *sink = acc;
}
}  // namespace
extern "C" int hayate_glm_dev_read_probe(const void* src, size_t bytes, void* sink, unsigned blocks, cudaStream_t st) {
  if (!src || !sink || bytes % 16 || !blocks) return cudaErrorInvalidValue;
  read_probe<<<blocks, 512, 0, st>>>((const uint4*)src, bytes / 16, (unsigned*)sink);
  END;
}
