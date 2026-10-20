// M1 decode GEMVs: one warp per output row, x staged in shared memory, fp32 accumulation.
#include "decode.cuh"
#include "rank1.cuh"

// y[o] = sum_k W[o][k] x[k]. W bf16 [O][K] row-major, K % 8 == 0. Dynamic smem: K floats.
extern "C" __global__ void __launch_bounds__(256) gemv_bf16(const __nv_bfloat16* __restrict__ W,
                                                           const float* __restrict__ x,
                                                           float* __restrict__ y, int O, int K) {
  TSJ_PDL_ENTRY;
  extern __shared__ float xsm[];
  for (int i = threadIdx.x; i < K; i += blockDim.x) xsm[i] = x[i];
  __syncthreads();
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (row >= O) return;
  const uint4* w = reinterpret_cast<const uint4*>(W + size_t(row) * K);
  float acc = 0.f;
  for (int c = lane; c < K / 8; c += 32) {
    float f[8];
    bf16x8(__ldcs(w + c), f);
#pragma unroll
    for (int j = 0; j < 8; ++j) acc = fmaf(f[j], xsm[c * 8 + j], acc);
  }
  acc = warp_sum(acc);
  if (lane == 0) y[row] = acc;
}

// MoE experts: the topk routed experts (NVFP4) and the shared expert (BF16, slot topk) in one
// launch per projection. Routed expert block layout (bytes, F x H matrices, see
// crates/tsumuji-engine/src/weights.rs):
//   [gate codes | up codes | down codes | gate sf | up sf | down sf], stride 27/16 F H.
// G lanes share a row (each lane takes every G-th 16-byte vector), so short rows still keep
// every lane loading: gate/up (K = H) use G = 16, down (K = F) uses G = 4. 256 threads = 8 warps
// of 32 / G rows each. The shared expert does not depend on the router, so its rows are
// prefetched into L2 before `griddepcontrol.wait`.

TSJ_DEVICE void prefetch_l2(const void* p) { asm volatile("prefetch.global.L2 [%0];" ::"l"(p)); }

// This lane's part of a dense row . x (vectors start, start + step, ... of K / VW; bf16 or MXFP8 with
// the row's scales `sc`), x in the padded layout of `x32`.
template <int FMT>
TSJ_DEVICE float dense_dot_part(const uint4* __restrict__ row, const uint8_t* __restrict__ sc, const float* __restrict__ x,
                                int K, int start, int step) {
  using F = Fmt<FMT>;
  constexpr int VW = F::VW;
  float acc = 0.f;
#pragma unroll 4
  for (int v = start; v < K / VW; v += step) {
    float f[VW];
    F::dq(__ldcs(row + v), FMT != 0 ? ue8m0(__ldcs(sc + v * VW / 32)) : 1.f, f);
#pragma unroll
    for (int j = 0; j < VW; ++j) acc = fmaf(f[j], x[x32((v * VW + j) / 32, (v * VW + j) % 32)], acc);
  }
  return acc;
}

TSJ_DEVICE void prefetch_row(const uint4* row, int vectors, int start, int step) {
  for (int v = start; v < vectors; v += step)
    if ((v & 7) == 0) prefetch_l2(row + v);  // one per 128-byte line
}

// h[n][s][r] = silu(gate_s[r] . x[n]) * (up_s[r] . x[n]) for token n, slot s <= topk (topk: shared
// expert), row r < F. grid = (ceil(F / (8 * 32 / G)), topk + 1, N), smem H floats.
template <int G, int FMT>
__device__ __forceinline__ void moe_gate_up_impl(const uint8_t* __restrict__ experts, const float* __restrict__ gs,
                                                 const int* __restrict__ sel, const void* __restrict__ shared_gu,
                                                 const uint8_t* __restrict__ shared_sc,
                                                 const float* __restrict__ x, float* __restrict__ h, int H, int F,
                                                 int topk) {
  extern __shared__ float xs[];
  const int lane = threadIdx.x & 31, s = blockIdx.y, j = lane % G, n = blockIdx.z;
  x += size_t(n) * H;
  h += size_t(n) * (topk + 1) * F;
  sel += n * topk;
  // Rows past F compute row 0 and skip the store: every lane must reach the group shuffles.
  const int rq = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * (32 / G) + lane / G;
  const int r = rq < F ? rq : 0;
  const bool shared = s == topk;
  constexpr int VW = Fmt<FMT>::VW;
  const uint4* sg = reinterpret_cast<const uint4*>(shared_gu) + size_t(r) * (H / VW);
  const uint4* su = reinterpret_cast<const uint4*>(shared_gu) + size_t(F + r) * (H / VW);
  const uint8_t* sgs = shared_sc + size_t(r) * (H / 32);
  const uint8_t* sus = shared_sc + size_t(F + r) * (H / 32);
  if (shared) {
    prefetch_row(sg, H / VW, j, G);
    prefetch_row(su, H / VW, j, G);
  }
  griddep_wait();
  griddep_launch_dependents();
  for (int i = threadIdx.x; i < H; i += blockDim.x) xs[x32(i / 32, i % 32)] = x[i];
  __syncthreads();
  float g, u;
  if (shared) {
    g = group_sum<G>(dense_dot_part<FMT>(sg, sgs, xs, H, j, G));
    u = group_sum<G>(dense_dot_part<FMT>(su, sus, xs, H, j, G));
  } else {
    const int e = sel[s];
    const size_t fh = size_t(F) * H;
    const uint8_t* base = experts + size_t(e) * (fh * 27 / 16);
    const uint4* gate = reinterpret_cast<const uint4*>(base + size_t(r) * (H / 2));
    const uint4* up = reinterpret_cast<const uint4*>(base + fh / 2 + size_t(r) * (H / 2));
    const uint16_t* gsf = reinterpret_cast<const uint16_t*>(base + fh * 3 / 2 + size_t(r) * (H / 16));
    const uint16_t* usf = reinterpret_cast<const uint16_t*>(base + fh * 3 / 2 + fh / 16 + size_t(r) * (H / 16));
    g = group_sum<G>(nvfp4_dot_regs<5>(gate, gsf, xs, H, j, G)) * gs[e * 3 + 0];
    u = group_sum<G>(nvfp4_dot_regs<5>(up, usf, xs, H, j, G)) * gs[e * 3 + 1];
  }
  if (j == 0 && rq < F) h[s * F + r] = siluf(g) * u;
}

// y[n][s][o] = down_s[o] . h[n][s] for token n, slot s <= topk (down is H x F), plus the rank-1
// adapters of the routed experts (`ex`: a [E][F], b [E][H]) and of the shared expert (`sh`).
// grid = (ceil(H / (8 * 32 / G)), topk + 1, N), smem F floats.
template <int G, int FMT>
__device__ __forceinline__ void moe_down_impl(const uint8_t* __restrict__ experts, const float* __restrict__ gs,
                                              const int* __restrict__ sel, const void* __restrict__ shared_down,
                                              const uint8_t* __restrict__ shared_sc,
                                              const float* __restrict__ h, float* __restrict__ y, int H, int F,
                                              int topk, const Rank1 ex, const Rank1 sh) {
  extern __shared__ float hs[];
  __shared__ float red[32];
  const int lane = threadIdx.x & 31, s = blockIdx.y, j = lane % G, n = blockIdx.z;
  h += size_t(n) * (topk + 1) * F;
  y += size_t(n) * (topk + 1) * H;
  sel += n * topk;
  const int oq = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * (32 / G) + lane / G;
  const int o = oq < H ? oq : 0;
  const bool shared = s == topk;
  constexpr int VW = Fmt<FMT>::VW;
  const uint4* sd = reinterpret_cast<const uint4*>(shared_down) + size_t(o) * (F / VW);
  const uint8_t* sds = shared_sc + size_t(o) * (F / 32);
  if (shared) prefetch_row(sd, F / VW, j, G);
  griddep_wait();
  griddep_launch_dependents();
  for (int i = threadIdx.x; i < F; i += blockDim.x) hs[x32(i / 32, i % 32)] = h[s * F + i];
  __syncthreads();
  // Adapter: s (a_e . h) for this slot (uniform per block), times b_e[o] below.
  const Rank1& ad = shared ? sh : ex;
  const size_t ae = shared ? 0 : size_t(sel[s]);
  float lora = 0.f;
  if (ad.a != nullptr && *ad.scale != 0.f) {
    float p = 0.f;
    for (int i = threadIdx.x; i < F; i += blockDim.x) p = fmaf(ad.a[ae * F + i], hs[x32(i / 32, i % 32)], p);
    lora = *ad.scale * block_sum(p, red);
  }
  float v;
  if (shared) {
    v = group_sum<G>(dense_dot_part<FMT>(sd, sds, hs, F, j, G));
  } else {
    const int e = sel[s];
    const size_t fh = size_t(F) * H;
    const uint8_t* base = experts + size_t(e) * (fh * 27 / 16);
    const uint4* down = reinterpret_cast<const uint4*>(base + fh + size_t(o) * (F / 2));
    const uint16_t* dsf = reinterpret_cast<const uint16_t*>(base + fh * 3 / 2 + fh / 8 + size_t(o) * (F / 16));
    v = group_sum<G>(nvfp4_dot_regs<5>(down, dsf, hs, F, j, G)) * gs[e * 3 + 2];
  }
  if (lora != 0.f) v = fmaf(lora, ad.b[ae * H + o], v);
  if (j == 0 && oq < H) y[s * H + o] = v;
}

#define MOE(SUFFIX, FMT)                                                                                     \
  extern "C" __global__ void __launch_bounds__(256) moe_gate_up##SUFFIX(                                     \
      const uint8_t* __restrict__ experts, const float* __restrict__ gs, const int* __restrict__ sel,         \
      const void* __restrict__ shared_gu, const uint8_t* __restrict__ shared_sc, const float* __restrict__ x,  \
      float* __restrict__ h, int H, int F, int topk) {                                                       \
    moe_gate_up_impl<16, FMT>(experts, gs, sel, shared_gu, shared_sc, x, h, H, F, topk);                      \
  }                                                                                                           \
  extern "C" __global__ void __launch_bounds__(256) moe_down##SUFFIX(                                        \
      const uint8_t* __restrict__ experts, const float* __restrict__ gs, const int* __restrict__ sel,         \
      const void* __restrict__ shared_down, const uint8_t* __restrict__ shared_sc, const float* __restrict__ h, \
      float* __restrict__ y, int H, int F, int topk, const Rank1 ex, const Rank1 sh) {                       \
    moe_down_impl<4, FMT>(experts, gs, sel, shared_down, shared_sc, h, y, H, F, topk, ex, sh);                \
  }
MOE(, 0)
MOE(_fp8, 1)
MOE(_i8, 2)

// Each (token, router slot) is a candidate union entry. Only the earliest token selecting the
// expert runs; it gathers all of that expert's tokens and scatters to their original slots.
// The shared expert always belongs to token 0. No router changes, extra launch or atomics.
// N=1 keeps the kernels above, including their resource usage. Union smem: NT * (K/32*33) floats.
template <int NT>
TSJ_DEVICE bool expert_tokens(const int* sel, int topk, int* slots) {
  const int s = blockIdx.y, n = blockIdx.z, t = threadIdx.x;
  const bool shared = s == topk;
  if (shared && n != 0) return false;
  const int e = shared ? -1 : sel[n * topk + s];
  if (!shared && n != 0) {
    // Every warp checks the same immutable prefix. Duplicate candidates leave before the
    // block barrier and before gathering the later tokens; all warps get the same answer.
    bool seen = false;
    for (int i = (t & 31); i < n * topk; i += 32) seen |= sel[i] == e;
    if (__any_sync(0xffffffffu, seen)) return false;
  }
  if (t < NT) {
    int slot = -1;
    if (t < int(gridDim.z)) {
      if (shared) slot = topk;
      else
        for (int i = 0; i < topk; ++i)
          if (sel[t * topk + i] == e) slot = i;
    }
    slots[t] = slot;
  }
  __syncthreads();
  return true;
}

template <int FMT, int NT>
TSJ_DEVICE void dense_dot_tokens(const uint4* row, const uint8_t* sc, const float* x, int stride,
                                 const int* slots, int K, int start, int step, float (&acc)[NT]) {
  using F = Fmt<FMT>;
  constexpr int VW = F::VW;
#pragma unroll 1
  for (int v = start; v < K / VW; v += step) {
    float f[VW];
    F::dq(__ldcs(row + v), FMT != 0 ? ue8m0(__ldcs(sc + v * VW / 32)) : 1.f, f);
#pragma unroll
    for (int n = 0; n < NT; ++n) {
      if (slots[n] < 0) continue;
#pragma unroll
      for (int j = 0; j < VW; ++j)
        acc[n] = fmaf(f[j], x[n * stride + x32((v * VW + j) / 32, (v * VW + j) % 32)], acc[n]);
    }
  }
}

// Exactly nvfp4_dot_regs<5>'s arithmetic: independent 16-element sums, sf after each sum,
// the first five vectors, then a separately accumulated tail, and finally acc + tail. At large
// NT the token FMAs cover the load latency; fewer prefetched vectors keep the live set small.
template <int NT, int NP>
TSJ_DEVICE void nvfp4_dot_tokens(const uint4* codes, const uint16_t* sf, const float* x, int stride,
                                 const int* slots, int K, int start, int step, float (&acc)[NT]) {
  uint4 w[NP];
  uint16_t sc[NP];
#pragma unroll
  for (int i = 0; i < NP; ++i) {
    const int v = start + i * step;
    if (v < K / 32) {
      w[i] = __ldcs(codes + v);
      sc[i] = __ldcs(sf + v);
    }
  }
  auto apply = [&](uint4 wt, uint16_t s, int v, float (&a)[NT]) {
    const uint32_t words[4] = {wt.x, wt.y, wt.z, wt.w};
#pragma unroll
    for (int g = 0; g < 2; ++g) {
      float part[NT] = {};
#pragma unroll
      for (int q = 0; q < 2; ++q) {
#pragma unroll
        for (int b = 0; b < 4; ++b) {
          const float2 f = __half22float2(e2m1x2_to_half2(uint8_t(words[g * 2 + q] >> (8 * b))));
          const int k = x32(v, g * 16 + q * 8 + b * 2);
#pragma unroll
          for (int n = 0; n < NT; ++n)
            if (slots[n] >= 0) {
              part[n] = fmaf(f.x, x[n * stride + k], part[n]);
              part[n] = fmaf(f.y, x[n * stride + k + 1], part[n]);
            }
        }
      }
      const float scale = e4m3_to_float(uint8_t(s >> (8 * g)));
#pragma unroll
      for (int n = 0; n < NT; ++n)
        if (slots[n] >= 0) a[n] = fmaf(part[n], scale, a[n]);
    }
  };
#pragma unroll
  for (int i = 0; i < NP; ++i) {
    const int v = start + i * step;
    if (v < K / 32) apply(w[i], sc[i], v, acc);
  }
  if constexpr (NP < 5)
    for (int v = start + NP * step; v < min(K / 32, start + 5 * step); v += step)
      apply(__ldcs(codes + v), __ldcs(sf + v), v, acc);
  float tail[NT] = {};
  for (int v = start + 5 * step; v < K / 32; v += step) apply(codes[v], sf[v], v, tail);
#pragma unroll
  for (int n = 0; n < NT; ++n) acc[n] += tail[n];
}

template <int G, int FMT, int NT, bool DOWN>
__device__ __forceinline__ void moe_union_impl(const uint8_t* __restrict__ experts, const float* __restrict__ gs,
                                               const int* __restrict__ sel, const void* __restrict__ sw,
                                               const uint8_t* __restrict__ sc, const float* __restrict__ x,
                                               float* __restrict__ y, int H, int F, int topk) {
  extern __shared__ float xs[];
  __shared__ int slots[NT];
  const int lane = threadIdx.x & 31, j = lane % G;
  const int K = DOWN ? F : H, O = DOWN ? H : F, stride = K / 32 * 33;
  const int rq = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * (32 / G) + lane / G;
  const int r = rq < O ? rq : 0;
  const bool shared = blockIdx.y == topk;
  constexpr int VW = Fmt<FMT>::VW;
  // ptxas: these prefetch counts avoid spills at NT=4 and NT=8 without limiting registers.
  constexpr int NP = NT == 4 || (DOWN && NT == 8) ? 1 : 5;
  const uint4* sd = reinterpret_cast<const uint4*>(sw) + size_t(r) * (K / VW);
  const uint8_t* sds = sc + size_t(r) * (K / 32);
  if (shared && blockIdx.z == 0) {
    prefetch_row(sd, K / VW, j, G);
    if constexpr (!DOWN) prefetch_row(sd + size_t(F) * (K / VW), K / VW, j, G);
  }
  griddep_wait();
  griddep_launch_dependents();
  if (!expert_tokens<NT>(sel, topk, slots)) return;
#pragma unroll
  for (int n = 0; n < NT; ++n)
    if (slots[n] >= 0)
      for (int k = threadIdx.x; k < K; k += blockDim.x) {
        const size_t at = DOWN ? (size_t(n) * (topk + 1) + slots[n]) * F : size_t(n) * H;
        xs[n * stride + x32(k / 32, k % 32)] = x[at + k];
      }
  __syncthreads();
  float a[NT] = {}, b[NT] = {};
  int e = 0;
  if (shared) {
    dense_dot_tokens<FMT, NT>(sd, sds, xs, stride, slots, K, j, G, a);
    if constexpr (!DOWN)
      dense_dot_tokens<FMT, NT>(sd + size_t(F) * (K / VW), sds + size_t(F) * (K / 32), xs, stride, slots, K, j, G, b);
  } else {
    e = sel[blockIdx.z * topk + blockIdx.y];
    const size_t fh = size_t(F) * H;
    const uint8_t* base = experts + size_t(e) * (fh * 27 / 16);
    const size_t co = DOWN ? fh : 0, so = fh * 3 / 2 + (DOWN ? fh / 8 : 0);
    const uint4* code = reinterpret_cast<const uint4*>(base + co + size_t(r) * (K / 2));
    const uint16_t* sf = reinterpret_cast<const uint16_t*>(base + so + size_t(r) * (K / 16));
    nvfp4_dot_tokens<NT, NP>(code, sf, xs, stride, slots, K, j, G, a);
    if constexpr (!DOWN)
      nvfp4_dot_tokens<NT, NP>(code + fh / 32, sf + fh / 32, xs, stride, slots, K, j, G, b);
  }
#pragma unroll
  for (int n = 0; n < NT; ++n) {
    if (slots[n] < 0) continue;
    float v = group_sum<G>(a[n]);
    if constexpr (DOWN) {
      if (!shared) v *= gs[e * 3 + 2];
    } else {
      float u = group_sum<G>(b[n]);
      if (!shared) v *= gs[e * 3], u *= gs[e * 3 + 1];
      v = siluf(v) * u;
    }
    if (j == 0 && rq < O) y[(size_t(n) * (topk + 1) + slots[n]) * O + r] = v;
  }
}

#define MOE_UNION(SUFFIX, FMT, NT)                                                                          \
  extern "C" __global__ void __launch_bounds__(256) moe_gate_up##SUFFIX##_union_n##NT(                        \
      const uint8_t* __restrict__ experts, const float* __restrict__ gs, const int* __restrict__ sel,          \
      const void* __restrict__ sw, const uint8_t* __restrict__ sc, const float* __restrict__ x,                 \
      float* __restrict__ y, int H, int F, int topk) {                                                       \
    moe_union_impl<16, FMT, NT, false>(experts, gs, sel, sw, sc, x, y, H, F, topk);                            \
  }                                                                                                       \
  extern "C" __global__ void __launch_bounds__(256) moe_down##SUFFIX##_union_n##NT(                           \
      const uint8_t* __restrict__ experts, const float* __restrict__ gs, const int* __restrict__ sel,          \
      const void* __restrict__ sw, const uint8_t* __restrict__ sc, const float* __restrict__ x,                 \
      float* __restrict__ y, int H, int F, int topk) {                                                       \
    moe_union_impl<4, FMT, NT, true>(experts, gs, sel, sw, sc, x, y, H, F, topk);                              \
  }
#define MOE_UNION_ALL(SUFFIX, FMT) MOE_UNION(SUFFIX, FMT, 2) MOE_UNION(SUFFIX, FMT, 4) MOE_UNION(SUFFIX, FMT, 8)
MOE_UNION_ALL(, 0)
MOE_UNION_ALL(_fp8, 1)
MOE_UNION_ALL(_i8, 2)
