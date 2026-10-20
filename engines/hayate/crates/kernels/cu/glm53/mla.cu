// GLM-5.3 sparse attention in the absorbed (MLA) form. The sparse layers have
// no RoPE: K_h = W_K[h] latent and V_h = W_V[h] latent (kv_b_proj rows
// h*512 + 0..256 and h*512 + 256..512). Scores use q_abs[h] = W_K[h]^T q[h]
// against the 512-wide latent, and the output is W_V[h] (sum_j p_j latent_j).
// The token selection (top-512 index pools + tail) is shared by all 64 heads,
// so one query row is a [64 x 512] x [512 x tokens] tensor-core GEMM.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
namespace {
using B = __nv_bfloat16;
__device__ __forceinline__ float f(B x) { return __bfloat162float(x); }
__device__ __forceinline__ B b(float x) { return __float2bfloat16_rn(x); }
__device__ __forceinline__ float rnd(float x) { return f(b(x)); }

constexpr unsigned WIDTH = 2051, LAT = 512, HEADS = 64, HG = 16, TILE = 64;
constexpr unsigned QS = LAT + 8, LS = LAT + 8, PS = TILE + 8;
constexpr size_t SMEM = (HG * QS + TILE * LS + HG * PS) * sizeof(B) + (8 * 16 * 2 + 16 * 3) * sizeof(float) + TILE * sizeof(int);

__device__ __forceinline__ void mma(float* c, const uint32_t* a, const uint32_t* v) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(v[0]), "r"(v[1]));
}
__device__ __forceinline__ uint32_t ld32(const B* p) { return *reinterpret_cast<const uint32_t*>(p); }
__device__ __forceinline__ uint32_t pack(B lo, B hi) {
  return uint32_t(__bfloat16_as_ushort(lo)) | (uint32_t(__bfloat16_as_ushort(hi)) << 16);
}

// grid (rows, HEADS / HG, splits), 256 threads. ids: [rows][WIDTH], valid
// entries first, -1 after. splits == 1 writes normalized BF16 o_lat
// [HEADS][rows][LAT]; otherwise unnormalized F32 partials and (max, sum).
__global__ void __launch_bounds__(256) attention(B* out, float* part, float* stats, const B* qabs,
                                                 const B* latent, const int* ids, unsigned rows) {
  extern __shared__ __align__(16) unsigned char smem[];
  B* Qs = reinterpret_cast<B*>(smem);
  B* Ls = Qs + HG * QS;
  B* Ps = Ls + TILE * LS;
  float* redm = reinterpret_cast<float*>(Ps + HG * PS);
  float* redl = redm + 8 * 16;
  float* rowm = redl + 8 * 16;
  float* rowl = rowm + 16;
  float* rowc = rowl + 16;
  int* tok = reinterpret_cast<int*>(rowc + 16);
  __shared__ unsigned count_s;
  const unsigned row = blockIdx.x, hg = blockIdx.y, split = blockIdx.z, splits = gridDim.z;
  const unsigned tid = threadIdx.x, warp = tid / 32, lane = tid % 32, g = lane / 4, t = lane % 4;
  const int* sel = ids + size_t(row) * WIDTH;
  if (tid == 0) count_s = WIDTH;
  __syncthreads();
  for (unsigned j = tid; j < WIDTH; j += 256)
    if (sel[j] < 0) atomicMin(&count_s, j);
  for (unsigned i = tid; i < HG * LAT / 8; i += 256) {
    unsigned h = i / (LAT / 8), c = (i % (LAT / 8)) * 8;
    *reinterpret_cast<uint4*>(&Qs[h * QS + c]) =
        *reinterpret_cast<const uint4*>(&qabs[(size_t(hg * HG + h) * rows + row) * LAT + c]);
  }
  if (tid < 16) { rowm[tid] = -INFINITY; rowl[tid] = 0.f; }
  float o[8][4];
#pragma unroll
  for (int n = 0; n < 8; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
  __syncthreads();
  const unsigned count = count_s, tiles = (count + TILE - 1) / TILE;
  const unsigned per = (tiles + splits - 1) / splits, t0 = split * per, t1 = min(tiles, t0 + per);
  for (unsigned tile = t0; tile < t1; ++tile) {
    const unsigned base = tile * TILE;
    if (tid < TILE) tok[tid] = base + tid < count ? sel[base + tid] : -1;
    __syncthreads();
    for (unsigned i = tid; i < TILE * LAT / 8; i += 256) {
      unsigned r = i / (LAT / 8), c = (i % (LAT / 8)) * 8;
      int id = tok[r];
      uint4 v = make_uint4(0, 0, 0, 0);
      if (id >= 0) v = *reinterpret_cast<const uint4*>(&latent[size_t(id) * LAT + c]);
      *reinterpret_cast<uint4*>(&Ls[r * LS + c]) = v;
    }
    __syncthreads();
    // S [16 heads x 8 tokens] per warp: tokens warp*8 .. warp*8+7.
    float s[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 8
    for (unsigned k0 = 0; k0 < LAT; k0 += 16) {
      uint32_t a[4] = {ld32(&Qs[g * QS + k0 + 2 * t]), ld32(&Qs[(g + 8) * QS + k0 + 2 * t]),
                       ld32(&Qs[g * QS + k0 + 2 * t + 8]), ld32(&Qs[(g + 8) * QS + k0 + 2 * t + 8])};
      uint32_t v[2] = {ld32(&Ls[(warp * 8 + g) * LS + k0 + 2 * t]),
                       ld32(&Ls[(warp * 8 + g) * LS + k0 + 2 * t + 8])};
      mma(s, a, v);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      unsigned col = warp * 8 + 2 * t + (i & 1);
      s[i] = base + col < count ? rnd(rnd(s[i]) / 16.f) : -INFINITY;
    }
    float m0 = fmaxf(s[0], s[1]), m1 = fmaxf(s[2], s[3]);
    m0 = fmaxf(m0, __shfl_xor_sync(~0u, m0, 1)); m0 = fmaxf(m0, __shfl_xor_sync(~0u, m0, 2));
    m1 = fmaxf(m1, __shfl_xor_sync(~0u, m1, 1)); m1 = fmaxf(m1, __shfl_xor_sync(~0u, m1, 2));
    if (t == 0) { redm[warp * 16 + g] = m0; redm[warp * 16 + g + 8] = m1; }
    __syncthreads();
    if (tid < 16) {
      float m = rowm[tid];
      for (int w = 0; w < 8; ++w) m = fmaxf(m, redm[w * 16 + tid]);
      rowc[tid] = rowm[tid] == -INFINITY ? 0.f : expf(rowm[tid] - m);
      rowm[tid] = m;
    }
    __syncthreads();
    const float mg = rowm[g], mh = rowm[g + 8];
    float p[4] = {s[0] == -INFINITY ? 0.f : expf(s[0] - mg), s[1] == -INFINITY ? 0.f : expf(s[1] - mg),
                  s[2] == -INFINITY ? 0.f : expf(s[2] - mh), s[3] == -INFINITY ? 0.f : expf(s[3] - mh)};
    Ps[g * PS + warp * 8 + 2 * t] = b(p[0]);
    Ps[g * PS + warp * 8 + 2 * t + 1] = b(p[1]);
    Ps[(g + 8) * PS + warp * 8 + 2 * t] = b(p[2]);
    Ps[(g + 8) * PS + warp * 8 + 2 * t + 1] = b(p[3]);
    float l0 = p[0] + p[1], l1 = p[2] + p[3];
    l0 += __shfl_xor_sync(~0u, l0, 1); l0 += __shfl_xor_sync(~0u, l0, 2);
    l1 += __shfl_xor_sync(~0u, l1, 1); l1 += __shfl_xor_sync(~0u, l1, 2);
    if (t == 0) { redl[warp * 16 + g] = l0; redl[warp * 16 + g + 8] = l1; }
    const float c0 = rowc[g], c1 = rowc[g + 8];
#pragma unroll
    for (int n = 0; n < 8; ++n) { o[n][0] *= c0; o[n][1] *= c0; o[n][2] *= c1; o[n][3] *= c1; }
    __syncthreads();
    if (tid < 16) {
      float l = rowl[tid] * rowc[tid];
      for (int w = 0; w < 8; ++w) l += redl[w * 16 + tid];
      rowl[tid] = l;
    }
    // O [16 x 512] += P [16 x 64] . L [64 x 512]; warp owns columns warp*64 .. +64.
#pragma unroll
    for (unsigned k0 = 0; k0 < TILE; k0 += 16) {
      uint32_t a[4] = {ld32(&Ps[g * PS + k0 + 2 * t]), ld32(&Ps[(g + 8) * PS + k0 + 2 * t]),
                       ld32(&Ps[g * PS + k0 + 2 * t + 8]), ld32(&Ps[(g + 8) * PS + k0 + 2 * t + 8])};
#pragma unroll
      for (int n = 0; n < 8; ++n) {
        unsigned col = warp * 64 + n * 8 + g;
        uint32_t v[2] = {pack(Ls[(k0 + 2 * t) * LS + col], Ls[(k0 + 2 * t + 1) * LS + col]),
                         pack(Ls[(k0 + 2 * t + 8) * LS + col], Ls[(k0 + 2 * t + 9) * LS + col])};
        mma(o[n], a, v);
      }
    }
    __syncthreads();
  }
  const unsigned h0 = hg * HG + g, h1 = h0 + 8;
  if (splits == 1) {
    const float i0 = rowl[g] > 0.f ? 1.f / rowl[g] : 0.f, i1 = rowl[g + 8] > 0.f ? 1.f / rowl[g + 8] : 0.f;
#pragma unroll
    for (int n = 0; n < 8; ++n) {
      unsigned col = warp * 64 + n * 8 + 2 * t;
      B* r0 = out + (size_t(h0) * rows + row) * LAT + col;
      B* r1 = out + (size_t(h1) * rows + row) * LAT + col;
      r0[0] = b(o[n][0] * i0); r0[1] = b(o[n][1] * i0);
      r1[0] = b(o[n][2] * i1); r1[1] = b(o[n][3] * i1);
    }
  } else {
#pragma unroll
    for (int n = 0; n < 8; ++n) {
      unsigned col = warp * 64 + n * 8 + 2 * t;
      float* r0 = part + ((size_t(split) * HEADS + h0) * rows + row) * LAT + col;
      float* r1 = part + ((size_t(split) * HEADS + h1) * rows + row) * LAT + col;
      r0[0] = o[n][0]; r0[1] = o[n][1]; r1[0] = o[n][2]; r1[1] = o[n][3];
    }
    if (tid < 16) {
      float* st = stats + ((size_t(split) * HEADS + hg * HG + tid) * rows + row) * 2;
      st[0] = rowm[tid]; st[1] = rowl[tid];
    }
  }
}
// grid (HEADS * rows), 512 threads (one per latent column).
__global__ void combine(B* out, const float* part, const float* stats, unsigned rows, unsigned splits) {
  const size_t hr = blockIdx.x;  // h * rows + row
  const unsigned c = threadIdx.x;
  float m = -INFINITY;
  for (unsigned s = 0; s < splits; ++s) m = fmaxf(m, stats[(s * HEADS * rows + hr) * 2]);
  float l = 0.f, o = 0.f;
  for (unsigned s = 0; s < splits; ++s) {
    float ms = stats[(s * HEADS * rows + hr) * 2];
    if (ms == -INFINITY) continue;
    float w = expf(ms - m);
    l += w * stats[(s * HEADS * rows + hr) * 2 + 1];
    o += w * part[(s * HEADS * rows + hr) * LAT + c];
  }
  out[hr * LAT + c] = b(l > 0.f ? o / l : 0.f);
}
// q [rows][64][256] x wkt [64][512][256] -> qabs [64][rows][512]; warp per output.
template <int M>
__global__ void absorb_q(B* qabs, const B* q, const B* wkt) {
  unsigned w = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x % 32;
  unsigned h = w / LAT, c = w % LAT;
  const B* wr = wkt + (size_t(h) * LAT + c) * 256 + lane * 8;
  uint4 raw = *reinterpret_cast<const uint4*>(wr);
  const B* wv = reinterpret_cast<const B*>(&raw);
  float acc[M];
#pragma unroll
  for (int m = 0; m < M; ++m) {
    uint4 xr = *reinterpret_cast<const uint4*>(q + (size_t(m) * HEADS + h) * 256 + lane * 8);
    const B* xv = reinterpret_cast<const B*>(&xr);
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) s += f(wv[i]) * f(xv[i]);
    for (int o = 16; o; o /= 2) s += __shfl_xor_sync(~0u, s, o);
    acc[m] = s;
  }
  if (lane == 0)
    for (int m = 0; m < M; ++m) qabs[(size_t(h) * M + m) * LAT + c] = b(acc[m]);
}
// olat [64][rows][512] x wv [64][256][512] -> out [rows][64][256]; warp per output.
template <int M>
__global__ void absorb_out(B* out, const B* olat, const B* wv) {
  unsigned w = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x % 32;
  unsigned h = w / 256, e = w % 256;
  const B* wr = wv + (size_t(h) * 256 + e) * LAT + lane * 16;
  uint4 raw[2] = {reinterpret_cast<const uint4*>(wr)[0], reinterpret_cast<const uint4*>(wr)[1]};
  const B* wvv = reinterpret_cast<const B*>(raw);
#pragma unroll
  for (int m = 0; m < M; ++m) {
    const uint4* xp = reinterpret_cast<const uint4*>(olat + (size_t(h) * M + m) * LAT + lane * 16);
    uint4 xr[2] = {xp[0], xp[1]};
    const B* xv = reinterpret_cast<const B*>(xr);
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) s += f(wvv[i]) * f(xv[i]);
    for (int o = 16; o; o /= 2) s += __shfl_xor_sync(~0u, s, o);
    if (lane == 0) out[(size_t(m) * HEADS + h) * 256 + e] = b(s);
  }
}
// kvb [32768][512] -> wkt [64][512][256] (K transposed), wv [64][256][512].
__global__ void prepare(B* wkt, B* wv, const B* kvb) {
  size_t i = size_t(blockIdx.x) * 256 + threadIdx.x;
  if (i >= size_t(HEADS) * 512 * 512) return;
  unsigned h = i / (512 * 512), r = (i / 512) % 512, c = i % 512;
  B v = kvb[i];
  if (r < 256) wkt[(size_t(h) * LAT + c) * 256 + r] = v;
  else wv[(size_t(h) * 256 + (r - 256)) * LAT + c] = v;
}
// [rows][64][256] <-> [64][rows][256]
__global__ void heads_split(B* out, const B* x, unsigned rows) {
  size_t i = size_t(blockIdx.x) * 256 + threadIdx.x;
  if (i >= size_t(rows) * HEADS * 256) return;
  size_t row = i / (HEADS * 256), h = (i / 256) % HEADS, d = i % 256;
  out[(h * rows + row) * 256 + d] = x[i];
}
__global__ void heads_merge(B* out, const B* x, unsigned rows) {
  size_t i = size_t(blockIdx.x) * 256 + threadIdx.x;
  if (i >= size_t(rows) * HEADS * 256) return;
  size_t row = i / (HEADS * 256), h = (i / 256) % HEADS, d = i % 256;
  out[i] = x[(h * rows + row) * 256 + d];
}

// Index-pool scores, identical arithmetic to the reference kernels.
// grid (ceil(max_pools / 128), rows), 128 threads. Row t sits at absolute
// position start + t and sees (start + t + 1) / 4 complete pools.
__global__ void scores(float* out, const B* q, const B* pools, const B* weights, unsigned start, size_t stride,
                       const unsigned* position) {
  if (position) start = *position;
  __shared__ float qs[32 * 128];
  __shared__ float ws[32];
  unsigned t = blockIdx.y, np = (start + t + 1) / 4, p = blockIdx.x * 128 + threadIdx.x;
  if (blockIdx.x * 128 >= np) return;
  for (unsigned i = threadIdx.x; i < 32 * 128; i += 128) qs[i] = f(q[size_t(t) * 4096 + i]);
  if (threadIdx.x < 32) ws[threadIdx.x] = f(weights[t * 32 + threadIdx.x]);
  __syncthreads();
  if (p >= np) return;
  float key[128];
  const uint4* src = reinterpret_cast<const uint4*>(pools + size_t(p) * 128);
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    uint4 r = src[i];
    const B* v = reinterpret_cast<const B*>(&r);
#pragma unroll
    for (int j = 0; j < 8; ++j) key[i * 8 + j] = f(v[j]);
  }
  float sum = 0.f;
  for (unsigned h = 0; h < 32; ++h) {
    float dot = 0.f;
#pragma unroll
    for (unsigned c = 0; c < 128; ++c) dot += qs[h * 128 + c] * key[c];
    sum += fmaxf(0.f, dot / sqrtf(128.f)) * ws[h] / sqrtf(32.f);
  }
  out[t * stride + p] = sum;
}
__device__ __forceinline__ uint32_t order(float x) {
  uint32_t u = __float_as_uint(x);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
// Exclusive block prefix sum over 1024 threads.
__device__ unsigned scan(unsigned v, unsigned* warps, unsigned* total) {
  unsigned lane = threadIdx.x % 32, warp = threadIdx.x / 32, x = v;
  for (int o = 1; o < 32; o *= 2) { unsigned y = __shfl_up_sync(~0u, x, o); if (lane >= o) x += y; }
  if (lane == 31) warps[warp] = x;
  __syncthreads();
  if (warp == 0) {
    unsigned w = warps[lane], s = w;
    for (int o = 1; o < 32; o *= 2) { unsigned y = __shfl_up_sync(~0u, s, o); if (lane >= o) s += y; }
    warps[lane] = s - w;
    if (lane == 31) warps[32] = s;
  }
  __syncthreads();
  unsigned result = x - v + warps[warp];
  *total = warps[32];
  __syncthreads();
  return result;
}
// Top-512 pools by score (ties: lowest index), then the incomplete tail, all
// in ascending token order. grid rows, 1024 threads.
__global__ void __launch_bounds__(1024) topk(int* ids, const float* score, unsigned start, size_t stride,
                                             const unsigned* position) {
  if (position) start = *position;
  __shared__ unsigned hist[256];
  __shared__ unsigned warps[33];
  __shared__ uint32_t prefix_s;
  __shared__ unsigned need_s;
  const unsigned t = blockIdx.x, length = start + t + 1, np = length / 4, tid = threadIdx.x;
  int* out = ids + size_t(t) * WIDTH;
  const float* s = score + t * stride;
  unsigned pools;
  if (np <= 512) {
    for (unsigned j = tid; j < np * 4; j += 1024) out[j] = j;
    pools = np;
  } else {
    uint32_t prefix = 0, mask = 0;
    unsigned need = 512;
    for (int shift = 24; shift >= 0; shift -= 8) {
      for (unsigned i = tid; i < 256; i += 1024) hist[i] = 0;
      __syncthreads();
      for (unsigned p = tid; p < np; p += 1024) {
        uint32_t k = order(s[p]);
        if ((k & mask) == prefix) atomicAdd(&hist[(k >> shift) & 255], 1u);
      }
      __syncthreads();
      if (tid == 0) {
        unsigned cum = 0;
        for (int bin = 255; bin >= 0; --bin) {
          if (cum + hist[bin] >= need) { prefix |= uint32_t(bin) << shift; need -= cum; break; }
          cum += hist[bin];
        }
        prefix_s = prefix;
        need_s = need;
      }
      __syncthreads();
      prefix = prefix_s;
      need = need_s;
      mask |= 255u << shift;
    }
    // prefix is the 512th largest key; take every larger key and the first
    // `need` keys equal to it.
    unsigned written = 0, equal = 0;
    for (unsigned chunk = 0; chunk < np; chunk += 1024) {
      unsigned p = chunk + tid;
      uint32_t k = p < np ? order(s[p]) : 0;
      bool gt = p < np && k > prefix, eq = p < np && k == prefix;
      unsigned eq_total, take_total;
      unsigned before = scan(eq, warps, &eq_total);
      bool take = gt || (eq && equal + before < need);
      unsigned at = scan(take, warps, &take_total);
      if (take)
        for (unsigned c = 0; c < 4; ++c) out[(written + at) * 4 + c] = p * 4 + c;
      written += take_total;
      equal += eq_total;
    }
    pools = 512;
  }
  __syncthreads();
  for (unsigned c = np * 4 + tid; c < length; c += 1024) out[pools * 4 + (c - np * 4)] = c;
  for (unsigned j = pools * 4 + (length - np * 4) + tid; j < WIDTH; j += 1024) out[j] = -1;
}
}  // namespace

#define END return cudaGetLastError()
extern "C" int hayate_glm_mla_prepare(void* wkt, void* wv, const void* kvb, cudaStream_t st) {
  if (!wkt || !wv || !kvb) return cudaErrorInvalidValue;
  prepare<<<(HEADS * 512 * 512 + 255) / 256, 256, 0, st>>>((B*)wkt, (B*)wv, (const B*)kvb);
  END;
}
extern "C" int hayate_glm_mla_heads(void* out, const void* x, unsigned rows, int merge, cudaStream_t st) {
  if (!out || !x || !rows) return cudaErrorInvalidValue;
  size_t n = size_t(rows) * HEADS * 256;
  if (merge) heads_merge<<<(n + 255) / 256, 256, 0, st>>>((B*)out, (const B*)x, rows);
  else heads_split<<<(n + 255) / 256, 256, 0, st>>>((B*)out, (const B*)x, rows);
  END;
}
extern "C" int hayate_glm_mla_absorb_q(void* qabs, const void* q, const void* wkt, unsigned rows, cudaStream_t st) {
  if (!qabs || !q || !wkt || rows < 1 || rows > 4) return cudaErrorInvalidValue;
  unsigned grid = HEADS * LAT / 8;
  switch (rows) {
    case 1: absorb_q<1><<<grid, 256, 0, st>>>((B*)qabs, (const B*)q, (const B*)wkt); break;
    case 2: absorb_q<2><<<grid, 256, 0, st>>>((B*)qabs, (const B*)q, (const B*)wkt); break;
    case 3: absorb_q<3><<<grid, 256, 0, st>>>((B*)qabs, (const B*)q, (const B*)wkt); break;
    default: absorb_q<4><<<grid, 256, 0, st>>>((B*)qabs, (const B*)q, (const B*)wkt); break;
  }
  END;
}
extern "C" int hayate_glm_mla_absorb_out(void* out, const void* olat, const void* wv, unsigned rows, cudaStream_t st) {
  if (!out || !olat || !wv || rows < 1 || rows > 4) return cudaErrorInvalidValue;
  unsigned grid = HEADS * 256 / 8;
  switch (rows) {
    case 1: absorb_out<1><<<grid, 256, 0, st>>>((B*)out, (const B*)olat, (const B*)wv); break;
    case 2: absorb_out<2><<<grid, 256, 0, st>>>((B*)out, (const B*)olat, (const B*)wv); break;
    case 3: absorb_out<3><<<grid, 256, 0, st>>>((B*)out, (const B*)olat, (const B*)wv); break;
    default: absorb_out<4><<<grid, 256, 0, st>>>((B*)out, (const B*)olat, (const B*)wv); break;
  }
  END;
}
// splits == 1: out = normalized o_lat. splits > 1: part [splits][64][rows][512]
// F32 and stats [splits][64][rows][2] scratch, combined into out.
extern "C" int hayate_glm_mla_attention(void* out, void* part, void* stats, const void* qabs, const void* latent,
                                        const void* ids, unsigned rows, unsigned splits, cudaStream_t st) {
  if (!out || !qabs || !latent || !ids || !rows || !splits || splits > 64 || (splits > 1 && (!part || !stats)))
    return cudaErrorInvalidValue;
  static bool configured = false;
  if (!configured) {
    auto e = cudaFuncSetAttribute(attention, cudaFuncAttributeMaxDynamicSharedMemorySize, int(SMEM));
    if (e) return e;
    configured = true;
  }
  attention<<<dim3(rows, HEADS / HG, splits), 256, SMEM, st>>>((B*)out, (float*)part, (float*)stats, (const B*)qabs,
                                                               (const B*)latent, (const int*)ids, rows);
  if (splits == 1) END;
  auto e = cudaGetLastError();
  if (e) return e;
  combine<<<HEADS * rows, LAT, 0, st>>>((B*)out, (const float*)part, (const float*)stats, rows, splits);
  END;
}
// rows query rows at absolute positions start..start+rows-1; scores scratch
// [rows][stride] with stride >= (start + rows) / 4; ids [rows][2051].
extern "C" int hayate_glm_mla_select(void* ids, void* score, const void* q, const void* pools, const void* weights,
                                     unsigned start, unsigned rows, size_t stride, cudaStream_t st) {
  if (!ids || !score || !q || !pools || !weights || !rows || stride < (start + rows) / 4) return cudaErrorInvalidValue;
  unsigned most = (start + rows) / 4;
  if (most) {
    scores<<<dim3((most + 127) / 128, rows), 128, 0, st>>>((float*)score, (const B*)q, (const B*)pools,
                                                          (const B*)weights, start, stride, nullptr);
    auto e = cudaGetLastError();
    if (e) return e;
  }
  topk<<<rows, 1024, 0, st>>>((int*)ids, (const float*)score, start, stride, nullptr);
  END;
}
// One decode row at the device position (*position, read when the kernels run;
// graph-capturable): stride = pools of the context, ids [2051].
extern "C" int hayate_glm_mla_select_device(void* ids, void* score, const void* q, const void* pools,
                                            const void* weights, const void* position, size_t stride, cudaStream_t st) {
  if (!ids || !score || !q || !pools || !weights || !position || !stride) return cudaErrorInvalidValue;
  scores<<<dim3((stride + 127) / 128, 1), 128, 0, st>>>((float*)score, (const B*)q, (const B*)pools,
                                                       (const B*)weights, 0, stride, (const unsigned*)position);
  auto e = cudaGetLastError();
  if (e) return e;
  topk<<<1, 1024, 0, st>>>((int*)ids, (const float*)score, 0, stride, (const unsigned*)position);
  END;
}
