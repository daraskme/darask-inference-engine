// Helpers shared by the M1 decode kernels (batch 1, runtime dims, FP32 activations).
#pragma once
#include "tsumuji.cuh"

// First statement of every decode kernel: with PDL launches the kernel may start before its
// predecessor finishes, so nothing that depends on earlier kernels (or that they still read) is
// touched before the wait. Then the next kernel may start scheduling.
#ifdef TSJ_PDL_EARLY
// Experiment: let the next kernel start its weight prefetch while this one still waits.
#define TSJ_PDL_ENTRY            \
  do {                           \
    griddep_launch_dependents(); \
    griddep_wait();              \
  } while (0)
#else
#define TSJ_PDL_ENTRY            \
  do {                           \
    griddep_wait();              \
    griddep_launch_dependents(); \
  } while (0)
#endif

TSJ_DEVICE float sigmoidf(float x) { return 1.f / (1.f + expf(-x)); }
TSJ_DEVICE float siluf(float x) { return x / (1.f + expf(-x)); }

// A vector's lane owns VW consecutive floats. XOR within that vector makes a warp's scalar
// reads at a fixed component hit all 32 banks, without padding or changing the lane -> K map.
template <int VW>
TSJ_DEVICE int xvec(int v, int j) {
  return v * VW + (j ^ ((v / (32 / VW)) & (VW - 1)));
}

// V independent warp sums, scattered: lane l gets sum of a[l % V]. The xor stages stay
// 16,8,4,2,1 as in warp_sum; only ownership changes. At each scatter stage the partner selects
// the opposite half before the shuffle, so both operands still belong to the same value.
template <int V>
TSJ_DEVICE float warp_sum_scatter(float (&a)[V]) {
  static_assert(V <= 32 && (V & (V - 1)) == 0);
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int o = 16; o >= V; o >>= 1)
#pragma unroll
    for (int i = 0; i < V; ++i) a[i] += __shfl_xor_sync(0xffffffffu, a[i], o);
#pragma unroll
  for (int o = V / 2; o > 0; o >>= 1) {
#pragma unroll
    for (int i = 0; i < o; ++i) {
      const float own = (lane & o) ? a[i + o] : a[i];
      const float other = (lane & o) ? a[i] : a[i + o];
      a[i] = own + __shfl_xor_sync(0xffffffffu, other, o);
    }
  }
  return a[0];
}
// torch.nn.functional.softplus with beta 1, threshold 20.
TSJ_DEVICE float softplusf(float x) { return x > 20.f ? x : log1pf(expf(x)); }

// Sum over the block; every thread gets the result. `red` needs 32 floats.
TSJ_DEVICE float block_sum(float v, float* red) {
  v = warp_sum(v);
  const int w = threadIdx.x >> 5, l = threadIdx.x & 31, nw = (blockDim.x + 31) >> 5;
  __syncthreads();
  if (l == 0) red[w] = v;
  __syncthreads();
  float t = 0.f;
  for (int i = 0; i < nw; ++i) t += red[i];  // fixed order: deterministic
  return t;
}

TSJ_DEVICE float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

TSJ_DEVICE float block_max(float v, float* red) {
  v = warp_max(v);
  const int w = threadIdx.x >> 5, l = threadIdx.x & 31, nw = (blockDim.x + 31) >> 5;
  __syncthreads();
  if (l == 0) red[w] = v;
  __syncthreads();
  float t = -INFINITY;
  for (int i = 0; i < nw; ++i) t = fmaxf(t, red[i]);
  return t;
}

TSJ_DEVICE float bf16_to_f(__nv_bfloat16 v) { return __bfloat162float(v); }

// Unpacks 8 bf16 (one uint4) into floats.
TSJ_DEVICE void bf16x8(uint4 v, float (&f)[8]) {
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 t = __bfloat1622float2(h[i]);
    f[2 * i] = t.x;
    f[2 * i + 1] = t.y;
  }
}

// Dense weight formats of the decode GEMVs: FMT 0 = bf16 (8 per 16-byte vector), FMT 1 = MXFP8
// (16 e4m3 per vector, one ue8m0 scale byte per 32 elements of a row), FMT 2 = MXINT8.
template <int FMT>
struct Fmt;
template <>
struct Fmt<0> {
  static constexpr int VW = 8;
  TSJ_DEVICE static void dq(uint4 r, float, float (&f)[8]) { bf16x8(r, f); }
};
template <>
struct Fmt<1> {
  static constexpr int VW = 16;
  TSJ_DEVICE static void dq(uint4 r, float s, float (&f)[16]) {
    const __nv_fp8x2_storage_t* p = reinterpret_cast<const __nv_fp8x2_storage_t*>(&r);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(p[i], __NV_E4M3);
      const float2 v = __half22float2(*reinterpret_cast<__half2*>(&h));
      f[2 * i] = v.x * s;
      f[2 * i + 1] = v.y * s;
    }
  }
};

// FMT 2 = MXINT8 (16 int8 per vector, same ue8m0 scales).
template <>
struct Fmt<2> {
  static constexpr int VW = 16;
  // int8 -> float without I2F (quarter rate): flipping the sign bit gives q + 128 as a byte; placed
  // in the low mantissa byte of 2^23 that is the float 2^23 + 128 + q, and one FMA with the scale
  // removes the bias. Every step is exact.
  TSJ_DEVICE static void dq(uint4 r, float s, float (&f)[16]) {
    const uint32_t words[4] = {r.x ^ 0x80808080u, r.y ^ 0x80808080u, r.z ^ 0x80808080u, r.w ^ 0x80808080u};
    const float bias = -8388736.f * s;  // -(2^23 + 128) s, exact for a power-of-two s
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int b = 0; b < 4; ++b)
        f[4 * i + b] = fmaf(__int_as_float(__byte_perm(words[i], 0x4B000000u, 0x7540 + b)), s, bias);
  }
};

// ue8m0 scale byte -> 2^(b - 127) (0 maps to 0, only used by all-zero blocks).
TSJ_DEVICE float ue8m0(uint8_t b) { return __int_as_float(int(b) << 23); }

// Partial dot of one NVFP4 row with x (this lane's share). codes: K/2 bytes, sf: K/16 e4m3.
// Exact products (e2m1 x e4m3 x fp32 x), fp32 sums. The caller warp-sums and applies the
// global scale.
TSJ_DEVICE float nvfp4_row_dot(const uint4* __restrict__ codes, const uint16_t* __restrict__ sf,
                               const float* __restrict__ x, int K, int lane) {
  float acc = 0.f;
  for (int v = lane; v < K / 32; v += 32) {
    const uint4 w = codes[v];
    const uint16_t s = sf[v];
    const uint32_t words[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int g = 0; g < 2; ++g) {
      float part = 0.f;
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        const uint32_t word = words[g * 2 + q];
#pragma unroll
        for (int b = 0; b < 4; ++b) {
          const float2 wf = __half22float2(e2m1x2_to_half2(uint8_t(word >> (8 * b))));
          const int k = v * 32 + g * 16 + q * 8 + b * 2;
          part = fmaf(wf.x, x[k], part);
          part = fmaf(wf.y, x[k + 1], part);
        }
      }
      acc = fmaf(part, e4m3_to_float(uint8_t(s >> (8 * g))), acc);
    }
  }
  return acc;
}

// The NVFP4 dot helpers below read x from shared memory in a padded layout, element k at
// (k / 32) * 33 + k % 32 (`stage_x32`): lanes taking consecutive 32-element vectors then hit
// consecutive banks instead of all hitting one (a 16-way conflict with 16 lanes per row).
TSJ_DEVICE int x32(int v, int off) { return v * 33 + off; }

// Like nvfp4_row_dot, but this lane covers vectors start, start + step, ... of the K / 32 vectors.
TSJ_DEVICE float nvfp4_dot_part(const uint4* __restrict__ codes, const uint16_t* __restrict__ sf,
                                const float* __restrict__ x, int K, int start, int step) {
  float acc = 0.f;
  for (int v = start; v < K / 32; v += step) {
    const uint4 w = codes[v];
    const uint16_t s = sf[v];
    const uint32_t words[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int g = 0; g < 2; ++g) {
      float part = 0.f;
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        const uint32_t word = words[g * 2 + q];
#pragma unroll
        for (int b = 0; b < 4; ++b) {
          const float2 wf = __half22float2(e2m1x2_to_half2(uint8_t(word >> (8 * b))));
          const int k = x32(v, g * 16 + q * 8 + b * 2);
          part = fmaf(wf.x, x[k], part);
          part = fmaf(wf.y, x[k + 1], part);
        }
      }
      acc = fmaf(part, e4m3_to_float(uint8_t(s >> (8 * g))), acc);
    }
  }
  return acc;
}

// nvfp4_dot_part with every load issued up front: this lane's vectors start, start + step, ...
// (at most N of them) go to registers first, so N loads are in flight instead of one.
template <int N>
TSJ_DEVICE float nvfp4_dot_regs(const uint4* __restrict__ codes, const uint16_t* __restrict__ sf,
                                const float* __restrict__ x, int K, int start, int step) {
  uint4 w[N];
  uint16_t s[N];
#pragma unroll
  for (int i = 0; i < N; ++i) {
    const int v = start + i * step;
    if (v < K / 32) {
      w[i] = __ldcs(codes + v);
      s[i] = __ldcs(sf + v);
    }
  }
  float acc = 0.f;
#pragma unroll
  for (int i = 0; i < N; ++i) {
    const int v = start + i * step;
    if (v >= K / 32) break;
    const uint32_t words[4] = {w[i].x, w[i].y, w[i].z, w[i].w};
#pragma unroll
    for (int g = 0; g < 2; ++g) {
      float part = 0.f;
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        const uint32_t word = words[g * 2 + q];
#pragma unroll
        for (int b = 0; b < 4; ++b) {
          const float2 wf = __half22float2(e2m1x2_to_half2(uint8_t(word >> (8 * b))));
          const int k = x32(v, g * 16 + q * 8 + b * 2);
          part = fmaf(wf.x, x[k], part);
          part = fmaf(wf.y, x[k + 1], part);
        }
      }
      acc = fmaf(part, e4m3_to_float(uint8_t(s[i] >> (8 * g))), acc);
    }
  }
  // Vectors beyond N * step (longer rows than planned for) are handled the slow way.
  return acc + nvfp4_dot_part(codes, sf, x, K, start + N * step, step);
}

// Sum over the G lanes of a group (G a power of two <= 32, groups aligned).
template <int G>
TSJ_DEVICE float group_sum(float v) {
#pragma unroll
  for (int o = G / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// NeoX-style RoPE on the first `rope` dims of a head held in `x` (fp32, in place), at position
// `pos`, with inv_freq computed exactly like transformers (fp32 pow, fp32 angle, accurate sin/cos).
TSJ_DEVICE void rope_inplace(float* x, int rope, int pos, float theta, int tid, int nthreads) {
  const int half = rope / 2;
  for (int i = tid; i < half; i += nthreads) {
    const float inv = 1.f / powf(theta, float(2 * i) / float(rope));
    const float ang = float(pos) * inv;
    const float c = cosf(ang), s = sinf(ang);
    const float a = x[i], b = x[i + half];
    x[i] = a * c - b * s;
    x[i + half] = b * c + a * s;
  }
}
