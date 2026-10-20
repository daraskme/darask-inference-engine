// M0 bench (3): batch-1 W4A16 GEMV on NVFP4 weights, plain CUDA cores (the baseline the
// swap-AB MMA kernel of M1 must beat). One warp per output row.
//
// Layout (modelopt NVFP4, as stored in the checkpoint):
//   w  [N][K/2]  e2m1 pairs, low nibble = even k
//   sf [N][K/16] e4m3 block scales
//   gs           fp32 global scale (weight_scale_2)
// y[n] = gs * sum_g sf[n][g] * sum_{k in g} e2m1(w[n][k]) * x[k], exact products, fp32 sums.
#include "tsumuji.cuh"

template <int K>
__device__ __forceinline__ void gemv_nvfp4_impl(const uint4* __restrict__ w, const uint16_t* __restrict__ sf,
                                                float gs, const __nv_bfloat16* __restrict__ x,
                                                float* __restrict__ y, int n_rows) {
  constexpr int VEC = K / 32;                 // uint4 per row: 32 e2m1 each
  constexpr int PER_LANE = (VEC + 31) / 32;
  __shared__ __align__(16) __nv_bfloat162 xs[K / 2];
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  const bool live = row < n_rows;

  // Weights and scales do not depend on the previous kernel: issue them before the wait.
  uint4 wv[PER_LANE];
  uint16_t sv[PER_LANE];
#pragma unroll
  for (int j = 0; j < PER_LANE; ++j) {
    const int v = lane + 32 * j;
    if (live && v < VEC) {
      wv[j] = __ldcs(w + size_t(row) * VEC + v);
      sv[j] = __ldcs(sf + size_t(row) * (K / 32) + v);  // two e4m3 scales per uint4
    }
  }
  griddep_wait();
  for (int i = threadIdx.x; i < K / 8; i += blockDim.x)
    reinterpret_cast<uint4*>(xs)[i] = reinterpret_cast<const uint4*>(x)[i];
  __syncthreads();
  if (!live) return;

  float acc = 0.f;
#pragma unroll
  for (int j = 0; j < PER_LANE; ++j) {
    const int v = lane + 32 * j;
    if (v >= VEC) break;
    const uint32_t words[4] = {wv[j].x, wv[j].y, wv[j].z, wv[j].w};
#pragma unroll
    for (int g = 0; g < 2; ++g) {           // two 16-element scale groups per uint4
      float part = 0.f;
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        const uint32_t word = words[g * 2 + q];
#pragma unroll
        for (int b = 0; b < 4; ++b) {
          const float2 wf = __half22float2(e2m1x2_to_half2(uint8_t(word >> (8 * b))));
          const float2 xf = __bfloat1622float2(xs[v * 16 + g * 8 + q * 4 + b]);
          part = fmaf(wf.x, xf.x, part);
          part = fmaf(wf.y, xf.y, part);
        }
      }
      acc = fmaf(part, e4m3_to_float(uint8_t(sv[j] >> (8 * g))), acc);
    }
  }
  acc = warp_sum(acc);
  if (lane == 0) y[row] = acc * gs;
}

extern "C" __global__ void __launch_bounds__(128) gemv_nvfp4_k2560(
    const uint4* __restrict__ w, const uint16_t* __restrict__ sf, float gs,
    const __nv_bfloat16* __restrict__ x, float* __restrict__ y, int n_rows) {
  gemv_nvfp4_impl<2560>(w, sf, gs, x, y, n_rows);
}
