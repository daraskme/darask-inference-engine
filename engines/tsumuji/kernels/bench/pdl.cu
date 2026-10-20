// M0 bench (2): the gap between dependent decode kernels, with and without PDL.
//
// `chain_step` stands in for one decode GEMV: every thread owns VPT uint4 of this step's weights
// (distinct per step, so they come from DRAM), and the step depends on one float the previous
// step wrote. With PDL the weight loads are issued before `griddep_wait`, so they overlap the
// tail of the previous kernel; without it they start only after the previous kernel retires.
#include "tsumuji.cuh"

template <int VPT>
__device__ __forceinline__ void chain_step_impl(const uint4* __restrict__ w, const float* __restrict__ in,
                                                float* __restrict__ out) {
  griddep_launch_dependents();
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  const int threads = gridDim.x * blockDim.x;
  uint4 r[VPT];
#pragma unroll
  for (int j = 0; j < VPT; ++j) r[j] = __ldcs(w + t + j * threads);
  griddep_wait();
  const float x = in[0];
  float s = 0.f;
#pragma unroll
  for (int j = 0; j < VPT; ++j) s += float(xor4(r[j]) & 0xffu);
  s = warp_sum(s * x);
  __shared__ float part[32];
  if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = s;
  __syncthreads();
  if (threadIdx.x == 0) {
    float b = 0.f;
    for (int i = 0; i < int(blockDim.x >> 5); ++i) b += part[i];
    // Block 0 forwards a value that keeps the next step's input finite and data-dependent.
    if (blockIdx.x == 0) out[0] = 1.f + 1e-30f * b;
  }
}

#define CHAIN(VPT)                                                                              \
  extern "C" __global__ void __launch_bounds__(256)                                             \
      chain_step_##VPT(const uint4* __restrict__ w, const float* __restrict__ in, float* __restrict__ out) { \
    chain_step_impl<VPT>(w, in, out);                                                          \
  }
CHAIN(1)
CHAIN(2)
CHAIN(4)
CHAIN(8)
CHAIN(16)

// Minimal dependent kernel: measures pure launch/dependency overhead.
extern "C" __global__ void chain_empty(const float* __restrict__ in, float* __restrict__ out) {
  griddep_launch_dependents();
  griddep_wait();
  if (blockIdx.x == 0 && threadIdx.x == 0) out[0] = in[0] + 1.f;
}
