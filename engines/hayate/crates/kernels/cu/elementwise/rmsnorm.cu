// Reference-quality RMSNorm used to validate the build pipeline end to end.
// The production path fuses this into neighbouring kernels (ARCHITECTURE.md §6).

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kThreads = 256;

__global__ void rmsnorm_bf16(__nv_bfloat16* __restrict__ y,
                             const __nv_bfloat16* __restrict__ x,
                             const __nv_bfloat16* __restrict__ w,
                             unsigned cols, float eps) {
  const unsigned row = blockIdx.x;
  const __nv_bfloat16* xr = x + static_cast<size_t>(row) * cols;
  __nv_bfloat16* yr = y + static_cast<size_t>(row) * cols;

  float sum = 0.f;
  for (unsigned i = threadIdx.x; i < cols; i += kThreads) {
    float v = __bfloat162float(xr[i]);
    sum += v * v;
  }

  __shared__ float warp_sums[kThreads / 32];
  for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, off);
  if ((threadIdx.x & 31) == 0) warp_sums[threadIdx.x >> 5] = sum;
  __syncthreads();
  if (threadIdx.x < 32) {
    sum = threadIdx.x < kThreads / 32 ? warp_sums[threadIdx.x] : 0.f;
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, off);
    if (threadIdx.x == 0) warp_sums[0] = sum;
  }
  __syncthreads();

  const float inv = rsqrtf(warp_sums[0] / cols + eps);
  for (unsigned i = threadIdx.x; i < cols; i += kThreads) {
    yr[i] = __float2bfloat16(__bfloat162float(xr[i]) * inv * __bfloat162float(w[i]));
  }
}

}  // namespace

extern "C" int hayate_rmsnorm_bf16(void* y, const void* x, const void* w, unsigned rows,
                                   unsigned cols, float eps, cudaStream_t stream) {
  rmsnorm_bf16<<<rows, kThreads, 0, stream>>>(static_cast<__nv_bfloat16*>(y),
                                              static_cast<const __nv_bfloat16*>(x),
                                              static_cast<const __nv_bfloat16*>(w), cols, eps);
  return static_cast<int>(cudaGetLastError());
}
