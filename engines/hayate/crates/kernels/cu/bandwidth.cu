#include <cuda_runtime.h>

namespace {

constexpr int kThreads = 256;
constexpr int kUnroll = 4;

// Coalesced 16-byte reads/writes; the working set exceeds either target's L2.
__global__ void bandwidth_copy(uint4* __restrict__ dst,
                               const uint4* __restrict__ src, size_t count) {
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  // Issue four independent vector loads before their stores to hide latency.
  for (; i + (kUnroll - 1) * stride < count; i += kUnroll * stride) {
    uint4 values[kUnroll];
#pragma unroll
    for (int j = 0; j < kUnroll; ++j) values[j] = src[i + j * stride];
#pragma unroll
    for (int j = 0; j < kUnroll; ++j) dst[i + j * stride] = values[j];
  }
  for (; i < count; i += stride) {
    dst[i] = src[i];
  }
}

}  // namespace

// Queried once before timing; kernel submission does not query or synchronize.
extern "C" int hayate_bandwidth_blocks(int* blocks) {
  int sms = 0;
  cudaError_t error = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  if (error != cudaSuccess) return static_cast<int>(error);
  int resident = 0;
  error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, bandwidth_copy,
                                                      kThreads, 0);
  if (error != cudaSuccess) return static_cast<int>(error);
  *blocks = sms * resident;
  return static_cast<int>(cudaSuccess);
}

extern "C" int hayate_bandwidth_copy(void* dst, const void* src, size_t bytes,
                                      int blocks, cudaStream_t stream) {
  bandwidth_copy<<<blocks, kThreads, 0, stream>>>(static_cast<uint4*>(dst),
                                         static_cast<const uint4*>(src),
                                         bytes / sizeof(uint4));
  return static_cast<int>(cudaGetLastError());
}
