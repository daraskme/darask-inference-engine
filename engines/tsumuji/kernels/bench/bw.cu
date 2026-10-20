// M0 bench (1) and (5): streaming read bandwidth over a buffer of `n` uint4.
// Small buffers read repeatedly measure L2; large ones measure DRAM.
#include "tsumuji.cuh"

// Grid-stride read, 4 independent 16B loads in flight per thread per iteration.
// `sink` is written only if the XOR is a value no real buffer produces, so the loads stay.
extern "C" __global__ void __launch_bounds__(256) bw_read(const uint4* __restrict__ p, uint64_t n,
                                                         uint32_t* __restrict__ sink) {
  const uint64_t stride = uint64_t(gridDim.x) * blockDim.x;
  uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  uint32_t acc = 0;
  for (; i + 3 * stride < n; i += 4 * stride) {
    const uint4 a = p[i], b = p[i + stride], c = p[i + 2 * stride], d = p[i + 3 * stride];
    acc ^= xor4(a) ^ xor4(b) ^ xor4(c) ^ xor4(d);
  }
  for (; i < n; i += stride) acc ^= xor4(p[i]);
  if (acc == 0x9E3779B9u) sink[0] = acc;
}

// Fills a buffer with SplitMix64 noise. Constant fills would let compressible-memory paths
// inflate the bandwidth numbers.
extern "C" __global__ void fill_random(uint4* __restrict__ p, uint64_t n, uint64_t seed) {
  const uint64_t stride = uint64_t(gridDim.x) * blockDim.x;
  for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
    uint64_t a = (seed + i) * 0x9E3779B97F4A7C15ull, b = a ^ 0xD1B54A32D192ED03ull;
    a = (a ^ (a >> 30)) * 0xBF58476D1CE4E5B9ull;
    a = (a ^ (a >> 27)) * 0x94D049BB133111EBull;
    a ^= a >> 31;
    b = (b ^ (b >> 30)) * 0xBF58476D1CE4E5B9ull;
    b = (b ^ (b >> 27)) * 0x94D049BB133111EBull;
    b ^= b >> 31;
    p[i] = make_uint4(uint32_t(a), uint32_t(a >> 32), uint32_t(b), uint32_t(b >> 32));
  }
}

// Same with evict-first loads (`ld.global.cs`): what a one-pass weight stream should use.
extern "C" __global__ void __launch_bounds__(256) bw_read_cs(const uint4* __restrict__ p, uint64_t n,
                                                            uint32_t* __restrict__ sink) {
  const uint64_t stride = uint64_t(gridDim.x) * blockDim.x;
  uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  uint32_t acc = 0;
  for (; i + 3 * stride < n; i += 4 * stride) {
    const uint4 a = __ldcs(p + i), b = __ldcs(p + i + stride), c = __ldcs(p + i + 2 * stride),
                d = __ldcs(p + i + 3 * stride);
    acc ^= xor4(a) ^ xor4(b) ^ xor4(c) ^ xor4(d);
  }
  for (; i < n; i += stride) acc ^= xor4(__ldcs(p + i));
  if (acc == 0x9E3779B9u) sink[0] = acc;
}
