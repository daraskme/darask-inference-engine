// M0 bench (4): PLE-style gathers. Each token reads 16 random 160-byte rows (one per n-gram
// head) from a table that lives either in pinned host memory mapped into the device (zero-copy
// over PCIe) or in device memory, and writes them out contiguously.
// grid = T tokens, block = 16 heads x 10 lanes (one 16-byte lane = 16 FP8 values).
#include "tsumuji.cuh"

__device__ __forceinline__ uint64_t mix64(uint64_t v) {
  v += 0x9E3779B97F4A7C15ull;
  v = (v ^ (v >> 30)) * 0xBF58476D1CE4E5B9ull;
  v = (v ^ (v >> 27)) * 0x94D049BB133111EBull;
  return v ^ (v >> 31);
}

extern "C" __global__ void __launch_bounds__(160) ple_gather_probe(const uint4* __restrict__ rows,
                                                                  uint64_t n_rows, uint64_t seed,
                                                                  uint4* __restrict__ out) {
  const int t = blockIdx.x, h = threadIdx.x / 10, l = threadIdx.x % 10;
  const uint64_t row = mix64(seed ^ (uint64_t(t) * 16 + h)) % n_rows;
  out[(size_t(t) * 16 + h) * 10 + l] = rows[row * 10 + l];
}
