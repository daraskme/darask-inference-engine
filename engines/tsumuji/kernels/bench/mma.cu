// M0 bench (6): sustained mma.sync throughput on sm_120a for the three formats the plan weighs
// for prefill: BF16 (W4A16 after exact dequant), FP8 e4m3 (W8A8), and block-scaled NVFP4 (W4A4).
// Operands stay in registers; 8 independent accumulators per warp hide the MMA latency.
#include "tsumuji.cuh"

constexpr int ACC = 8;

__device__ __forceinline__ void store_sink(float (&d)[ACC][4], float* sink) {
  float s = 0.f;
#pragma unroll
  for (int a = 0; a < ACC; ++a) s += d[a][0] + d[a][1] + d[a][2] + d[a][3];
  if (s == 1234.5f) sink[0] = s;  // keeps the MMAs live
}

// m16n8k16, bf16 x bf16 -> f32: 4096 flop per instruction.
extern "C" __global__ void __launch_bounds__(128) mma_bf16(int iters, uint32_t seed, float* sink) {
  const uint32_t one = 0x3F803F80u ^ (seed & (threadIdx.x == 999));  // bf16x2 (1, 1)
  uint32_t a[4] = {one, one, one, one}, b[2] = {one, one};
  float d[ACC][4] = {};
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int k = 0; k < ACC; ++k)
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
          "{%0,%1,%2,%3};\n"
          : "+f"(d[k][0]), "+f"(d[k][1]), "+f"(d[k][2]), "+f"(d[k][3])
          : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  }
  store_sink(d, sink);
}

// m16n8k32, e4m3 x e4m3 -> f32 (kind::f8f6f4): 8192 flop per instruction.
extern "C" __global__ void __launch_bounds__(128) mma_fp8(int iters, uint32_t seed, float* sink) {
  const uint32_t one = 0x38383838u ^ (seed & (threadIdx.x == 999));  // four e4m3 1.0
  uint32_t a[4] = {one, one, one, one}, b[2] = {one, one};
  float d[ACC][4] = {};
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int k = 0; k < ACC; ++k)
      asm volatile(
          "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, "
          "{%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(d[k][0]), "+f"(d[k][1]), "+f"(d[k][2]), "+f"(d[k][3])
          : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  }
  store_sink(d, sink);
}

// m16n8k64, e2m1 x e2m1 with ue4m3 scales per 16 (NVFP4, scale_vec::4X): 16384 flop per instruction.
extern "C" __global__ void __launch_bounds__(128) mma_nvfp4(int iters, uint32_t seed, float* sink) {
  const uint32_t one = 0x22222222u ^ (seed & (threadIdx.x == 999));  // eight e2m1 1.0
  const uint32_t sf = 0x38383838u;                                    // four ue4m3 1.0
  const uint16_t bid = 0, tid = 0;
  uint32_t a[4] = {one, one, one, one}, b[2] = {one, one};
  float d[ACC][4] = {};
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int k = 0; k < ACC; ++k)
      asm volatile(
          "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
          : "+f"(d[k][0]), "+f"(d[k][1]), "+f"(d[k][2]), "+f"(d[k][3])
          : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sf), "h"(bid),
            "h"(tid), "r"(sf), "h"(bid), "h"(tid));
  }
  store_sink(d, sink);
}
