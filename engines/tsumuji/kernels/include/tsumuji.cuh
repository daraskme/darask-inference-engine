// Shared device helpers for every Tsumuji kernel. sm_120a only.
#pragma once
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include "q38.h"

#define TSJ_DEVICE __device__ __forceinline__

// Programmatic Dependent Launch. `griddep_wait` blocks until every prerequisite grid has
// completed and its memory is visible; without a PDL launch attribute it returns at once.
TSJ_DEVICE void griddep_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
// Lets the next PDL-launched kernel in the stream start scheduling its blocks.
TSJ_DEVICE void griddep_launch_dependents() {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

TSJ_DEVICE float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

TSJ_DEVICE uint32_t xor4(uint4 v) { return v.x ^ v.y ^ v.z ^ v.w; }

// Two packed e2m1 values (low nibble first) -> half2. Exact.
TSJ_DEVICE __half2 e2m1x2_to_half2(uint8_t b) {
  __half2_raw r = __nv_cvt_fp4x2_to_halfraw2(static_cast<__nv_fp4x2_storage_t>(b), __NV_E2M1);
  return *reinterpret_cast<__half2*>(&r);
}

// One e4m3 scale -> float. Exact.
TSJ_DEVICE float e4m3_to_float(uint8_t b) {
  __half_raw r = __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(b), __NV_E4M3);
  return __half2float(*reinterpret_cast<__half*>(&r));
}
