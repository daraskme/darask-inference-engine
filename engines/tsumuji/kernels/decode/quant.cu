// Load-time conversion of dense bf16 matrices to MXFP8: e4m3 values with one ue8m0 scale
// (a power of two, byte = exponent + 127) per 32 consecutive elements of a row.
//
// The scale is the smallest power of two that keeps the block's largest magnitude within
// e4m3's 448, so nothing saturates and the block uses the top of the e4m3 range. Dividing by a
// power of two is exact; the e4m3 rounding (round to nearest even) is the only error, and
// dequantization (e4m3 * 2^e) is exact in fp32.
#include "decode.cuh"

// One thread per 32-element block. q: [O][K] bytes, sc: [O][K/32] bytes. K % 32 == 0.
extern "C" __global__ void quant_mxfp8(const __nv_bfloat16* __restrict__ w, uint8_t* __restrict__ q,
                                       uint8_t* __restrict__ sc, size_t blocks) {
  const size_t b = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (b >= blocks) return;
  const __nv_bfloat16* src = w + b * 32;
  float v[32];
  float amax = 0.f;
#pragma unroll
  for (int i = 0; i < 32; ++i) {
    v[i] = __bfloat162float(src[i]);
    amax = fmaxf(amax, fabsf(v[i]));
  }
  // e = ceil(log2(amax / 448)), clamped to the ue8m0 range.
  int e = -127;
  if (amax > 0.f) {
    e = int(ceilf(log2f(amax / 448.f)));
    while (ldexpf(448.f, e) < amax) ++e;       // guard log2f rounding
    while (e > -126 && ldexpf(448.f, e - 1) >= amax) --e;
    e = max(-126, min(127, e));  // byte 0 is reserved for all-zero blocks (decoded as 0)
  }
  sc[b] = uint8_t(e + 127);
  uint8_t* dst = q + b * 32;
#pragma unroll
  for (int i = 0; i < 32; ++i)
    dst[i] = __nv_cvt_float_to_fp8(ldexpf(v[i], -e), __NV_SATFINITE, __NV_E4M3);
}

// MXINT8 (OCP MX): int8 values in [-127, 127] with the same ue8m0 scale per 32 elements, the
// smallest power of two with max |w| / 2^e <= 127. Round to nearest even. The int8 grid is
// uniform, so for weight-like blocks the error is several times below e4m3's 3-bit mantissa, at
// the same bytes; an int8 value times a power of two is exact in bf16 and fp32.
extern "C" __global__ void quant_mxint8(const __nv_bfloat16* __restrict__ w, int8_t* __restrict__ q,
                                        uint8_t* __restrict__ sc, size_t blocks) {
  const size_t b = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (b >= blocks) return;
  const __nv_bfloat16* src = w + b * 32;
  float v[32];
  float amax = 0.f;
#pragma unroll
  for (int i = 0; i < 32; ++i) {
    v[i] = __bfloat162float(src[i]);
    amax = fmaxf(amax, fabsf(v[i]));
  }
  int e = -127;
  if (amax > 0.f) {
    e = int(ceilf(log2f(amax / 127.f)));
    while (ldexpf(127.f, e) < amax) ++e;
    while (e > -126 && ldexpf(127.f, e - 1) >= amax) --e;
    e = max(-126, min(127, e));  // byte 0 is reserved for all-zero blocks (decoded as 0)
  }
  sc[b] = uint8_t(e + 127);
  int8_t* dst = q + b * 32;
#pragma unroll
  for (int i = 0; i < 32; ++i) dst[i] = int8_t(max(-127.f, min(127.f, rintf(ldexpf(v[i], -e)))));
}

// Load-time NVFP4 quantization of bf16 expert matrices (the MTP layer's experts, which the
// checkpoint keeps in bf16), into the routed-expert layout of the main layers (weights.rs):
// e2m1 codes (two per byte, low nibble first) with one e4m3 scale per 16 elements of a row and one
// fp32 global scale per matrix, value = code * scale * global. Round to nearest, as ModelOpt:
// global = amax / (6 * 448), scale = e4m3(block amax / 6 / global), code = e2m1(x / (scale * global)).
// Matrix m of the batch (grid.y) reads src + m * src_stride (elements, rows x cols) and writes its
// codes to dst + m * dst_stride + co (rows x cols / 2 bytes), scales to dst + m * dst_stride + so
// (rows x cols / 16 bytes) and the global scale to gs[m * gs_stride].

// amax[m] = max |x| of matrix m, as float bits (non-negative floats order like their bits).
extern "C" __global__ void nvfp4_amax(const __nv_bfloat16* __restrict__ src, size_t src_stride, size_t n,
                                      unsigned* __restrict__ amax) {
  const __nv_bfloat16* w = src + blockIdx.y * src_stride;
  float a = 0.f;
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
    a = fmaxf(a, fabsf(__bfloat162float(w[i])));
  a = warp_max(a);
  if ((threadIdx.x & 31) == 0) atomicMax(amax + blockIdx.y, __float_as_uint(a));
}

// One thread per 16-element block of matrix blockIdx.y.
extern "C" __global__ void nvfp4_quant(const __nv_bfloat16* __restrict__ src, size_t src_stride, int rows, int cols,
                                       const unsigned* __restrict__ amax, uint8_t* __restrict__ dst,
                                       size_t dst_stride, size_t co, size_t so, float* __restrict__ gs,
                                       int gs_stride) {
  const int m = blockIdx.y;
  const size_t b = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  float global = __uint_as_float(amax[m]) / (6.f * 448.f);
  if (global == 0.f) global = 1.f;  // all-zero matrix
  if (b == 0) gs[m * gs_stride] = global;
  if (b >= size_t(rows) * cols / 16) return;
  const __nv_bfloat16* x = src + m * src_stride + b * 16;
  float v[16], a = 0.f;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    v[i] = __bfloat162float(x[i]);
    a = fmaxf(a, fabsf(v[i]));
  }
  const uint8_t s = __nv_cvt_float_to_fp8(a / 6.f / global, __NV_SATFINITE, __NV_E4M3);
  const float d = e4m3_to_float(s) * global;
  uint8_t* out = dst + m * dst_stride;
  out[so + b] = s;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const float2 q = d > 0.f ? make_float2(v[2 * i] / d, v[2 * i + 1] / d) : make_float2(0.f, 0.f);
    out[co + b * 8 + i] = __nv_cvt_float2_to_fp4x2(q, __NV_E2M1, cudaRoundNearest);
  }
}
