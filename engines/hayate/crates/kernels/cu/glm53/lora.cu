// Unmerged LoRA side path for GLM-5.3 linears: out[row] = bf16(out[row] + scale * B (A x[row])).
// A is [count][rank][k] and B [count][n][rank] in F32, F16 or BF16 (as stored by the adapter);
// count > 1 picks the matrices per row from `experts` (routed experts). Rows whose `images`
// entry is null are skipped (decode CPU experts, which add their LoRA on the host).
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
namespace {
using B = __nv_bfloat16;
struct Lora {
  const void* a; const void* b;
  unsigned long long a_stride, b_stride;
  unsigned rank, k, n, dtype, count;
  float scale;
};
__device__ inline float load(const void* p, size_t i, unsigned dtype) {
  if (dtype == 1) return __half2float(static_cast<const __half*>(p)[i]);
  if (dtype == 2) return __bfloat162float(static_cast<const B*>(p)[i]);
  return static_cast<const float*>(p)[i];
}
__device__ inline bool expert_of(unsigned row, const int* experts, const void* const* images, const Lora& l,
    unsigned& e) {
  if (images && !images[row]) return false;
  e = experts ? unsigned(experts[row]) : 0u;
  return e < l.count;
}
// t[row][j] = A_e[j] . x[row], one warp per (row, j); F32 sums in lane order then a shuffle tree.
__global__ void __launch_bounds__(256) down(float* t, const B* x, const int* experts, const void* const* images,
    Lora l) {
  const unsigned row = blockIdx.x, j = blockIdx.y * 8 + threadIdx.x / 32, lane = threadIdx.x % 32;
  unsigned e;
  if (j >= l.rank || !expert_of(row, experts, images, l, e)) return;
  const B* xr = x + size_t(row) * l.k;
  const size_t base = size_t(e) * l.a_stride + size_t(j) * l.k;
  float sum = 0;
  for (unsigned c = lane; c < l.k; c += 32) sum += load(l.a, base + c, l.dtype) * __bfloat162float(xr[c]);
  for (unsigned o = 16; o; o >>= 1) sum += __shfl_xor_sync(~0u, sum, o);
  if (!lane) t[size_t(row) * l.rank + j] = sum;
}
__global__ void __launch_bounds__(256) up(B* out, const float* t, const int* experts, const void* const* images,
    Lora l) {
  const unsigned c = blockIdx.x * 256 + threadIdx.x, row = blockIdx.y;
  unsigned e;
  if (c >= l.n || !expert_of(row, experts, images, l, e)) return;
  const size_t base = size_t(e) * l.b_stride + size_t(c) * l.rank;
  const float* tr = t + size_t(row) * l.rank;
  float sum = 0;
  for (unsigned j = 0; j < l.rank; ++j) sum += load(l.b, base + j, l.dtype) * tr[j];
  B* o = out + size_t(row) * l.n + c;
  *o = __float2bfloat16_rn(__bfloat162float(*o) + l.scale * sum);
}
}  // namespace
// `t` holds rows x rank F32. Graph-capturable (no allocation or synchronization).
extern "C" int hayate_glm_lora(void* out, const void* x, void* t, Lora l, unsigned rows, const void* experts,
    const void* images, cudaStream_t s) {
  if (!out || !x || !t || !l.a || !l.b || !rows || rows > 65535 || !l.rank || !l.k || !l.n || l.dtype > 2 ||
      !l.count || (l.count > 1 && !experts))
    return cudaErrorInvalidValue;
  down<<<dim3(rows, (l.rank + 7) / 8), 256, 0, s>>>((float*)t, (const B*)x, (const int*)experts,
      (const void* const*)images, l);
  auto e = cudaGetLastError();
  if (e) return e;
  up<<<dim3((l.n + 255) / 256, rows), 256, 0, s>>>((B*)out, (const float*)t, (const int*)experts,
      (const void* const*)images, l);
  return cudaGetLastError();
}
