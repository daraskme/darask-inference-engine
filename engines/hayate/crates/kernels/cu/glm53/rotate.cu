// Orthogonal input rotation for routed NVFP4 weights, with F32 butterflies.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "math.cuh"
namespace {
__global__ void rotate(__nv_bfloat16* out,const __nv_bfloat16* x,unsigned block) {
  __shared__ float z[256];
  unsigned lane=threadIdx.x;
  size_t offset=size_t(blockIdx.x)*block;
  z[lane]=__bfloat162float(x[offset+lane]);
  __syncthreads();
  for(unsigned stride=1;stride<block;stride*=2) {
    float v=glm53::hadamard_value(z[lane&~stride],z[lane|stride],lane,stride);
    __syncthreads();z[lane]=v;__syncthreads();
  }
  out[offset+lane]=__float2bfloat16_rn(z[lane]*(1.f/sqrtf(float(block))));
}
}
extern "C" int hayate_glm_hadamard(void* out,const void* x,unsigned rows,unsigned cols,
    unsigned block,cudaStream_t stream) {
  if(!out||!x||!rows||!cols||!(block==32||block==64||block==128||block==256)||cols%block)
    return cudaErrorInvalidValue;
  rotate<<<size_t(rows)*cols/block,block,0,stream>>>((__nv_bfloat16*)out,(const __nv_bfloat16*)x,block);
  return cudaGetLastError();
}
