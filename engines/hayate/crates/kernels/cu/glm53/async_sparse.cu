#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "async_math.cuh"
#pragma nv_diag_suppress 20013
#pragma nv_diag_suppress 20015
#include <cutlass/gemm/device/gemm.h>
namespace {
__global__ void sort_queries(unsigned* sorted,const int* selected) {
  __shared__ unsigned values[4096];unsigned row=blockIdx.x;
  for(unsigned i=threadIdx.x;i<4096;i+=256)values[i]=i<2051&&selected[row*2051+i]>=0?unsigned(selected[row*2051+i]):~0u;
  __syncthreads();
  for(unsigned width=2;width<=4096;width*=2)for(unsigned stride=width/2;stride;stride/=2){
    for(unsigned i=threadIdx.x;i<4096;i+=256){glm53::sort_pair(values,i,stride,width);}
    __syncthreads();
  }
  for(unsigned i=threadIdx.x;i<4096;i+=256)sorted[row*4096+i]=values[i];
}
__global__ void union_queries(unsigned* ids,unsigned* counts,const unsigned* sorted,unsigned rows) {
  if(threadIdx.x)return;
  glm53::merge_queries(ids,counts,sorted,rows);
}
__global__ void union_maps(unsigned* maps,const unsigned* sorted,const unsigned* ids,const unsigned* counts,unsigned rows) {
  unsigned i=blockIdx.x*256+threadIdx.x;if(i>=rows*2051)return;
  unsigned r=i/2051,c=i%2051;if(c>=counts[1+r]){maps[i]=0;return;}
  unsigned id=sorted[r*4096+c],lo=0,hi=counts[0];
  while(lo<hi){unsigned mid=(lo+hi)/2;if(ids[mid]<id)lo=mid+1;else hi=mid;}
  maps[i]=lo;
}
__global__ void gather_latent(__nv_bfloat16* out,const __nv_bfloat16* host,const unsigned* ids,const unsigned* counts,unsigned bound) {
  unsigned i=blockIdx.x*256+threadIdx.x;if(i>=bound*512)return;
  unsigned row=i/512,c=i%512;
  out[i]=row<counts[0]?host[size_t(ids[row])*512+c]:__float2bfloat16_rn(0.f);
}
using B=cutlass::bfloat16_t;
using Gemm=cutlass::gemm::device::Gemm<B,cutlass::layout::RowMajor,B,cutlass::layout::ColumnMajor,B,cutlass::layout::RowMajor,float,cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,cutlass::gemm::GemmShape<128,128,32>,cutlass::gemm::GemmShape<64,64,32>,cutlass::gemm::GemmShape<16,8,16>>;
using Kernel=Gemm::GemmKernel;
__global__ void union_project(Kernel::Params params,const unsigned* count) {
  extern __shared__ int storage[];
  params.problem_size.m()=*count;
  params.grid_tiled_shape.m()=(*count+127)/128;
  Kernel operation;
  operation(params,*reinterpret_cast<Kernel::SharedStorage*>(storage));
}
}
extern "C" int hayate_glm_sparse_gather(void* out,void* sorted,void* ids,void* maps,void* counts,const void* selected,const void* host,unsigned rows,unsigned bound,cudaStream_t s) {
  if(!out||!sorted||!ids||!maps||!counts||!selected||!host||rows<1||rows>4||!bound||bound>rows*2051)return cudaErrorInvalidValue;
  sort_queries<<<rows,256,0,s>>>((unsigned*)sorted,(const int*)selected);
  auto e=cudaGetLastError();if(e)return e;
  union_queries<<<1,32,0,s>>>((unsigned*)ids,(unsigned*)counts,(const unsigned*)sorted,rows);
  e=cudaGetLastError();if(e)return e;
  union_maps<<<(rows*2051+255)/256,256,0,s>>>((unsigned*)maps,(const unsigned*)sorted,(const unsigned*)ids,(const unsigned*)counts,rows);
  e=cudaGetLastError();if(e)return e;
  gather_latent<<<(bound*512+255)/256,256,0,s>>>((__nv_bfloat16*)out,(const __nv_bfloat16*)host,(const unsigned*)ids,(const unsigned*)counts,bound);return cudaGetLastError();
}
extern "C" int hayate_glm_sparse_project(void* out,const void* x,const void* w,const void* counts,unsigned bound,cudaStream_t s) {
  if(!out||!x||!w||!counts||!bound||bound>4*2051)return cudaErrorInvalidValue;
  cutlass::gemm::GemmCoord problem(bound,32768,512),grid((bound+127)/128,256,1);
  Kernel::Params params(problem,grid,{(B*)x,512},{(B*)w,512},{(B*)out,32768},{(B*)out,32768},{1.f,0.f});
  constexpr unsigned shared=sizeof(Kernel::SharedStorage);
  if constexpr(shared>=48*1024){auto e=cudaFuncSetAttribute(union_project,cudaFuncAttributeMaxDynamicSharedMemorySize,shared);if(e)return e;}
  union_project<<<dim3(grid.m(),grid.n(),1),Kernel::kThreadCount,shared,s>>>(params,(const unsigned*)counts);return cudaGetLastError();
}
