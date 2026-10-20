#include <cuda_runtime.h>
#include "dense.cuh"
#include "async_math.cuh"
namespace {
using namespace glm53;
using Group=AsyncGroup;
__global__ void resolve(void** images,const int* ids,const void* const* table,unsigned rows) {
  unsigned i=threadIdx.x;if(i<rows*8)images[i]=const_cast<void*>(table[ids[i]]);
}
__global__ void merge(B* out,const B* host,const int* ids,const void* const* table,unsigned rows) {
  unsigned i=blockIdx.x*256+threadIdx.x;if(i>=rows*8*4096)return;
  unsigned a=i/4096,row=a/8,slot=a%8,c=i%4096;
  if(!table[ids[a]])out[(row*9+slot)*4096+c]=host[(row*9+slot)*4096+c];
}
__global__ void group_routes(Group* groups,unsigned* count,const int* ids,const void* const* table,unsigned rows) {
  if(threadIdx.x)return;
  *count=async_groups(groups,ids,table,rows);
}
__global__ void gather_groups(B* gathered,const B* x,const Group* groups,const unsigned* count) {
  unsigned i=blockIdx.x*256+threadIdx.x;if(i>=(*count)*4*4096)return;
  unsigned g=i/(4*4096),row=i/4096%4,c=i%4096;
  gathered[i]=row<groups[g].count?x[groups[g].assignments[row]/8*4096+c]:__float2bfloat16_rn(0.f);
}
__device__ DenseWeight weight(const void* image,unsigned part,bool nv,bool down_fp8,size_t stride) {
  auto p=static_cast<const unsigned char*>(image)+part*stride;
  bool packed=nv&&!(part==2&&down_fp8);
  return {p,packed?nullptr:p+4096*2048,packed?3u:1u};
}
template<unsigned M> __device__ void group_gate(B* out,const B* x,const Group& group,bool nv,bool down_fp8,size_t stride,unsigned r) {
  float a[M],v[M];dot_rows(a,x,weight(group.image,0,nv,down_fp8,stride),r,2048,4096,!nv);
  dot_rows(v,x,weight(group.image,1,nv,down_fp8,stride),r,2048,4096,!nv);
  if(threadIdx.x%32==0)for(unsigned t=0;t<M;++t)out[t*2048+r]=__float2bfloat16_rn(swiglu(bf16(a[t]),bf16(v[t])));
}
__global__ void grouped_gate(B* out,const B* x,const Group* groups,const unsigned* count,bool nv,bool down_fp8,size_t stride) {
  unsigned g=blockIdx.y,r=blockIdx.x*8+threadIdx.x/32;
  B* dst=out+g*4*2048;
  if(threadIdx.x%32==0)for(unsigned t=0;t<4;++t)dst[t*2048+r]=__float2bfloat16_rn(0.f);
  if(g>=*count||!groups[g].image)return;
  const B* src=x+g*4*4096;
  switch(groups[g].count){
    case 1:group_gate<1>(dst,src,groups[g],nv,down_fp8,stride,r);break;
    case 2:group_gate<2>(dst,src,groups[g],nv,down_fp8,stride,r);break;
    case 3:group_gate<3>(dst,src,groups[g],nv,down_fp8,stride,r);break;
    case 4:group_gate<4>(dst,src,groups[g],nv,down_fp8,stride,r);break;
  }
}
template<unsigned M> __device__ void group_down(B* out,const B* x,const Group& group,bool nv,bool down_fp8,size_t stride,unsigned r) {
  float values[M];dot_rows(values,x,weight(group.image,2,nv,down_fp8,stride),r,4096,2048,!nv||down_fp8);
  if(threadIdx.x%32==0)for(unsigned t=0;t<M;++t){unsigned a=group.assignments[t];out[(a/8*9+a%8)*4096+r]=__float2bfloat16_rn(values[t]);}
}
__global__ void grouped_down(B* out,const B* x,const Group* groups,const unsigned* count,bool nv,bool down_fp8,size_t stride) {
  unsigned g=blockIdx.y,r=blockIdx.x*8+threadIdx.x/32;if(g>=*count||!groups[g].image)return;
  const B* src=x+g*4*2048;
  switch(groups[g].count){
    case 1:group_down<1>(out,src,groups[g],nv,down_fp8,stride,r);break;
    case 2:group_down<2>(out,src,groups[g],nv,down_fp8,stride,r);break;
    case 3:group_down<3>(out,src,groups[g],nv,down_fp8,stride,r);break;
    case 4:group_down<4>(out,src,groups[g],nv,down_fp8,stride,r);break;
  }
}
}
extern "C" int hayate_glm_async_images(void* images,const void* ids,const void* table,unsigned rows,cudaStream_t s) {
  if(!images||!ids||!table||rows<1||rows>4)return cudaErrorInvalidValue;
  resolve<<<1,32,0,s>>>((void**)images,(const int*)ids,(const void* const*)table,rows);return cudaGetLastError();
}
extern "C" int hayate_glm_async_merge(void* out,const void* host,const void* ids,const void* table,unsigned rows,cudaStream_t s) {
  if(!out||!host||!ids||!table||rows<1||rows>4)return cudaErrorInvalidValue;
  merge<<<rows*8*16,256,0,s>>>((B*)out,(const B*)host,(const int*)ids,(const void* const*)table,rows);return cudaGetLastError();
}
extern "C" int hayate_glm_async_groups(void* groups,void* count,void* gathered,const void* x,const void* ids,const void* table,unsigned rows,cudaStream_t s) {
  if(!groups||!count||!gathered||!x||!ids||!table||rows<1||rows>4)return cudaErrorInvalidValue;
  group_routes<<<1,32,0,s>>>((Group*)groups,(unsigned*)count,(const int*)ids,(const void* const*)table,rows);
  auto e=cudaGetLastError();if(e)return e;
  gather_groups<<<rows*8*4*16,256,0,s>>>((B*)gathered,(const B*)x,(const Group*)groups,(const unsigned*)count);return cudaGetLastError();
}
extern "C" int hayate_glm_async_gate(void* out,const void* x,const void* groups,const void* count,unsigned bound,int nv,int down_fp8,size_t stride,cudaStream_t s) {
  if(!out||!x||!groups||!count||bound<1||bound>32)return cudaErrorInvalidValue;
  grouped_gate<<<dim3(256,bound),256,0,s>>>((B*)out,(const B*)x,(const Group*)groups,(const unsigned*)count,nv,down_fp8,stride);return cudaGetLastError();
}
extern "C" int hayate_glm_async_down(void* out,const void* x,const void* groups,const void* count,unsigned bound,int nv,int down_fp8,size_t stride,cudaStream_t s) {
  if(!out||!x||!groups||!count||bound<1||bound>32)return cudaErrorInvalidValue;
  grouped_down<<<dim3(512,bound),256,0,s>>>((B*)out,(const B*)x,(const Group*)groups,(const unsigned*)count,nv,down_fp8,stride);return cudaGetLastError();
}
namespace {
__global__ void publish(unsigned* out,const unsigned* in,unsigned words) {
  unsigned i=blockIdx.x*256+threadIdx.x;if(i<words)out[i]=in[i];
}
}
extern "C" int hayate_glm_async_publish(void* out,const void* in,unsigned bytes,cudaStream_t s) {
  if(!out||!in||!bytes||bytes%4)return cudaErrorInvalidValue;
  publish<<<(bytes/4+255)/256,256,0,s>>>((unsigned*)out,(const unsigned*)in,bytes/4);return cudaGetLastError();
}
