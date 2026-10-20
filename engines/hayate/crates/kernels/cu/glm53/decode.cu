#include <cuda_runtime.h>
#include "dense.cuh"
namespace {
using namespace glm53;
__global__ void dense(B* out,const B* x,DenseWeight w,unsigned n,unsigned k) {
  unsigned row=blockIdx.x*8+threadIdx.x/32;if(row>=n)return;
  float value=dot(x,w,row,n,k);
  if(threadIdx.x%32==0)out[row]=__float2bfloat16_rn(value);
}
__global__ void unpack(B* out,DenseWeight w,unsigned n,unsigned k) {
  size_t i=size_t(blockIdx.x)*256+threadIdx.x;if(i>=size_t(n)*k)return;
  auto image=static_cast<const unsigned char*>(w.data);
  float value=w.format==3?nvfp4(image,i/k,i%k,n,k)
      :fp8(image[i])*(w.format==1?static_cast<const float*>(w.scale)[(i/k/128)*((k+127)/128)+i%k/128]
      :*reinterpret_cast<const float*>(image+size_t(n)*k));
  out[i]=__float2bfloat16_rn(value);
}
__global__ void router_project(float* out,const B* x,const B* weight) {
  unsigned row=blockIdx.x*8+threadIdx.x/32;
  float value=dot(x,{weight,nullptr,0},row,288,4096);
  if(threadIdx.x%32==0)out[row]=value;
}
__global__ void router_select(float* weights,int* ids,const float* logits,const float* bias) {
  __shared__ float scores[288],choices[288];
  for(unsigned e=threadIdx.x;e<288;e+=256){scores[e]=sigmoid(logits[e]);choices[e]=scores[e]+bias[e];}
  __syncthreads();if(threadIdx.x)return;
  float sum=0;
  for(unsigned j=0;j<8;++j){unsigned best=0;for(unsigned e=1;e<288;++e)if(choices[e]>choices[best])best=e;
    ids[j]=best;weights[j]=scores[best];sum+=scores[best];choices[best]=-INFINITY;}
  for(unsigned j=0;j<8;++j)weights[j]=weights[j]/(sum+1.e-20f)*2.5f;
}
__device__ DenseWeight routed(const void* image,unsigned part,bool nv,bool down_fp8,size_t stride) {
  auto w=static_cast<const unsigned char*>(image)+part*stride;
  bool packed=nv&&!(part==2&&down_fp8);
  return {w,packed?nullptr:w+size_t(4096)*2048,packed?3u:1u};
}
__global__ void gate_up(B* out,const B* x,const B* original,const void* const* images,
    DenseWeight gate,DenseWeight up,bool nv,bool down_fp8,size_t stride) {
  unsigned row=blockIdx.x*8+threadIdx.x/32,g=blockIdx.y;
  if(g<8&&!images[g]){
    if(threadIdx.x%32==0)out[g*2048+row]=__float2bfloat16_rn(0.f);
    return;
  }
  float a=dot(g==8?original:x,g==8?gate:routed(images[g],0,nv,down_fp8,stride),row,2048,4096,g==8||!nv);
  float v=dot(g==8?original:x,g==8?up:routed(images[g],1,nv,down_fp8,stride),row,2048,4096,g==8||!nv);
  if(threadIdx.x%32==0)out[g*2048+row]=__float2bfloat16_rn(swiglu(bf16(a),bf16(v)));
}
__global__ void down(B* out,const B* x,const void* const* images,DenseWeight shared,
    bool nv,bool down_fp8,size_t stride) {
  unsigned row=blockIdx.x*8+threadIdx.x/32,g=blockIdx.y;
  if(g<8&&!images[g])return;
  float value=dot(x+g*2048,g==8?shared:routed(images[g],2,nv,down_fp8,stride),row,4096,2048,g==8||!nv||down_fp8);
  if(threadIdx.x%32==0)out[g*4096+row]=__float2bfloat16_rn(value);
}
__global__ void sum(B* out,const B* x,const int* ids,const float* weights) {
  unsigned c=blockIdx.x*256+threadIdx.x;int order[8];route_order(order,ids);
  float value=0;for(unsigned g=0;g<8;++g){unsigned a=order[g];value=expert_add(value,__bfloat162float(x[a*4096+c]),weights[a]);}
  out[c]=__float2bfloat16_rn(value+__bfloat162float(x[8*4096+c]));
}
}
extern "C" int hayate_glm_dense(void* out,const void* x,glm53::DenseWeight w,unsigned n,unsigned k,cudaStream_t s) {
  if(!out||!x||!w.data||!n||!k||k%16||w.format>3||(w.format==1&&!w.scale))return cudaErrorInvalidValue;
  dense<<<(n+7)/8,256,0,s>>>((B*)out,(const B*)x,w,n,k);return cudaGetLastError();
}
extern "C" int hayate_glm_dense_unpack(void* out,glm53::DenseWeight w,unsigned n,unsigned k,cudaStream_t s) {
  if(!out||!w.data||!n||!k||k%16||!w.format||w.format>3||(w.format==1&&!w.scale))return cudaErrorInvalidValue;
  unpack<<<(size_t(n)*k+255)/256,256,0,s>>>((B*)out,w,n,k);return cudaGetLastError();
}
extern "C" int hayate_glm_route_decode(void* logits,void* weights,void* ids,const void* x,const void* w,const void* bias,cudaStream_t s) {
  if(!logits||!weights||!ids||!x||!w||!bias)return cudaErrorInvalidValue;
  router_project<<<36,256,0,s>>>((float*)logits,(const B*)x,(const B*)w);
  auto e=cudaGetLastError();if(e)return e;
  router_select<<<1,256,0,s>>>((float*)weights,(int*)ids,(const float*)logits,(const float*)bias);return cudaGetLastError();
}
extern "C" int hayate_glm_moe_gate_up(void* out,const void* x,const void* original,const void* images,
    glm53::DenseWeight gate,glm53::DenseWeight up,int nv,int down_fp8,size_t stride,cudaStream_t s) {
  if(!out||!x||!original||!images||!gate.data||!up.data||!stride)return cudaErrorInvalidValue;
  gate_up<<<dim3(256,9),256,0,s>>>((B*)out,(const B*)x,(const B*)original,(const void* const*)images,gate,up,nv,down_fp8,stride);return cudaGetLastError();
}
extern "C" int hayate_glm_moe_down(void* out,const void* x,const void* images,glm53::DenseWeight shared,
    int nv,int down_fp8,size_t stride,cudaStream_t s) {
  if(!out||!x||!images||!shared.data||!stride)return cudaErrorInvalidValue;
  down<<<dim3(512,9),256,0,s>>>((B*)out,(const B*)x,(const void* const*)images,shared,nv,down_fp8,stride);return cudaGetLastError();
}
extern "C" int hayate_glm_moe_reduce(void* out,const void* x,const void* ids,const void* weights,cudaStream_t s) {
  if(!out||!x||!ids||!weights)return cudaErrorInvalidValue;
  sum<<<16,256,0,s>>>((B*)out,(const B*)x,(const int*)ids,(const float*)weights);return cudaGetLastError();
}

extern "C" int hayate_glm_latent_copy(void* host,const void* latent,cudaStream_t s) {
  if(!host||!latent)return cudaErrorInvalidValue;
  return cudaMemcpyAsync(host,latent,512*2,cudaMemcpyDeviceToHost,s);
}

namespace {
template<unsigned M> __global__ void batch_dense(B* out,const B* x,DenseWeight w,unsigned n,unsigned k,bool rounded) {
  unsigned row=blockIdx.x*8+threadIdx.x/32;if(row>=n)return;
  float sums[M];dot_rows(sums,x,w,row,n,k,rounded);
  if(threadIdx.x%32==0)for(unsigned t=0;t<M;++t)out[t*n+row]=__float2bfloat16_rn(sums[t]);
}
template<unsigned M> __global__ void batch_router(float* out,const B* x,const B* w) {
  unsigned row=blockIdx.x*8+threadIdx.x/32;
  float sums[M];dot_rows(sums,x,{w,nullptr,0},row,288,4096);
  if(threadIdx.x%32==0)for(unsigned t=0;t<M;++t)out[t*288+row]=sums[t];
}
__global__ void mtp_add(B* out,const B* x,const B* branch) {
  unsigned c=blockIdx.x*256+threadIdx.x;if(c>=4096)return;
  B value=__float2bfloat16_rn(__bfloat162float(x[c])+__bfloat162float(branch[c]));
  for(unsigned h=0;h<4;++h)out[h*4096+c]=value;
}
__global__ void accept_prefix(unsigned* out,const unsigned* drafts,const unsigned* predictions,unsigned n,unsigned cap) {
  if(threadIdx.x)return;
  unsigned a=0;while(a<n&&a<cap&&drafts[a]==predictions[a])++a;
  out[0]=a;for(unsigned i=0;i<a;++i)out[i+1]=drafts[i];out[a+1]=predictions[a];
}
__global__ void split_selected(B* k,B* v,const B* packed,const unsigned* indices,unsigned count) {
  size_t i=size_t(blockIdx.x)*256+threadIdx.x;if(i>=size_t(count)*16384)return;
  unsigned token=i/16384,h=i/256%64,c=i%256;
  size_t src=size_t(indices[token])*32768+h*512+c;
  k[i]=packed[src];v[i]=packed[src+256];
}
__global__ void spec_ids(unsigned* next,unsigned* drafts,unsigned anchor,unsigned d0,unsigned d1,unsigned d2,bool forced) {
  if(threadIdx.x)return;
  *next=anchor;if(forced){drafts[0]=d0;drafts[1]=d1;drafts[2]=d2;}
}
__global__ void restore_prefix(unsigned char* out,const unsigned char* snapshots,size_t bytes,const unsigned* result,unsigned add) {
  size_t i=size_t(blockIdx.x)*256+threadIdx.x;if(i<bytes)out[i]=snapshots[(result[0]+add)*bytes+i];
}
__global__ void copy_if(unsigned char* out,const unsigned char* x,size_t bytes,const unsigned* result,unsigned row) {
  size_t i=size_t(blockIdx.x)*256+threadIdx.x;if(i<bytes&&row<=result[0])out[i]=x[i];
}
}
extern "C" int hayate_glm_restore_prefix(void* out,const void* snapshots,size_t bytes,const void* result,unsigned add,cudaStream_t s) {
  if(!out||!snapshots||!bytes||!result||add>1)return cudaErrorInvalidValue;
  restore_prefix<<<(bytes+255)/256,256,0,s>>>((unsigned char*)out,(const unsigned char*)snapshots,bytes,(const unsigned*)result,add);return cudaGetLastError();
}
extern "C" int hayate_glm_copy_if(void* out,const void* x,size_t bytes,const void* result,unsigned row,cudaStream_t s) {
  if(!out||!x||!bytes||!result||row>3)return cudaErrorInvalidValue;
  copy_if<<<(bytes+255)/256,256,0,s>>>((unsigned char*)out,(const unsigned char*)x,bytes,(const unsigned*)result,row);return cudaGetLastError();
}
extern "C" int hayate_glm_spec_ids(void* next,void* drafts,unsigned anchor,unsigned d0,unsigned d1,unsigned d2,int forced,cudaStream_t s) {
  if(!next||(forced&&!drafts))return cudaErrorInvalidValue;
  spec_ids<<<1,32,0,s>>>((unsigned*)next,(unsigned*)drafts,anchor,d0,d1,d2,forced);return cudaGetLastError();
}
extern "C" int hayate_glm_split_selected(void* k,void* v,const void* packed,const void* indices,unsigned count,cudaStream_t s) {
  if(!k||!v||!packed||!indices||!count||count>2051)return cudaErrorInvalidValue;
  split_selected<<<(size_t(count)*16384+255)/256,256,0,s>>>((B*)k,(B*)v,(const B*)packed,(const unsigned*)indices,count);return cudaGetLastError();
}
extern "C" int hayate_glm_dense_batch(void* out,const void* x,glm53::DenseWeight w,unsigned m,unsigned n,unsigned k,int rounded,cudaStream_t s) {
  if(!out||!x||!w.data||m<1||m>4||!n||!k||k%16||w.format>3||(w.format==1&&!w.scale))return cudaErrorInvalidValue;
  #define RUN(M) batch_dense<M><<<(n+7)/8,256,0,s>>>((B*)out,(const B*)x,w,n,k,rounded)
  switch(m){case 1:RUN(1);break;case 2:RUN(2);break;case 3:RUN(3);break;case 4:RUN(4);break;}
  #undef RUN
  return cudaGetLastError();
}
extern "C" int hayate_glm_route_batch(void* logits,void* weights,void* ids,const void* x,const void* w,const void* bias,unsigned m,cudaStream_t s) {
  if(!logits||!weights||!ids||!x||!w||!bias||m<1||m>4)return cudaErrorInvalidValue;
  #define RUN(M) batch_router<M><<<36,256,0,s>>>((float*)logits,(const B*)x,(const B*)w)
  switch(m){case 1:RUN(1);break;case 2:RUN(2);break;case 3:RUN(3);break;case 4:RUN(4);break;}
  #undef RUN
  auto e=cudaGetLastError();if(e)return e;
  for(unsigned t=0;t<m;++t)router_select<<<1,256,0,s>>>((float*)weights+t*8,(int*)ids+t*8,(const float*)logits+t*288,(const float*)bias);
  return cudaGetLastError();
}
extern "C" int hayate_glm_mtp_add(void* out,const void* x,const void* branch,cudaStream_t s) {
  if(!out||!x||!branch)return cudaErrorInvalidValue;
  mtp_add<<<16,256,0,s>>>((B*)out,(const B*)x,(const B*)branch);return cudaGetLastError();
}
extern "C" int hayate_glm_accept(void* out,const void* drafts,const void* predictions,unsigned n,unsigned cap,cudaStream_t s) {
  if(!out||!drafts||!predictions||n<1||n>3||cap>n)return cudaErrorInvalidValue;
  accept_prefix<<<1,32,0,s>>>((unsigned*)out,(const unsigned*)drafts,(const unsigned*)predictions,n,cap);return cudaGetLastError();
}
namespace {
__global__ void embed_id(B* out,const B* table,const unsigned* id,unsigned vocab) {
  unsigned c=blockIdx.x*256+threadIdx.x;if(c>=4096)return;
  B value=__float2bfloat16_rn(0.f);if(*id<vocab)value=table[size_t(*id)*4096+c];
  for(unsigned h=0;h<4;++h)out[h*4096+c]=value;
}
}
extern "C" int hayate_glm_embed_id(void* out,const void* table,const void* id,unsigned vocab,cudaStream_t s) {
  if(!out||!table||!id||!vocab)return cudaErrorInvalidValue;
  embed_id<<<16,256,0,s>>>((B*)out,(const B*)table,(const unsigned*)id,vocab);return cudaGetLastError();
}
