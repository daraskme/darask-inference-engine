#include <cuda_runtime.h>
#include "dense.cuh"
#include "lowbit.cuh"
namespace {
using namespace glm53;
using namespace glm53::lowbit;
template<unsigned M,unsigned GRID,unsigned GROUP>
__device__ __forceinline__ void low_dot(float (&sum)[M],const B* x,const unsigned char* image,unsigned row,unsigned n,unsigned k) {
  constexpr unsigned U=8,STEP=256;
  unsigned lane=threadIdx.x%32;
  auto scales=image+scale_start(n,k,GRID);
  float tensor=*reinterpret_cast<const float*>(scales+size_t(n)*k/GROUP);
  #pragma unroll
  for(unsigned t=0;t<M;++t)sum[t]=0;
  // Same columns/FMA/shuffle order as NVFP4. Repeated addresses in a warp
  // coalesce: 256 weights need 64 low-plane bytes and (for Q3) 32 high bytes.
  for(unsigned base=0;base<k;base+=STEP*U) {
    uint2 lo[U],hi[U]={};float multiplier[U];
    #pragma unroll
    for(unsigned u=0;u<U;++u) {
      unsigned c=base+u*STEP+lane*8;
      if(c<k) {
        lo[u]=*reinterpret_cast<const uint2*>(image+size_t(row)*k/4+c/128*32+c%32);
        if constexpr(bits(GRID)==3)hi[u]=*reinterpret_cast<const uint2*>(image+size_t(n)*k/4+size_t(row)*k/8+c/256*32+c%32);
        __nv_fp8_e4m3 scale;scale.__x=scales[size_t(row)*(k/GROUP)+c/GROUP];
        multiplier[u]=float(scale)*tensor;
      }
    }
    #pragma unroll
    for(unsigned u=0;u<U;++u) {
      unsigned c=base+u*STEP+lane*8;
      if(c<k) {
        unsigned packed=packed8(lo[u].x,lo[u].y,hi[u].x,hi[u].y,c%128/32,GRID,c%256/128*4);
        float ws[8];
        #pragma unroll
        for(unsigned j=0;j<4;++j) {
          unsigned q=(packed>>(j*8))&255;
          auto pair=__half22float2(static_cast<__half2>(__nv_cvt_fp4x2_to_halfraw2(q,__NV_E2M1)));
          ws[2*j]=pair.x*multiplier[u];ws[2*j+1]=pair.y*multiplier[u];
        }
        #pragma unroll
        for(unsigned t=0;t<M;++t) {
          uint4 raw=*reinterpret_cast<const uint4*>(x+size_t(t)*k+c);
          auto p=reinterpret_cast<const __nv_bfloat162*>(&raw);
          #pragma unroll
          for(unsigned j=0;j<4;++j) {
            float2 v=__bfloat1622float2(p[j]);
            sum[t]=fmaf(ws[2*j],v.x,sum[t]);sum[t]=fmaf(ws[2*j+1],v.y,sum[t]);
          }
        }
      }
    }
  }
  #pragma unroll
  for(unsigned t=0;t<M;++t)for(unsigned s=16;s;s/=2)sum[t]+=__shfl_down_sync(0xffffffff,sum[t],s);
}

template<unsigned M>
__device__ __forceinline__ void mixed_dot(float (&sum)[M],const B* x,const unsigned char* image,MatrixFormat f,unsigned row,unsigned n,unsigned k) {
  if(!f.format) {dot_rows_format<M,3>(sum,x,{image,nullptr,3},row,n,k,false);return;}
  #define GRID(G) case G: if(f.group==16)low_dot<M,G,16>(sum,x,image,row,n,k);else low_dot<M,G,32>(sum,x,image,row,n,k);break
  switch(f.grid){GRID(0);GRID(1);GRID(2);GRID(3);GRID(4);}
  #undef GRID
}
template<unsigned M>
__global__ void gemv(B* out,const B* x,const unsigned char* image,MatrixFormat f,unsigned n,unsigned k) {
  unsigned row=blockIdx.x*8+threadIdx.x/32;if(row>=n)return;
  float sums[M];mixed_dot(sums,x,image,f,row,n,k);
  if(threadIdx.x%32==0)for(unsigned t=0;t<M;++t)out[size_t(t)*n+row]=__float2bfloat16_rn(sums[t]);
}
template<unsigned M,bool DOWN>
__global__ void moe(B* out,const B* x,const B* original,const void* const* images,const ExpertFormat* formats,
    DenseWeight gate,DenseWeight up,unsigned hidden,unsigned intermediate) {
  unsigned row=blockIdx.x*8+threadIdx.x/32,g=blockIdx.y,n=DOWN?hidden:intermediate,k=DOWN?intermediate:hidden;
  if(row>=n)return;
  if(g<8&&!images[g]) {
    if(!DOWN&&threadIdx.x%32==0)for(unsigned t=0;t<M;++t)out[(size_t(g)*M+t)*n+row]=__float2bfloat16_rn(0.f);
    return;
  }
  float a[M],b[M];
  if(g==8) {
    dot_rows(a,DOWN?x+size_t(g)*M*k:original,gate,row,n,k,true);
    if(!DOWN)dot_rows(b,original,up,row,n,k,true);
  } else {
    ExpertFormat f=formats[g];auto image=static_cast<const unsigned char*>(images[g]);
    size_t step=stride(intermediate,hidden,f.gate_up);
    mixed_dot(a,DOWN?x+size_t(g)*M*k:x,image+(DOWN?2*step:0),DOWN?f.down:f.gate_up,row,n,k);
    if(!DOWN)mixed_dot(b,x,image+step,f.gate_up,row,n,k);
  }
  if(threadIdx.x%32==0)for(unsigned t=0;t<M;++t)
    out[(size_t(g)*M+t)*n+row]=__float2bfloat16_rn(DOWN?a[t]:swiglu(bf16(a[t]),bf16(b[t])));
}
__global__ void convert(unsigned char* out,const unsigned char* image,unsigned n,unsigned k,unsigned grid,unsigned group,size_t bytes) {
  size_t i=size_t(blockIdx.x)*256+threadIdx.x;
  if(i<bytes)out[i]=convert_byte(image,i,n,k,grid,group);
}
__global__ void convert_expert(unsigned char* out,const unsigned char* image,ExpertFormat formats,unsigned hidden,unsigned intermediate) {
  unsigned part=blockIdx.y;
  size_t step=stride(intermediate,hidden,{0,0,0,0});
  size_t i=size_t(blockIdx.x)*256+threadIdx.x;if(i>=step)return;
  out[part*step+i]=convert_expert_byte(image,i,part,formats,hidden,intermediate);
}
bool valid(MatrixFormat f) {
  return f.reserved==0&&((f.format==0&&f.grid==0&&f.group==0)
    ||(f.grid<=4&&(f.group==16||f.group==32)&&f.format==(bits(f.grid)==3?1:2)));
}
}
extern "C" int hayate_glm_lowbit_convert(void* out,const void* image,unsigned n,unsigned k,unsigned grid,unsigned group,cudaStream_t s) {
  if(!out||!image||out==image||!n||!k||k%256||grid>4||(group!=16&&group!=32))return cudaErrorInvalidValue;
  size_t bytes=storage(n,k,{0,0,0,0});
  convert<<<(bytes+255)/256,256,0,s>>>((unsigned char*)out,(const unsigned char*)image,n,k,grid,group,bytes);
  return cudaGetLastError();
}
extern "C" int hayate_glm_lowbit_convert_expert(void* out,const void* image,ExpertFormat f,unsigned hidden,unsigned intermediate,cudaStream_t s) {
  if(!out||!image||out==image||!valid(f.gate_up)||!valid(f.down)||!hidden||hidden%256||!intermediate||intermediate%256)return cudaErrorInvalidValue;
  size_t step=stride(intermediate,hidden,{0,0,0,0});
  convert_expert<<<dim3(step/256,3),256,0,s>>>((unsigned char*)out,(const unsigned char*)image,f,hidden,intermediate);
  return cudaGetLastError();
}
extern "C" int hayate_glm_lowbit_gemv(void* out,const void* x,const void* image,MatrixFormat f,unsigned m,unsigned n,unsigned k,cudaStream_t s) {
  if(!out||!x||!image||!valid(f)||m<1||m>4||!n||!k||k%256)return cudaErrorInvalidValue;
  #define RUN(M) gemv<M><<<(n+7)/8,256,0,s>>>((B*)out,(const B*)x,(const unsigned char*)image,f,n,k)
  switch(m){case 1:RUN(1);break;case 2:RUN(2);break;case 3:RUN(3);break;case 4:RUN(4);break;}
  #undef RUN
  return cudaGetLastError();
}
extern "C" int hayate_glm_lowbit_gate_up(void* out,const void* x,const void* original,const void* images,const ExpertFormat* formats,
    DenseWeight gate,DenseWeight up,unsigned m,unsigned hidden,unsigned intermediate,cudaStream_t s) {
  if(!out||!x||!original||!images||!formats||!gate.data||!up.data||gate.format>3||up.format>3
      ||(gate.format==1&&!gate.scale)||(up.format==1&&!up.scale)||m<1||m>4||!hidden||hidden%256||!intermediate||intermediate%256)return cudaErrorInvalidValue;
  #define RUN(M) moe<M,false><<<dim3((intermediate+7)/8,9),256,0,s>>>((B*)out,(const B*)x,(const B*)original,(const void* const*)images,formats,gate,up,hidden,intermediate)
  switch(m){case 1:RUN(1);break;case 2:RUN(2);break;case 3:RUN(3);break;case 4:RUN(4);break;}
  #undef RUN
  return cudaGetLastError();
}
extern "C" int hayate_glm_lowbit_down(void* out,const void* x,const void* images,const ExpertFormat* formats,
    DenseWeight shared,unsigned m,unsigned hidden,unsigned intermediate,cudaStream_t s) {
  if(!out||!x||!images||!formats||!shared.data||shared.format>3||(shared.format==1&&!shared.scale)
      ||m<1||m>4||!hidden||hidden%256||!intermediate||intermediate%256)return cudaErrorInvalidValue;
  #define RUN(M) moe<M,true><<<dim3((hidden+7)/8,9),256,0,s>>>((B*)out,(const B*)x,nullptr,(const void* const*)images,formats,shared,{nullptr,nullptr,0},hidden,intermediate)
  switch(m){case 1:RUN(1);break;case 2:RUN(2);break;case 3:RUN(3);break;case 4:RUN(4);break;}
  #undef RUN
  return cudaGetLastError();
}
