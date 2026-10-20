#pragma once
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include "math.cuh"
namespace glm53 {
using B=__nv_bfloat16;
// 0: BF16, 1: source block-FP8, 2: HYT per-tensor FP8, 3: HYT NVFP4.
struct DenseWeight { const void* data; const void* scale; unsigned format; };
// Lane l owns columns l*W + i*32*W .. +W (W = 8 for NVFP4, else 4) and sums
// them in that order before the shuffle tree; the batched loads below only
// move memory reads ahead of the arithmetic, so results are bit-identical.
template<unsigned M,unsigned F>
__device__ inline void dot_rows_format(float (&sum)[M],const B* x,DenseWeight w,unsigned row,unsigned n,unsigned k,bool rounded) {
  constexpr unsigned W=F==3?8:4,STEP=32*W,U=8;
  auto image=static_cast<const unsigned char*>(w.data);
  float global=1;
  unsigned cols=0;
  const unsigned char* scales=nullptr;
  if(F==3) {
    cols=(k/16+3)/4*4;scales=image+size_t(n)*k/2;
    global=*reinterpret_cast<const float*>(scales+size_t((n+127)/128*128)*cols);
  } else if(F==2) global=*reinterpret_cast<const float*>(image+size_t(n)*k);
  auto load=[&](unsigned c,unsigned& packed,float& multiplier) {
    if(F==3) {
      unsigned scol=c/16;
      size_t sf=((row/128)*(cols/4)+scol/4)*512+(row%32)*16+(row%128/32)*4+scol%4;
      __nv_fp8_e4m3 scale;scale.__x=scales[sf];
      multiplier=float(scale)*global;
      packed=*reinterpret_cast<const unsigned*>(image+(size_t(row)*k+c)/2);
    } else if(F) {
      multiplier=F==1?static_cast<const float*>(w.scale)[(row/128)*((k+127)/128)+c/128]:global;
      packed=*reinterpret_cast<const unsigned*>(image+size_t(row)*k+c);
    } else {
      multiplier=1;
      packed=0;
    }
  };
  auto accumulate=[&](unsigned c,unsigned packed,float multiplier) {
    float xs[M][W];
    #pragma unroll
    for(unsigned t=0;t<M;++t) {
      if(W==8) {
        uint4 raw=*reinterpret_cast<const uint4*>(x+size_t(t)*k+c);
        auto p=reinterpret_cast<const __nv_bfloat162*>(&raw);
        #pragma unroll
        for(unsigned j=0;j<4;++j){float2 v=__bfloat1622float2(p[j]);xs[t][2*j]=v.x;xs[t][2*j+1]=v.y;}
      } else {
        uint2 raw=*reinterpret_cast<const uint2*>(x+size_t(t)*k+c);
        auto p=reinterpret_cast<const __nv_bfloat162*>(&raw);
        #pragma unroll
        for(unsigned j=0;j<2;++j){float2 v=__bfloat1622float2(p[j]);xs[t][2*j]=v.x;xs[t][2*j+1]=v.y;}
      }
    }
    float ws[W];
    if(F==0) {
      uint2 raw=*reinterpret_cast<const uint2*>(static_cast<const B*>(w.data)+size_t(row)*k+c);
      auto p=reinterpret_cast<const __nv_bfloat162*>(&raw);
      #pragma unroll
      for(unsigned j=0;j<2;++j){float2 v=__bfloat1622float2(p[j]);ws[2*j]=v.x;ws[2*j+1]=v.y;}
    } else {
      #pragma unroll
      for(unsigned j=0;j<W;++j) {
        float value;
        if(F==3) {
          auto pair=__half22float2(static_cast<__half2>(__nv_cvt_fp4x2_to_halfraw2((packed>>(j/2*8))&255,__NV_E2M1)));
          value=(j%2?pair.y:pair.x)*multiplier;
        } else {
          __nv_fp8_e4m3 v;v.__x=(packed>>(j*8))&255;value=float(v)*multiplier;
        }
        ws[j]=rounded?bf16(value):value;
      }
    }
    #pragma unroll
    for(unsigned j=0;j<W;++j)
      #pragma unroll
      for(unsigned t=0;t<M;++t)sum[t]=fmaf(ws[j],xs[t][j],sum[t]);
  };
  #pragma unroll
  for(unsigned t=0;t<M;++t)sum[t]=0;
  unsigned c=threadIdx.x%32*W;
  // Plain row-major payloads only need 4/8-byte alignment; k%16 == 0.
  if(F==0) {
    for(;c<k;c+=STEP)accumulate(c,0,1);
  } else {
    for(;c+(U-1)*STEP<k;c+=U*STEP) {
      unsigned packed[U];float multiplier[U];
      #pragma unroll
      for(unsigned u=0;u<U;++u)load(c+u*STEP,packed[u],multiplier[u]);
      #pragma unroll
      for(unsigned u=0;u<U;++u)accumulate(c+u*STEP,packed[u],multiplier[u]);
    }
    for(;c<k;c+=STEP){unsigned packed;float multiplier;load(c,packed,multiplier);accumulate(c,packed,multiplier);}
  }
  #pragma unroll
  for(unsigned t=0;t<M;++t)for(unsigned s=16;s;s/=2)sum[t]+=__shfl_down_sync(0xffffffff,sum[t],s);
}
template<unsigned M>
__device__ inline void dot_rows(float (&sum)[M],const B* x,DenseWeight w,unsigned row,unsigned n,unsigned k,bool rounded=true) {
  switch(w.format) {
    case 0: dot_rows_format<M,0>(sum,x,w,row,n,k,false); break;
    case 1: dot_rows_format<M,1>(sum,x,w,row,n,k,rounded); break;
    case 2: dot_rows_format<M,2>(sum,x,w,row,n,k,rounded); break;
    default: dot_rows_format<M,3>(sum,x,w,row,n,k,rounded); break;
  }
}
__device__ inline float dot(const B* x,DenseWeight w,unsigned row,unsigned n,unsigned k,bool rounded=true) {
  float sum[1];dot_rows(sum,x,w,row,n,k,rounded);return sum[0];
}
}
