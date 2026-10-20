#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include "math.cuh"
namespace {
using B=__nv_bfloat16;
__device__ float warp_sum(float v) {
  for(unsigned s=16;s;s>>=1)v+=__shfl_down_sync(0xffffffff,v,s);
  return v;
}
__device__ void output(void* y,unsigned row,float sum,int logits) {
  if(!logits){static_cast<B*>(y)[row]=__float2bfloat16_rn(sum);return;}
  float z=__bfloat162float(__float2bfloat16_rn(sum));
  z=__bfloat162float(__float2bfloat16_rn(z/30.f));
  z=__bfloat162float(__float2bfloat16_rn(tanhf(z)));
  static_cast<float*>(y)[row]=__bfloat162float(__float2bfloat16_rn(z*30.f));
}
__device__ void finish(void* y,unsigned row,float gate,float up,bool fused,int logits) {
  if(!fused){output(y,row,gate,logits);return;}
  static_cast<B*>(y)[row]=__float2bfloat16_rn(hayate_llm::gate_up(gate,up));
}
// The measured tied BF16 head already reads at 1.4 TB/s. Retain its
// register-only four-row path; staging it is not justified by that profile.
__global__ void lm_head(void* y,const B* x,const B* w,unsigned n,unsigned k,int logits) {
  unsigned lane=threadIdx.x&31,row=blockIdx.x*4+threadIdx.x/32;
  if(row>=n)return;
  float sum=0.f;
  for(unsigned j=lane*8;j<k;j+=256) {
    uint4 packed=*reinterpret_cast<const uint4*>(w+size_t(row)*k+j);
    unsigned words[]={packed.x,packed.y,packed.z,packed.w};
    #pragma unroll
    for(unsigned u=0;u<8;++u) {
      float weight=__uint_as_float(((words[u/2]>>(16*(u%2)))&65535)<<16);
      sum+=__bfloat162float(x[j+u])*weight;
    }
  }
  sum=warp_sum(sum);
  if(lane==0)output(y,row,sum,logits);
}
// Uniform 16-byte copies. Empty tail vectors use cp.async's zero-fill operand.
__device__ void copy16(uint4* dst,const uint8_t* src,bool valid,bool async_load) {
  if(async_load) {
    unsigned shared=static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::
        "r"(shared),"l"(src),"r"(valid?16:0):"memory");
  } else *dst=valid?*reinterpret_cast<const uint4*>(src):make_uint4(0,0,0,0);
}
template<int F> __device__ float dot(const uint4& packed,const B* x,
    const uint8_t* w,unsigned row,unsigned j,unsigned n,unsigned k) {
  constexpr unsigned width=F==1?32:F==2?16:8;
  unsigned words[]={packed.x,packed.y,packed.z,packed.w};
  float s0=1.f,s1=1.f;
  if constexpr(F==1) {
    auto scales=w+size_t(n)*k/2;
    unsigned cols=(k/16+3)/4*4;
    float global=*reinterpret_cast<const float*>(scales+size_t((n+127)/128*128)*cols);
    unsigned sf=*reinterpret_cast<const uint16_t*>(scales+hayate_llm::scale_offset(row,j/16,cols));
    auto pair=__half22float2(static_cast<__half2>(__nv_cvt_fp8x2_to_halfraw2(sf,__NV_E4M3)));
    s0=pair.x*global;s1=pair.y*global;
  } else if constexpr(F==2) s0=*reinterpret_cast<const float*>(w+size_t(n)*k);
  float sum=0.f;
  #pragma unroll
  for(unsigned u=0;u<width;u+=2) {
    float2 weight;
    if constexpr(F==1) weight=__half22float2(static_cast<__half2>(
        __nv_cvt_fp4x2_to_halfraw2((words[u/8]>>(4*(u%8)))&255,__NV_E2M1)));
    else if constexpr(F==2) weight=__half22float2(static_cast<__half2>(
        __nv_cvt_fp8x2_to_halfraw2((words[u/4]>>(8*(u%4)))&65535,__NV_E4M3)));
    else {weight.x=__uint_as_float((words[u/2]&65535)<<16);weight.y=__uint_as_float(words[u/2]&0xffff0000);}
    auto input=__bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(x+u));
    float scale=u<16?s0:s1;
    sum=fmaf(input.x,weight.x*scale,sum);
    sum=fmaf(input.y,weight.y*scale,sum);
  }
  return sum;
}
// Four warps/CTA: either four rows x one warp, or one row x four warps.
// X is staged once for every row and both gate/up weights. Double buffering
// overlaps the next 128-bit global loads with native packed FP4/FP8 conversion.
template<int F,unsigned Rows,bool Fused> __global__ void tiled(void* y,float* partial,
    const B* x,const uint8_t* gate,const uint8_t* up,unsigned n,unsigned k,
    unsigned splits,int logits,bool async_load) {
  constexpr unsigned warps=4/Rows,width=F==1?32:F==2?16:8,tile=warps*32*width;
  __shared__ uint4 inputs[2][tile/8],weights[2][Fused?2:1][128];
  __shared__ float sums[Fused?2:1][4];
  unsigned t=threadIdx.x,local_row=t/(warps*32),row=blockIdx.x*Rows+local_row;
  unsigned column=(t%(warps*32))*width;
  unsigned tiles=(k+tile-1)/tile,chunk=(tiles+splits-1)/splits;
  unsigned begin=blockIdx.y*chunk,end=min(tiles,begin+chunk);
  auto load=[&](unsigned b,unsigned base) {
    for(unsigned v=t;v<tile/8;v+=128)
      copy16(&inputs[b][v],reinterpret_cast<const uint8_t*>(x)+size_t(base+v*8)*2,base+v*8<k,async_load);
    copy16(&weights[b][0][t],gate+(size_t(row)*k+base+column)*(F==1?1:F==2?2:4)/2,
        row<n&&base+column<k,async_load);
    if constexpr(Fused)copy16(&weights[b][1][t],up+(size_t(row)*k+base+column)*(F==1?1:F==2?2:4)/2,
        row<n&&base+column<k,async_load);
    if(async_load)asm volatile("cp.async.commit_group;" ::: "memory");
  };
  float g=0.f,u=0.f;
  if(begin<end)load(0,begin*tile);
  for(unsigned it=begin;it<end;++it) {
    unsigned b=(it-begin)&1;
    if(it+1<end)load(b^1,(it+1)*tile);
    if(async_load) {
      if(it+1<end)asm volatile("cp.async.wait_group 1;" ::: "memory");
      else asm volatile("cp.async.wait_group 0;" ::: "memory");
    }
    __syncthreads();
    unsigned j=it*tile+column;
    if(row<n&&j<k) {
      auto input=reinterpret_cast<const B*>(inputs[b])+column;
      g+=dot<F>(weights[b][0][t],input,gate,row,j,n,k);
      if constexpr(Fused)u+=dot<F>(weights[b][1][t],input,up,row,j,n,k);
    }
    __syncthreads();
  }
  g=warp_sum(g);u=warp_sum(u);
  if((t&31)==0){sums[0][t/32]=g;if constexpr(Fused)sums[1][t/32]=u;}
  __syncthreads();
  if(row<n&&t%(warps*32)==0) {
    g=u=0.f;
    #pragma unroll
    for(unsigned w=0;w<warps;++w){g+=sums[0][local_row*warps+w];if constexpr(Fused)u+=sums[1][local_row*warps+w];}
    if(splits==1)finish(y,row,g,u,Fused,logits);
    else {
      partial[size_t(blockIdx.y)*n+row]=g;
      if constexpr(Fused)partial[size_t(splits)*n+size_t(blockIdx.y)*n+row]=u;
    }
  }
}
template<bool Fused> __global__ void merge(void* y,const float* partial,unsigned n,unsigned splits,int logits) {
  unsigned row=blockIdx.x*256+threadIdx.x;
  if(row>=n)return;
  float g=0.f,u=0.f;
  for(unsigned s=0;s<splits;++s){g+=partial[size_t(s)*n+row];if constexpr(Fused)u+=partial[size_t(splits)*n+size_t(s)*n+row];}
  finish(y,row,g,u,Fused,logits);
}
template<int F,bool Fused> int launch(void* y,const void* x,const void* gate,const void* up,
    unsigned n,unsigned k,int logits,float* partial,unsigned rows,unsigned splits,int async_load,cudaStream_t s) {
  auto gx=static_cast<const B*>(x);auto g=static_cast<const uint8_t*>(gate),u=static_cast<const uint8_t*>(up);
  dim3 grid((n+rows-1)/rows,splits);
  if(rows==1)tiled<F,1,Fused><<<grid,128,0,s>>>(y,partial,gx,g,u,n,k,splits,logits,async_load);
  else tiled<F,4,Fused><<<grid,128,0,s>>>(y,partial,gx,g,u,n,k,splits,logits,async_load);
  auto error=cudaGetLastError();
  if(error!=cudaSuccess||splits==1)return error;
  merge<Fused><<<(n+255)/256,256,0,s>>>(y,partial,n,splits,logits);
  return cudaGetLastError();
}
template<bool Fused> int dispatch(void* y,const void* x,const void* g,const void* u,unsigned n,unsigned k,
    int format,int logits,float* partial,unsigned rows,unsigned splits,int async_load,cudaStream_t s) {
  if(!n||!k||k%32||(rows!=1&&rows!=4)||(splits!=1&&splits!=2&&splits!=4&&splits!=8)||
      (splits>1&&!partial)||(async_load!=0&&async_load!=1))return cudaErrorInvalidValue;
  if constexpr(!Fused)if(format==0&&logits) {
    lm_head<<<(n+3)/4,128,0,s>>>(y,static_cast<const B*>(x),static_cast<const B*>(g),n,k,logits);
    return cudaGetLastError();
  }
  if(format==1)return launch<1,Fused>(y,x,g,u,n,k,logits,partial,rows,splits,async_load,s);
  if(format==2)return launch<2,Fused>(y,x,g,u,n,k,logits,partial,rows,splits,async_load,s);
  if(format==0)return launch<0,Fused>(y,x,g,u,n,k,logits,partial,rows,splits,async_load,s);
  return cudaErrorInvalidValue;
}
}
extern "C" int hayate_llm_gemv(void* y,const void* x,const void* w,unsigned n,unsigned k,int format,
    int logits,float* partial,unsigned rows,unsigned splits,int async_load,cudaStream_t s) {
  return dispatch<false>(y,x,w,nullptr,n,k,format,logits,partial,rows,splits,async_load,s);
}
extern "C" int hayate_llm_gate_up(void* y,const void* x,const void* g,const void* u,unsigned n,unsigned k,int format,
    float* partial,unsigned rows,unsigned splits,int async_load,cudaStream_t s) {
  return dispatch<true>(y,x,g,u,n,k,format,0,partial,rows,splits,async_load,s);
}
