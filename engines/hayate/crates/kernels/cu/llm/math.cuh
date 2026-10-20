#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>
#ifdef __CUDACC__
#define HLLM __host__ __device__
// Call libdevice explicitly: the build's --use_fast_math rewrites ordinary
// powf/sinf/cosf to reduced-accuracy intrinsics, including large RoPE angles.
extern "C" __device__ float __nv_powf(float, float);
extern "C" __device__ float __nv_cosf(float);
extern "C" __device__ float __nv_sinf(float);
#else
#define HLLM
#endif
namespace hayate_llm {
HLLM inline float fp4(unsigned c) {
  constexpr float values[] = {0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f};
  return c & 8 ? -values[c & 7] : values[c & 7];
}
// FP4 conversion avoids a per-thread table in local memory.
HLLM inline float fp4_bits(unsigned c) {
  unsigned magnitude=c&7;
  if(magnitude<2)return (c&8?-1.f:1.f)*magnitude*.5f;
  uint32_t bits=((c&8)<<28)|(((magnitude/2)+126)<<23)|((magnitude&1)<<22);
  float value;
#ifdef __CUDA_ARCH__
  value=__uint_as_float(bits);
#else
  std::memcpy(&value,&bits,4);
#endif
  return value;
}
HLLM inline float e4(unsigned c) {
  unsigned e=(c&127)>>3, m=c&7;
  float x=e ? ldexpf(1.f+m*.125f,int(e)-7) : m/512.f;
  return c&128 ? -x : x;
}
// Exact E4M3 conversion for GEMV, avoiding ldexp per element.
HLLM inline float e4_bits(unsigned c) {
  unsigned e=(c&127)>>3,m=c&7;
  if(e==0)return (c&128?-1.f:1.f)*m/512.f;
  uint32_t bits=((c&128)<<24)|((e+120)<<23)|(m<<20);
  float value;
#ifdef __CUDA_ARCH__
  value=__uint_as_float(bits);
#else
  std::memcpy(&value,&bits,4);
#endif
  return value;
}
HLLM inline float merge_scale(float maximum,float denom,float total_maximum) {
  return denom==0.f?0.f:expf(maximum-total_maximum);
}
HLLM inline size_t scale_offset(unsigned r,unsigned c,unsigned cols) {
  return (size_t(r/128)*(cols/4)+c/4)*512+(r%32)*16+(r%128/32)*4+c%4;
}
HLLM inline float inv_freq(unsigned pair,unsigned dim,bool global) {
  if(global && pair>=dim/8)return 0.f;
  // HF forms an F32 positive power followed by an F32 reciprocal.
#ifdef __CUDA_ARCH__
  return __fdiv_rn(1.f,__nv_powf(global?1e6f:1e4f,2.f*pair/dim));
#else
  return 1.f/powf(global?1e6f:1e4f,2.f*pair/dim);
#endif
}
HLLM inline float rope_cos(float angle) {
#ifdef __CUDA_ARCH__
  return __nv_cosf(angle);
#else
  return cosf(angle);
#endif
}
HLLM inline float rope_sin(float angle) {
#ifdef __CUDA_ARCH__
  return __nv_sinf(angle);
#else
  return sinf(angle);
#endif
}
HLLM inline float gelu(float x) {
  return .5f*x*(1.f+tanhf(.7978845608028654f*(x+.044715f*x*x*x)));
}
HLLM inline float round_bf16(float value) {
#ifdef __CUDA_ARCH__
  return __bfloat162float(__float2bfloat16_rn(value));
#else
  uint32_t bits;
  std::memcpy(&bits,&value,4);
  bits=(bits+0x7fff+((bits>>16)&1))&0xffff0000;
  std::memcpy(&value,&bits,4);
  return value;
#endif
}
HLLM inline float gate_up(float gate,float up) {
  return round_bf16(round_bf16(gelu(round_bf16(gate)))*round_bf16(up));
}

}
#undef HLLM
