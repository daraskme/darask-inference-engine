#pragma once
// EXL3 (exllamav3 trellis) decoding on the device: crates/models/src/glm53/EXL3.md.
//
// The codebooks, the trellis window arithmetic and decoding a tile straight into mma.m16n8k16 fragments follow
// exllamav3 (MIT, turboderp: exllamav3_ext/quant/codebook.cuh, exl3_dq.cuh) as Project Maya PR #265 ports them
// (MIT, src/kernels/cuda/exl3.cu), including exllamav3 1.6's half-integer K.  A tile is 4·K2 little-endian u32
// words (K2 = twice the bits per weight); step p's 16-bit state is the window ending at bit floor((p+1)·K2/2) of the
// tile's MSB-first word ring; step p sits at tile row (p/8 % 4)·2 + (p & 1) + 8·(p/2 & 1) (an input) and column
// p/32 + 8·(p/4 & 1) (an output).  A lane decodes steps lane·8 .. lane·8 + 7: f[i] = steps 2i, 2i+1 as FP16 pairs.
#include <cuda_fp16.h>
#include <cstdint>
namespace glm53::exl3 {
constexpr float HAD=0.088388347648f;   // 1 / sqrt(128)

__device__ __forceinline__ uint32_t bytesum_plus(uint32_t x,uint32_t acc) { return __dp4a(x,0x01010101u,acc); }
// two states -> two values in FP16 arithmetic (exllamav3's decode_3inst_2): cb 0 3inst, 1 mcg, 2 mul1
template<int CB>
__device__ __forceinline__ __half2 decode2h(uint32_t s0,uint32_t s1) {
  if constexpr(CB==2) {
    const uint32_t a=bytesum_plus(s0*0x83DCD12Du,0x6400u),b=bytesum_plus(s1*0x83DCD12Du,0x6400u);
    const __half2 h=__halves2half2(__ushort_as_half((unsigned short)a),__ushort_as_half((unsigned short)b));
    return __hfma2(h,__half2half2(__ushort_as_half(0x1eee)),__half2half2(__ushort_as_half(0xc931)));
  } else {
    uint32_t x0=CB==1?s0*0xCBAC1FEDu:s0*89226354u+64248484u,x1=CB==1?s1*0xCBAC1FEDu:s1*89226354u+64248484u;
    x0=(x0&0x8fff8fffu)^0x3b603b60u;x1=(x1&0x8fff8fffu)^0x3b603b60u;
    const __half2 lo=__halves2half2(__ushort_as_half((unsigned short)(x0&0xffffu)),__ushort_as_half((unsigned short)(x1&0xffffu)));
    const __half2 hi=__halves2half2(__ushort_as_half((unsigned short)(x0>>16)),__ushort_as_half((unsigned short)(x1>>16)));
    return __hadd2(lo,hi);
  }
}
__device__ __forceinline__ uint32_t h2u(__half2 h) { return *reinterpret_cast<uint32_t*>(&h); }
// (hi : lo) >> s, the low 32 bits
__device__ __forceinline__ uint32_t fsh(uint32_t lo,uint32_t hi,int s) {
  return (uint32_t)((((uint64_t)hi<<32)|(uint64_t)lo)>>s);
}
template<int CB>
__device__ __forceinline__ void decode_states(const uint32_t (&s)[8],uint32_t (&f)[4]) {
  #pragma unroll
  for(int i=0;i<4;++i)f[i]=h2u(decode2h<CB>(s[2*i]&0xffffu,s[2*i+1]&0xffffu));
}
// K = 4: the lane's steps end in its own word b (word lane), the first three reach into a (word lane - 1)
__device__ __forceinline__ void states4(uint32_t a,uint32_t b,uint32_t (&s)[8]) {
  const uint32_t x=fsh(b,a,20);
  s[0]=x>>8;s[1]=x>>4;s[2]=x;s[3]=b>>16;s[4]=b>>12;s[5]=b>>8;s[6]=b>>4;s[7]=b;
}
// steps p .. p + G - 1's windows (G = 4 or 2) from the two words that hold them
template<int K,int G>
__device__ __forceinline__ void states_group(const uint32_t* w,int p,uint32_t* s) {
  constexpr int NW=8*K;
  const int b0=(p+257)*K-16,b2=b0+(G-1)*K+16;
  const int i0=b0>>5,i2=(b2-1)>>5,s2=((i2+1)<<5)-b2;
  const uint32_t a=w[i0%NW],b=w[i2%NW];
  #pragma unroll
  for(int j=0;j<G;++j)s[G-1-j]=fsh(b,a,s2+j*K);
}
template<int K,int CB>
__device__ __forceinline__ void frag8_int(const uint32_t* w,int lane,uint32_t (&f)[4]) {
  const int p=lane*8;
  uint32_t s[8];
  if constexpr(K==4) {
    states4(w[(lane+31)&31],w[lane],s);
  } else if constexpr(K==1||K==2) {
    constexpr int NW=8*K,per=32/(8*K);
    const int i1=lane/per,i0=(i1+NW-1)%NW;
    const uint32_t b=fsh(w[i1],w[i0],(per-1-lane%per)*8*K);
    #pragma unroll
    for(int j=0;j<8;++j)s[j]=b>>((7-j)*K);
  } else if constexpr(K==3) {
    constexpr int NW=24;
    const int b1=(p+257)*3,b0=b1-16,b2=b1+21;
    const int i0=b0>>5,i2=(b2-1)>>5,s2=((i2+1)<<5)-b2;
    const uint32_t a=w[i0%NW],b=w[i2%NW];
    const uint32_t hi=fsh(b,a,s2),lo=fsh(b,a,s2+12);
    #pragma unroll
    for(int j=0;j<4;++j){s[7-j]=hi>>(3*j);s[3-j]=lo>>(3*j);}
  } else if constexpr(K==7) {
    #pragma unroll
    for(int g=0;g<4;++g)states_group<K,2>(w,p+2*g,s+2*g);
  } else {   // 5, 6, 8
    states_group<K,4>(w,p,s);
    states_group<K,4>(w,p+4,s+4);
  }
  decode_states<CB>(s,f);
}
// a half-integer K (K2 = 2 KA + 1, mul1): exllamav3's dq8_half.  The lane's steps start even, so their windows end at
// S(p + 7) = (p / 2 + 4) K2 and then KA + 1, KA, KA + 1, ... bits earlier; four windows span 18 + 3 KA <= 27 bits, one
// funnel shift of two words each
template<int K2,int CB>
__device__ __forceinline__ void frag8_half(const uint32_t* w,int lane,uint32_t (&f)[4]) {
  constexpr int KA=K2/2,NW=4*K2,span=18+3*KA;
  const int e7=(lane*4+4)*K2+128*K2;   // one ring on: no negative bit index
  uint32_t s[8];
  #pragma unroll
  for(int g=0;g<2;++g) {
    const int e=e7-(1-g)*2*K2;
    const int hi=(e-1)>>5,lo=(e-span)>>5,sh=((hi+1)<<5)-e;
    const uint32_t x=fsh(w[hi%NW],w[lo%NW],sh);   // the window ending at e in the low bits
    s[4*g+3]=x;s[4*g+2]=x>>(KA+1);s[4*g+1]=x>>(2*KA+1);s[4*g+0]=x>>(3*KA+2);
  }
  decode_states<CB>(s,f);
}
// K2 = 2 K: an even K2 is the integer K2 / 2, an odd one a half-integer K (mul1 only)
template<int K2,int CB>
__device__ __forceinline__ void frag8(const uint32_t* w,int lane,uint32_t (&f)[4]) {
  if constexpr((K2&1)!=0) {
    static_assert(CB==2&&K2>=3&&K2<=7,"a half-integer K: 1.5, 2.5 or 3.5, mul1");
    frag8_half<K2,CB>(w,lane,f);
  } else {
    frag8_int<K2/2,CB>(w,lane,f);
  }
}
// the 128-point transform of the 128 values a warp holds 4 per lane (value lane * 4 + i), over sqrt(128)
__device__ __forceinline__ void had128(float* v,int lane) {
  const float s0=v[0]+v[1],d0=v[0]-v[1],s1=v[2]+v[3],d1=v[2]-v[3];
  v[0]=s0+s1;v[1]=d0+d1;v[2]=s0-s1;v[3]=d0-d1;
  #pragma unroll
  for(int m=1;m<32;m<<=1) {
    #pragma unroll
    for(int i=0;i<4;++i) {
      const float o=__shfl_xor_sync(0xffffffffu,v[i],m);
      v[i]=(lane&m)?o-v[i]:v[i]+o;
    }
  }
  #pragma unroll
  for(int i=0;i<4;++i)v[i]*=HAD;
}
__device__ __forceinline__ void mma16816(float (&c)[4],uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,uint32_t b0,uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
               :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}
// One [n = outputs, k = inputs] matrix image: trellis [k/16][n/16] tiles, FP16 suh [k], FP16 svh [n].
struct Mat { const uint32_t* tr; const __half* suh; const __half* svh; unsigned ld; };
__host__ __device__ inline size_t matrix_bytes(unsigned n,unsigned k,unsigned k2) {
  return size_t(n)*k*k2/16+2*(size_t(n)+k);
}
__device__ inline Mat mat(const unsigned char* image,unsigned n,unsigned k,unsigned k2) {
  const size_t tr=size_t(n)*k*k2/16;
  return {reinterpret_cast<const uint32_t*>(image),reinterpret_cast<const __half*>(image+tr),
          reinterpret_cast<const __half*>(image+tr+2*size_t(k)),n/16};
}
}
