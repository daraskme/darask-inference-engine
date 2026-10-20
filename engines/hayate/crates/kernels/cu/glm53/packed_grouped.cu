#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#endif
#include "lowbit.cuh"

namespace glm53::packed_grouped {
using namespace lowbit;
constexpr unsigned BLOCKS=512;
// Two adjacent E2M1 nibbles and their shared E4M3 scale. c is even.
// Also compiled by the CPU oracle, including LOWBIT's striped bit planes.
GLM_HD inline unsigned weight_pair(const unsigned char* image,unsigned r,unsigned c,
    unsigned n,unsigned k,MatrixFormat f) {
  unsigned q,scale;
  if(!f.format) {
    q=image[(size_t(r)*k+c)/2];
    unsigned sc=c/16;
    size_t at=((r/128)*(k/64)+sc/4)*512+(r%32)*16+(r%128/32)*4+sc%4;
    scale=image[size_t(n)*k/2+at];
  } else {
    size_t at=size_t(r)*(k/4)+(c/128)*32+c%32;
    unsigned lo=unsigned(image[at])|(unsigned(image[at+1])<<8),shift=2*(c%128/32);
    unsigned a=(lo>>shift)&3,b=(lo>>(shift+8))&3;
    if(bits(f.grid)==3) {
      at=size_t(n)*k/4+size_t(r)*(k/8)+(c/256)*32+c%32;
      unsigned hi=unsigned(image[at])|(unsigned(image[at+1])<<8);
      a|=((hi>>(c%256/32))&1)<<2;b|=((hi>>(c%256/32+8))&1)<<2;
    }
    q=nibble(f.grid,a)|(nibble(f.grid,b)<<4);
    unsigned shift_group=f.group==16?4:5;
    scale=image[scale_start(n,k,f.grid)+size_t(r)*(k>>shift_group)+(c>>shift_group)];
  }
  return q|(scale<<8);
}
// Layout of a single staged packed row. LOWBIT's 32-byte high plane fits
// behind its low plane even at K=128. Row padding keeps 16 B copy alignment
// and avoids the 4/8-way shared bank conflicts of a power-of-two row stride.
template<unsigned TM,unsigned TN,unsigned TK,unsigned TW,unsigned ST>
struct Config {
  static constexpr unsigned m=TM,n=TN,k=TK,warps=TW,stages=ST;
  static constexpr unsigned payload=TK/2+16,scales=TK/16,weight=TN*(payload+scales),xs=TK+8;
  static_assert((TK==128||TK==256)&&TN==16*TW);
  GLM_HD static constexpr unsigned shared(bool down) {return ST*(TM*xs*2+(down?1:2)*weight);}
};
using C0=Config<64,128,128,8,2>;
using C1=Config<32,128,128,8,3>;
using C2=Config<32,64,128,4,2>;
using C3=Config<32,64,256,4,2>;
GLM_HD inline unsigned payload_bytes(unsigned k,MatrixFormat f) {
  return f.format?k/4+(bits(f.grid)==3?32:0):k/2;
}
// Source addresses are shared with the CPU staging oracle. Copies are 16 B
// for bit planes, 4 B for scales (NVFP4 scales are swizzled in 128x64 tiles).
GLM_HD inline size_t payload_source(unsigned r,unsigned byte,unsigned base,unsigned n,unsigned k,
    unsigned tk,MatrixFormat f) {
  if(!f.format)return (size_t(r)*k+base)/2+byte;
  if(byte<tk/4)return size_t(r)*(k/4)+(base/128)*32+byte;
  return size_t(n)*k/4+size_t(r)*(k/8)+(base/256)*32+byte-tk/4;
}
GLM_HD inline size_t scale_source(unsigned r,unsigned byte,unsigned base,unsigned n,unsigned k,MatrixFormat f) {
  if(f.format) {
    unsigned shift=f.group==16?4:5;
    return scale_start(n,k,f.grid)+size_t(r)*(k>>shift)+(base>>shift)+byte;
  }
  unsigned sc=base/16+byte;
  return size_t(n)*k/2+((r/128)*(k/64)+sc/4)*512+(r%32)*16+(r%128/32)*4;
}
GLM_HD inline unsigned staged_pair(const unsigned char* row,const unsigned char* scales,
    unsigned c,unsigned base,unsigned tk,MatrixFormat f) {
  unsigned q;
  if(!f.format)q=row[c/2];
  else {
    unsigned at=c/128*32+c%32,shift=2*(c%128/32);
    unsigned lo=unsigned(row[at])|(unsigned(row[at+1])<<8);
    unsigned a=(lo>>shift)&3,b=(lo>>(shift+8))&3;
    if(bits(f.grid)==3) {
      at=tk/4+c%32;
      unsigned hi=unsigned(row[at])|(unsigned(row[at+1])<<8),strip=(base+c)%256/32;
      a|=((hi>>strip)&1)<<2;b|=((hi>>(strip+8))&1)<<2;
    }
    q=nibble(f.grid,a)|(nibble(f.grid,b)<<4);
  }
  unsigned shift=f.format&&f.group==32?5:4;
  return q|(unsigned(scales[c>>shift])<<8);
}
GLM_HD inline unsigned tile_count(unsigned rows,unsigned n,unsigned tm,unsigned tn) {
  return ((rows+tm-1)/tm)*(n/tn);
}
GLM_HD inline unsigned find_expert(const unsigned* ends,unsigned count,unsigned tile) {
  unsigned lo=0,hi=count;
  while(lo<hi) {unsigned mid=(lo+hi)/2;if(tile<ends[mid])hi=mid;else lo=mid+1;}
  return lo;
}
}

#ifdef __CUDACC__
namespace {
using namespace glm53;
using namespace glm53::lowbit;
using namespace glm53::packed_grouped;
using B=__nv_bfloat16;
__device__ __forceinline__ unsigned unpack(unsigned raw) {
  auto v=__half22float2(static_cast<__half2>(__nv_cvt_fp4x2_to_halfraw2(raw&255,__NV_E2M1)));
  __nv_fp8_e4m3 scale;scale.__x=raw>>8;
  auto pair=__floats2bfloat162_rn(v.x*float(scale),v.y*float(scale));
  __nv_bfloat162_raw bits=pair;return unsigned(bits.x)|(unsigned(bits.y)<<16);
}
__device__ __forceinline__ unsigned ld32(const B* p) {return *reinterpret_cast<const unsigned*>(p);}
__device__ __forceinline__ void mma(float* c,const unsigned* a,const unsigned* b) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
      : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
}
template<unsigned BYTES>
__device__ __forceinline__ void copy_async(void* dst,const void* src,unsigned valid=BYTES) {
  unsigned address=static_cast<unsigned>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;"::
      "r"(address),"l"(src),"n"(BYTES),"r"(valid):"memory");
}
template<unsigned PENDING>
__device__ __forceinline__ void ready() {
  asm volatile("cp.async.wait_group %0;"::"n"(PENDING):"memory");
}
template<class C,unsigned BYTES,unsigned OFFSET=0>
__device__ __forceinline__ void prefetch_plane(unsigned char* dst,const unsigned char* image,
    unsigned n0,unsigned base,unsigned n,unsigned k,MatrixFormat f) {
  for(unsigned i=threadIdx.x;i<C::n*BYTES/16;i+=C::warps*32) {
    unsigned r=i/(BYTES/16),c=i%(BYTES/16)*16+OFFSET;
    copy_async<16>(dst+r*C::payload+c,image+payload_source(n0+r,c,base,n,k,C::k,f));
  }
}
template<class C>
__device__ __forceinline__ void prefetch_weight(unsigned char* dst,const unsigned char* image,
    unsigned n0,unsigned base,unsigned n,unsigned k,MatrixFormat f) {
  // Constant power-of-two copy widths: no integer divide for row addresses.
  if(!f.format)prefetch_plane<C,C::k/2>(dst,image,n0,base,n,k,f);
  else {
    prefetch_plane<C,C::k/4>(dst,image,n0,base,n,k,f);
    if(bits(f.grid)==3)prefetch_plane<C,32,C::k/4>(dst,image,n0,base,n,k,f);
  }
  unsigned shift=f.format&&f.group==32?5:4;
  unsigned row_shift=(C::k==128?7:8)-shift-2;
  for(unsigned i=threadIdx.x;i<(C::n<<row_shift);i+=C::warps*32) {
    unsigned r=i>>row_shift,c=(i&((1u<<row_shift)-1))*4;
    copy_async<4>(dst+C::n*C::payload+r*C::scales+c,image+scale_source(n0+r,c,base,n,k,f));
  }
}
template<bool DOWN,class C>
__device__ __forceinline__ void prefetch(unsigned char* stage,const B* input,const unsigned char* image,
    size_t step,unsigned rows,unsigned n0,unsigned base,unsigned n,unsigned k,MatrixFormat f) {
  B* xs=reinterpret_cast<B*>(stage);
  for(unsigned i=threadIdx.x;i<C::m*C::k/8;i+=C::warps*32) {
    unsigned r=i/(C::k/8),c=i%(C::k/8)*8;
    const B* src=r<rows?input+size_t(r)*k+base+c:input;
    copy_async<16>(xs+r*C::xs+c,src,r<rows?16:0);
  }
  auto weights=stage+C::m*C::xs*2;
  prefetch_weight<C>(weights,image,n0,base,n,k,f);
  if constexpr(!DOWN)prefetch_weight<C>(weights+C::weight,image+step,n0,base,n,k,f);
  asm volatile("cp.async.commit_group;":::"memory");
}
__device__ __forceinline__ float tensor(const unsigned char* image,unsigned n,unsigned k,MatrixFormat f) {
  return *reinterpret_cast<const float*>(image+storage(n,k,f)-4);
}
template<class C>
__device__ __forceinline__ unsigned weight(const unsigned char* stage,unsigned r,unsigned c,
    unsigned base,MatrixFormat f) {
  return unpack(staged_pair(stage+r*C::payload,stage+C::n*C::payload+r*C::scales,c,base,C::k,f));
}
// All operands are prefetched together. Only packed bytes, scales and activation
// tiles occupy shared memory; weights are decoded from shared into MMA registers.
// Accumulators keep the original ascending m16n8k16 order and BF16 epilogue.
// Budget registers for two CTAs; shared memory can limit actual residency.
template<bool DOWN,class C>
__global__ void __launch_bounds__(C::warps*32,2) grouped(B* out,const B* x,const void* const* images,
    const ExpertFormat* formats,const unsigned* ranges,unsigned count,unsigned hidden,unsigned intermediate) {
  __shared__ unsigned ends[288],job[3];
  extern __shared__ __align__(16) unsigned char smem[];
  constexpr unsigned SB=C::shared(DOWN)/C::stages;
  const unsigned tid=threadIdx.x,warp=tid/32,lane=tid%32,g=lane/4,t=lane%4;
  const unsigned n=DOWN?hidden:intermediate,k=DOWN?intermediate:hidden;
  if(tid==0) {
    unsigned sum=0;
    for(unsigned j=0;j<count;++j) {sum+=tile_count(ranges[j*2],n,C::m,C::n);ends[j]=sum;}
  }
  __syncthreads();
  for(unsigned tile=blockIdx.x;tile<ends[count-1];tile+=gridDim.x) {
    if(tid==0) {
      unsigned j=find_expert(ends,count,tile),local=tile-(j?ends[j-1]:0);
      job[0]=j;job[1]=(local/(n/C::n))*C::m;job[2]=(local%(n/C::n))*C::n;
    }
    __syncthreads();
    unsigned expert=job[0],m0=job[1],n0=job[2],rows=ranges[2*expert]-m0;
    unsigned offset=ranges[2*expert+1]+m0;
    ExpertFormat ef=formats[expert];MatrixFormat f=DOWN?ef.down:ef.gate_up;
    auto image=static_cast<const unsigned char*>(images[expert]);
    size_t step=stride(intermediate,hidden,ef.gate_up);
    image+=DOWN?2*step:0;
    float scale=tensor(image,n,k,f),up_scale=DOWN?0:tensor(image+step,n,k,f);
    float acc[C::m/16][2][4]={},up[DOWN?1:C::m/16][2][4]={};
    const B* input=x+size_t(offset)*k;
    #pragma unroll
    for(unsigned s=0;s<C::stages;++s)
      if(s*C::k<k)prefetch<DOWN,C>(smem+s*SB,input,image,step,rows,n0,s*C::k,n,k,f);
    for(unsigned base=0;base<k;base+=C::k) {
      // Drain the final groups as well; a fixed wait count would read stale
      // shared memory during the last one/two K tiles (and for small shapes).
      unsigned remaining=(k-base)/C::k;
      if constexpr(C::stages==3) {
        if(remaining>=3)ready<2>();else if(remaining==2)ready<1>();else ready<0>();
      } else {
        if(remaining>=2)ready<1>();else ready<0>();
      }
      __syncthreads();
      unsigned stage=(base/C::k)%C::stages;
      const B* xs=reinterpret_cast<const B*>(smem+stage*SB);
      const unsigned char* ws=smem+stage*SB+C::m*C::xs*2;
      #pragma unroll 2
      for(unsigned kk=0;kk<C::k;kk+=16) {
        unsigned b[2][2],u[2][2];
        #pragma unroll
        for(unsigned j=0;j<2;++j) {
          unsigned r=warp*16+j*8+g,c=kk+2*t;
          b[j][0]=weight<C>(ws,r,c,base,f);
          b[j][1]=weight<C>(ws,r,c+8,base,f);
          if constexpr(!DOWN) {
            u[j][0]=weight<C>(ws+C::weight,r,c,base,f);
            u[j][1]=weight<C>(ws+C::weight,r,c+8,base,f);
          }
        }
        #pragma unroll
        for(unsigned i=0;i<C::m/16;++i) {
          const B* a0=xs+(i*16+g)*C::xs+kk+2*t;
          unsigned a[4]={ld32(a0),ld32(a0+8*C::xs),ld32(a0+8),ld32(a0+8*C::xs+8)};
          #pragma unroll
          for(unsigned j=0;j<2;++j) {
            mma(acc[i][j],a,b[j]);
            if constexpr(!DOWN)mma(up[i][j],a,u[j]);
          }
        }
      }
      __syncthreads();
      if(base+C::stages*C::k<k)
        prefetch<DOWN,C>(smem+stage*SB,input,image,step,rows,n0,base+C::stages*C::k,n,k,f);
    }
    #pragma unroll
    for(unsigned i=0;i<C::m/16;++i) {
      #pragma unroll
      for(unsigned j=0;j<2;++j) {
        #pragma unroll
        for(unsigned v=0;v<4;++v) {
          unsigned r=i*16+g+(v/2)*8,c=n0+warp*16+j*8+2*t+(v%2);
          if(r<rows) {
            float a=bf16(acc[i][j][v]*scale);
            if constexpr(!DOWN)a=swiglu(a,bf16(up[i][j][v]*up_scale));
            out[(size_t(offset)+r)*n+c]=__float2bfloat16_rn(a);
          }
        }
      }
    }
    __syncthreads();
  }
}
// Host-thread-local tuning also works with concurrent streams on other host
// threads. Set before graph capture; replay keeps the captured configuration.
thread_local unsigned configuration=0;
template<bool DOWN,class C>
int launch(B* out,const B* x,const void* const* images,const ExpertFormat* formats,
    const unsigned* ranges,unsigned count,unsigned hidden,unsigned intermediate,cudaStream_t st) {
  constexpr unsigned shared=C::shared(DOWN);
  static_assert(shared+288*4+3*4<=99*1024);
  // Both supported machines use device 0. Warmup configures the attribute
  // once, so graph capture and subsequent launches only enqueue the kernel.
  static const cudaError_t configured=[] {
    if constexpr(shared>48*1024)
      return cudaFuncSetAttribute(grouped<DOWN,C>,cudaFuncAttributeMaxDynamicSharedMemorySize,shared);
    else return cudaSuccess;
  }();
  if(configured!=cudaSuccess)return configured;
  grouped<DOWN,C><<<BLOCKS,C::warps*32,shared,st>>>(out,x,images,formats,ranges,count,hidden,intermediate);
  return cudaGetLastError();
}
template<bool DOWN>
int dispatch(B* out,const B* x,const void* const* images,const ExpertFormat* formats,
    const unsigned* ranges,unsigned count,unsigned hidden,unsigned intermediate,cudaStream_t st) {
  switch(configuration) {
    case 0:return launch<DOWN,C0>(out,x,images,formats,ranges,count,hidden,intermediate,st);
    case 1:return launch<DOWN,C1>(out,x,images,formats,ranges,count,hidden,intermediate,st);
    case 2:return launch<DOWN,C2>(out,x,images,formats,ranges,count,hidden,intermediate,st);
    default:return launch<DOWN,C3>(out,x,images,formats,ranges,count,hidden,intermediate,st);
  }
}
bool valid(void* out,const void* x,const void* images,const void* formats,const void* ranges,
    unsigned count,unsigned hidden,unsigned intermediate) {
  return out&&x&&out!=x&&images&&formats&&ranges&&count&&count<=288
      &&hidden&&hidden<=4096&&hidden%256==0&&intermediate&&intermediate<=4096&&intermediate%256==0;
}
}
extern "C" int hayate_glm_grouped_packed_gate_up(void* product,const void* x,const void* images,
    const ExpertFormat* formats,const void* ranges,unsigned count,unsigned hidden,unsigned intermediate,cudaStream_t st) {
  if(!valid(product,x,images,formats,ranges,count,hidden,intermediate))return cudaErrorInvalidValue;
  return dispatch<false>((B*)product,(const B*)x,(const void* const*)images,formats,
      (const unsigned*)ranges,count,hidden,intermediate,st);
}
extern "C" int hayate_glm_grouped_packed_down(void* out,const void* product,const void* images,
    const ExpertFormat* formats,const void* ranges,unsigned count,unsigned hidden,unsigned intermediate,cudaStream_t st) {
  if(!valid(out,product,images,formats,ranges,count,hidden,intermediate))return cudaErrorInvalidValue;
  return dispatch<true>((B*)out,(const B*)product,(const void* const*)images,formats,
      (const unsigned*)ranges,count,hidden,intermediate,st);
}
// Does not change either launch ABI. No device work or synchronization.
extern "C" int hayate_glm_grouped_packed_set_config(unsigned config) {
  if(config>3)return cudaErrorInvalidValue;
  configuration=config;return cudaSuccess;
}
#endif
