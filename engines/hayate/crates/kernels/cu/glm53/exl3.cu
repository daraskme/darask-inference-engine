// EXL3 routed experts (crates/models/src/glm53/EXL3.md): the decode MoE, the prefill grouped GEMMs and a BF16
// restore.  An expert image is gate | up | down, each [outputs, inputs] = trellis | FP16 suh | FP16 svh, and
// y = H(H(x * suh) W') * svh (H: 128-point Hadamard over sqrt(128)).  The kernels follow Project Maya PR #265's
// (MIT) block GEMV and tensor-core GEMM after exllamav3 (MIT): a block owns 128 outputs (one rotation block), rotates
// its inputs 128 at a time into FP16 in shared memory and decodes each 16x16 tile straight into mma.m16n8k16
// fragments.  Inputs and outputs keep Hayate's layouts (BF16 rows, the LOWBIT kernels' [9][M][n] decode buffers, the
// packed grouped GEMM's ranges), so the router, LoRA side paths, gather/scatter and CUDA graphs are unchanged.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "dense.cuh"
#include "exl3.cuh"
namespace {
using namespace glm53;
using namespace glm53::exl3;
constexpr unsigned WARPS=8,THREADS=256;
__device__ __forceinline__ float bf(B v) { return __bfloat162float(v); }
__device__ __forceinline__ float hf(__half v) { return __half2float(v); }

// ---------------------------------------------------------------- decode
// out[t * 128 + i] (shared F32) = (H(x_t * suh) W')[c0 + i] for t < m: warp w takes tile column c0/16 + w through
// every 16-input slice; per 128 inputs warp t (< m) rotates activation t into FP16 (xq), then each warp decodes its
// tile of the chunk's 8 slices as mma A fragments (the activations are B's columns).  The next chunk's tile words are
// loaded while this one is decoded.  Ends with a __syncthreads (out complete).
template<int K2,int CB>
__device__ void block_mv(const Mat w,unsigned k,unsigned c0,const B* x,unsigned m,__half* xq,uint32_t* stage,float* out) {
  constexpr int W=4*K2;
  static_assert(W<=32,"a tile is at most 32 words");
  const int lane=threadIdx.x&31,warp=threadIdx.x>>5,g=lane>>2,t=lane&3;
  const unsigned nq=k/128;
  uint32_t* st=stage+warp*32;
  const uint32_t* col=w.tr+size_t(c0/16+warp)*W+lane;
  const size_t slice=size_t(w.ld)*W;
  uint32_t wc[8],wn[8];
  #pragma unroll
  for(int s=0;s<8;++s)wc[s]=lane<W?__ldg(col+s*slice):0u;
  float c[4]={0.f,0.f,0.f,0.f};
  for(unsigned q=0;q<nq;++q) {
    if(warp<int(m)) {
      const B* xr=x+size_t(warp)*k+q*128+lane*4;
      const __half* sr=w.suh+q*128+lane*4;
      float v[4];
      #pragma unroll
      for(int i=0;i<4;++i)v[i]=bf(xr[i])*hf(sr[i]);
      had128(v,lane);
      __half2* d=reinterpret_cast<__half2*>(xq+warp*128+lane*4);
      d[0]=__floats2half2_rn(v[0],v[1]);
      d[1]=__floats2half2_rn(v[2],v[3]);
    }
    __syncthreads();
    if(q+1<nq) {
      #pragma unroll
      for(int s=0;s<8;++s)wn[s]=lane<W?__ldg(col+((q+1)*8+s)*slice):0u;
    }
    #pragma unroll
    for(int s=0;s<8;++s) {
      __syncwarp();
      if(lane<W)st[lane]=wc[s];
      __syncwarp();
      uint32_t f[4];
      frag8<K2,CB>(st,lane,f);
      uint32_t b0=0,b1=0;
      if(g<int(m)) {
        const __half* xr=xq+g*128+s*16+2*t;
        b0=*reinterpret_cast<const uint32_t*>(xr);
        b1=*reinterpret_cast<const uint32_t*>(xr+8);
      }
      // A: rows (outputs) g / g + 8, columns (inputs) 2t.. / 2t + 8..: steps 0,1 / 4,5 / 2,3 / 6,7
      mma16816(c,f[0],f[2],f[1],f[3],b0,b1);
    }
    __syncthreads();
    #pragma unroll
    for(int s=0;s<8;++s)wc[s]=wn[s];
  }
  // c: outputs warp*16 + g (c[0], c[1]) and + 8 (c[2], c[3]) for activations 2t and 2t + 1
  const int o=warp*16+g;
  if(2*t<int(m)){out[(2*t)*128+o]=c[0];out[(2*t)*128+o+8]=c[2];}
  if(2*t+1<int(m)){out[(2*t+1)*128+o]=c[1];out[(2*t+1)*128+o+8]=c[3];}
  __syncthreads();
}
// y[i] = (H(out_t) * svh)[c0 + lane*4 + i]
__device__ __forceinline__ void rotate_out(const float* out,const __half* svh,unsigned c0,int t,float (&y)[4]) {
  const int lane=threadIdx.x&31;
  #pragma unroll
  for(int i=0;i<4;++i)y[i]=out[t*128+lane*4+i];
  had128(y,lane);
  #pragma unroll
  for(int i=0;i<4;++i)y[i]*=hf(svh[c0+lane*4+i]);
}
// The shared expert's rows [row0, row0 + 128), 16 a warp, with the dense kernels' arithmetic (lowbit.cu's g == 8).
template<unsigned M,bool DOWN>
__device__ void shared_rows(B* out,const B* x,DenseWeight a_w,DenseWeight b_w,unsigned row0,unsigned n,unsigned k) {
  const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
  for(unsigned i=0;i<16;++i) {
    const unsigned row=row0+warp*16+i;
    float a[M],b[M];
    dot_rows(a,x,a_w,row,n,k,true);
    if(!DOWN)dot_rows(b,x,b_w,row,n,k,true);
    if(lane==0)
      #pragma unroll
      for(unsigned t=0;t<M;++t)out[(size_t(8)*M+t)*n+row]=__float2bfloat16_rn(DOWN?a[t]:swiglu(bf16(a[t]),bf16(b[t])));
  }
}
template<bool DOWN>
__device__ void shared_block(B* out,const B* x,DenseWeight a_w,DenseWeight b_w,unsigned m,unsigned row0,unsigned n,unsigned k) {
  switch(m) {
    case 1:shared_rows<1,DOWN>(out,x,a_w,b_w,row0,n,k);break;
    case 2:shared_rows<2,DOWN>(out,x,a_w,b_w,row0,n,k);break;
    case 3:shared_rows<3,DOWN>(out,x,a_w,b_w,row0,n,k);break;
    default:shared_rows<4,DOWN>(out,x,a_w,b_w,row0,n,k);break;
  }
}
// grid (inter / 128, 9): expert blockIdx.y's gate and up for 128 outputs, the SwiGLU into product[9][m][inter]
// (null image: zeros, as lowbit.cu); y == 8 the shared expert.
template<int K2,int CB>
__global__ void __launch_bounds__(THREADS) gate_up(B* product,const B* x,const void* const* images,DenseWeight gate,
    DenseWeight up,unsigned m,unsigned hidden,unsigned inter) {
  __shared__ __align__(16) __half xq[4*128];
  __shared__ uint32_t stage[WARPS*32];
  __shared__ float og[4*128],ou[4*128];
  const unsigned c0=blockIdx.x*128,e=blockIdx.y;
  const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
  if(e==8) {
    shared_block<false>(product,x,gate,up,m,c0,inter,hidden);
    return;
  }
  auto image=static_cast<const unsigned char*>(images[e]);
  if(!image) {
    for(unsigned i=threadIdx.x;i<m*128;i+=THREADS)product[(size_t(e)*m+i/128)*inter+c0+i%128]=__float2bfloat16_rn(0.f);
    return;
  }
  const Mat gm=mat(image,inter,hidden,K2),um=mat(image+matrix_bytes(inter,hidden,K2),inter,hidden,K2);
  block_mv<K2,CB>(gm,hidden,c0,x,m,xq,stage,og);
  block_mv<K2,CB>(um,hidden,c0,x,m,xq,stage,ou);
  if(warp<int(m)) {
    float a[4],b[4];
    rotate_out(og,gm.svh,c0,warp,a);
    rotate_out(ou,um.svh,c0,warp,b);
    B* o=product+(size_t(e)*m+warp)*inter+c0+lane*4;
    #pragma unroll
    for(int i=0;i<4;++i)o[i]=__float2bfloat16_rn(swiglu(bf16(a[i]),bf16(b[i])));
  }
}
// grid (hidden / 128, 9): expert blockIdx.y's down for 128 outputs into out[9][m][hidden] (null image: untouched,
// for the CPU mailbox merge); y == 8 the shared expert.  gate_k2 locates the down matrix in the image.
template<int K2,int CB>
__global__ void __launch_bounds__(THREADS) down(B* out,const B* product,const void* const* images,DenseWeight shared,
    unsigned m,unsigned gate_k2,unsigned hidden,unsigned inter) {
  __shared__ __align__(16) __half xq[4*128];
  __shared__ uint32_t stage[WARPS*32];
  __shared__ float od[4*128];
  const unsigned c0=blockIdx.x*128,e=blockIdx.y;
  const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
  if(e==8) {
    shared_block<true>(out,product+size_t(8)*m*inter,shared,shared,m,c0,hidden,inter);
    return;
  }
  auto image=static_cast<const unsigned char*>(images[e]);
  if(!image)return;
  const Mat dm=mat(image+2*matrix_bytes(inter,hidden,gate_k2),hidden,inter,K2);
  block_mv<K2,CB>(dm,inter,c0,product+size_t(e)*m*inter,m,xq,stage,od);
  if(warp<int(m)) {
    float y[4];
    rotate_out(od,dm.svh,c0,warp,y);
    B* o=out+(size_t(e)*m+warp)*hidden+c0+lane*4;
    #pragma unroll
    for(int i=0;i<4;++i)o[i]=__float2bfloat16_rn(y[i]);
  }
}

// ---------------------------------------------------------------- prefill: grouped tensor-core GEMMs
// A tile: BM rows of one expert x 128 outputs.  Per 128 inputs the rows are scaled by suh, rotated and stored FP16 in
// shared memory (the A operands); per 16-input slice each warp decodes its 16x16 tile straight into B fragments (a
// lane's 8 steps are its fragment order) and multiplies every m-tile with them; the output rotation and svh run on
// the F32 tile in shared memory.  Tiles are scheduled over a persistent grid as packed_grouped.cu's.
namespace grouped {
constexpr int BM=32,MT=BM/16,AP=136;
constexpr unsigned BLOCKS=512;
__device__ void load_a(const B* x,unsigned ldx,int nrows,unsigned k0,const __half* suh,__half* A) {
  const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
  for(int rr=warp;rr<BM;rr+=WARPS) {
    float v[4]={0.f,0.f,0.f,0.f};
    if(rr<nrows) {
      const B* src=x+size_t(rr)*ldx+k0+lane*4;
      #pragma unroll
      for(int i=0;i<4;++i)v[i]=bf(src[i])*hf(suh[k0+lane*4+i]);
      had128(v,lane);
    }
    #pragma unroll
    for(int i=0;i<4;++i)A[rr*AP+lane*4+i]=__float2half_rn(v[i]);
  }
}
__device__ __forceinline__ void a_frags(const __half* A,int mt,int s,int lane,uint32_t (&a)[4]) {
  const int g=lane>>2,t=lane&3;
  const __half* p=A+(mt*16+g)*AP+s*16+2*t;
  a[0]=*reinterpret_cast<const uint32_t*>(p);
  a[1]=*reinterpret_cast<const uint32_t*>(p+8*AP);
  a[2]=*reinterpret_cast<const uint32_t*>(p+8);
  a[3]=*reinterpret_cast<const uint32_t*>(p+8*AP+8);
}
__device__ __forceinline__ void store_acc(const float (&acc)[MT][2][4],float* T) {
  const int lane=threadIdx.x&31,warp=threadIdx.x>>5,g=lane>>2,t=lane&3;
  #pragma unroll
  for(int mt=0;mt<MT;++mt)
    #pragma unroll
    for(int ns=0;ns<2;++ns) {
      const int c=warp*16+ns*8+2*t,r=mt*16+g;
      T[r*128+c]=acc[mt][ns][0];
      T[r*128+c+1]=acc[mt][ns][1];
      T[(r+8)*128+c]=acc[mt][ns][2];
      T[(r+8)*128+c+1]=acc[mt][ns][3];
    }
}
__device__ __forceinline__ unsigned find_expert(const unsigned* ends,unsigned count,unsigned tile) {
  unsigned lo=0,hi=count;
  while(lo<hi){unsigned mid=(lo+hi)/2;if(tile<ends[mid])hi=mid;else lo=mid+1;}
  return lo;
}
// DOWN: out[rows][hidden] = down(x[rows][inter]); else out[rows][inter] = SwiGLU(gate, up)(x[rows][hidden]).
// Expert j's rows are ranges[2j] rows from row ranges[2j + 1] of x and out.
template<int K2,int CB,bool DOWN>
__global__ void __launch_bounds__(THREADS) gemm(B* out,const B* x,const void* const* images,const unsigned* ranges,
    unsigned count,unsigned gate_k2,unsigned hidden,unsigned inter) {
  constexpr int W=4*K2,NMAT=DOWN?1:2;
  __shared__ unsigned ends[288],job[3];
  __shared__ __align__(16) unsigned char sm[2*BM*128*4];   // A (BM x AP halves each), then the F32 tiles
  __shared__ uint32_t stage[WARPS][32];
  const unsigned n=DOWN?hidden:inter,k=DOWN?inter:hidden,blocks=n/128;
  const int tid=threadIdx.x,lane=tid&31,warp=tid>>5;
  if(tid==0) {
    unsigned sum=0;
    for(unsigned j=0;j<count;++j){sum+=(ranges[2*j]+BM-1)/BM*blocks;ends[j]=sum;}
  }
  __syncthreads();
  __half* A[2]={reinterpret_cast<__half*>(sm),reinterpret_cast<__half*>(sm)+BM*AP};
  float* T[2]={reinterpret_cast<float*>(sm),reinterpret_cast<float*>(sm)+BM*128};
  const size_t gate_bytes=matrix_bytes(inter,hidden,gate_k2);
  for(unsigned tile=blockIdx.x;tile<ends[count-1];tile+=gridDim.x) {
    if(tid==0) {
      const unsigned j=find_expert(ends,count,tile),local=tile-(j?ends[j-1]:0);
      job[0]=j;job[1]=local/blocks*BM;job[2]=local%blocks;
    }
    __syncthreads();
    const unsigned j=job[0],r0=job[1],cb=job[2];
    const int nrows=min(BM,int(ranges[2*j]-r0));
    const unsigned offset=ranges[2*j+1]+r0;
    auto image=static_cast<const unsigned char*>(images[j]);
    Mat w[2];
    if(DOWN)w[0]=mat(image+2*gate_bytes,hidden,inter,K2);
    else {w[0]=mat(image,inter,hidden,K2);w[1]=mat(image+gate_bytes,inter,hidden,K2);}
    const B* input=x+size_t(offset)*k;
    float acc[NMAT][MT][2][4]={};
    for(unsigned k0=0;k0<k;k0+=128) {
      __syncthreads();
      #pragma unroll
      for(int mm=0;mm<NMAT;++mm)load_a(input,k,nrows,k0,w[mm].suh,A[mm]);
      __syncthreads();
      #pragma unroll 1
      for(int s=0;s<8;++s) {
        const unsigned ks=k0/16+s;
        #pragma unroll
        for(int mm=0;mm<NMAT;++mm) {
          const uint32_t* src=w[mm].tr+(size_t(ks)*w[mm].ld+cb*8+warp)*W;
          __syncwarp();
          if(lane<W)stage[warp][lane]=__ldg(src+lane);
          __syncwarp();
          uint32_t f[4];
          frag8<K2,CB>(stage[warp],lane,f);
          #pragma unroll
          for(int mt=0;mt<MT;++mt) {
            uint32_t a[4];
            a_frags(A[mm],mt,s,lane,a);
            mma16816(acc[mm][mt][0],a[0],a[1],a[2],a[3],f[0],f[1]);
            mma16816(acc[mm][mt][1],a[0],a[1],a[2],a[3],f[2],f[3]);
          }
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for(int mm=0;mm<NMAT;++mm)store_acc(acc[mm],T[mm]);
    __syncthreads();
    for(int rr=warp;rr<nrows;rr+=WARPS) {
      float v[4];
      #pragma unroll
      for(int i=0;i<4;++i)v[i]=T[0][rr*128+lane*4+i];
      had128(v,lane);
      #pragma unroll
      for(int i=0;i<4;++i)v[i]*=hf(w[0].svh[cb*128+lane*4+i]);
      if(!DOWN) {
        float u[4];
        #pragma unroll
        for(int i=0;i<4;++i)u[i]=T[NMAT-1][rr*128+lane*4+i];
        had128(u,lane);
        #pragma unroll
        for(int i=0;i<4;++i)v[i]=swiglu(bf16(v[i]),bf16(u[i]*hf(w[NMAT-1].svh[cb*128+lane*4+i])));
      }
      B* o=out+size_t(offset+rr)*n+cb*128+lane*4;
      #pragma unroll
      for(int i=0;i<4;++i)o[i]=__float2bfloat16_rn(v[i]);
    }
    __syncthreads();
  }
}
}

// ---------------------------------------------------------------- restore
// out[n][k] BF16 = W (Hayate's row layout) of one matrix: grid (n / 128, k / 128), the 128 x 128 block decoded into
// shared memory, rotated along both indices and scaled.  For the single-token prefill path and the tests.
template<int K2,int CB>
__global__ void __launch_bounds__(THREADS) restore(B* out,const unsigned char* image,unsigned n,unsigned k) {
  extern __shared__ float T[];   // [128 inputs][129]
  __shared__ uint32_t stage[WARPS][32];
  constexpr int W=4*K2,P=129;
  const unsigned n0=blockIdx.x*128,k0=blockIdx.y*128;
  const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
  const Mat w=mat(image,n,k,K2);
  for(int s=0;s<8;++s) {
    const uint32_t* src=w.tr+(size_t(k0/16+s)*w.ld+n0/16+warp)*W;
    __syncwarp();
    if(lane<W)stage[warp][lane]=src[lane];
    __syncwarp();
    uint32_t f[4];
    frag8<K2,CB>(stage[warp],lane,f);
    #pragma unroll
    for(int i=0;i<4;++i) {
      const __half2 h=*reinterpret_cast<const __half2*>(&f[i]);
      #pragma unroll
      for(int e=0;e<2;++e) {
        const int j=2*i+e,r=(lane&3)*2+(j&1)+8*((j>>1)&1),c=(lane>>2)+8*(j>>2);
        T[(s*16+r)*P+warp*16+c]=e?__high2float(h):__low2float(h);
      }
    }
  }
  __syncthreads();
  for(int cc=0;cc<16;++cc) {   // along the inputs, per output column
    const int c=warp*16+cc;
    float v[4];
    #pragma unroll
    for(int i=0;i<4;++i)v[i]=T[(lane*4+i)*P+c];
    had128(v,lane);
    #pragma unroll
    for(int i=0;i<4;++i)T[(lane*4+i)*P+c]=v[i];
  }
  __syncthreads();
  for(int rr=0;rr<16;++rr) {   // along the outputs, per input row; then suh and svh
    const int r=warp*16+rr;
    float v[4];
    #pragma unroll
    for(int i=0;i<4;++i)v[i]=T[r*P+lane*4+i];
    had128(v,lane);
    const float su=hf(w.suh[k0+r]);
    #pragma unroll
    for(int i=0;i<4;++i)T[r*P+lane*4+i]=v[i]*hf(w.svh[n0+lane*4+i])*su;
  }
  __syncthreads();
  for(int cc=0;cc<16;++cc) {
    const int c=warp*16+cc;
    #pragma unroll
    for(int i=0;i<4;++i)out[size_t(n0+c)*k+k0+lane*4+i]=__float2bfloat16_rn(T[(lane*4+i)*P+c]);
  }
}

// ---------------------------------------------------------------- the PCIe lane's fetch
// Copies up to 8 expert images from mapped (pinned) host memory, src[j] (null: none), into device slots dst[j],
// `bytes` each: wide reads, four 16-byte loads in flight a thread, so the link streams instead of serving the
// GEMV's small tile reads.  grid (blocks, 8).
__global__ void __launch_bounds__(THREADS) fetch(void* const* dst,const void* const* src,size_t bytes) {
  const uint4* s=static_cast<const uint4*>(src[blockIdx.y]);
  if(!s)return;
  uint4* d=static_cast<uint4*>(dst[blockIdx.y]);
  const size_t n=bytes/16,step=size_t(gridDim.x)*THREADS;
  size_t i=size_t(blockIdx.x)*THREADS+threadIdx.x;
  for(;i+3*step<n;i+=4*step) {
    const uint4 a=s[i],b=s[i+step],c=s[i+2*step],e=s[i+3*step];
    d[i]=a;d[i+step]=b;d[i+2*step]=c;d[i+3*step]=e;
  }
  for(;i<n;i+=step)d[i]=s[i];
}

// ---------------------------------------------------------------- dispatch: mul1 at K 2, 2.5, 3, 3.5, 4
#define EXL3_K2(F) switch(k2){case 4:F(4);break;case 5:F(5);break;case 6:F(6);break;case 7:F(7);break;case 8:F(8);break;default:return cudaErrorInvalidValue;}
bool supported(unsigned k2) { return k2>=4&&k2<=8; }
bool dims(unsigned hidden,unsigned inter) { return hidden&&inter&&hidden%128==0&&inter%128==0; }
}

extern "C" int hayate_glm_exl3_gate_up(void* product,const void* x,const void* images,DenseWeight gate,DenseWeight up,
    unsigned m,unsigned k2,unsigned hidden,unsigned inter,cudaStream_t s) {
  if(!product||!x||!images||!gate.data||!up.data||m<1||m>4||!supported(k2)||!dims(hidden,inter))return cudaErrorInvalidValue;
  #define RUN(K) gate_up<K,2><<<dim3(inter/128,9),THREADS,0,s>>>((B*)product,(const B*)x,(const void* const*)images,gate,up,m,hidden,inter)
  EXL3_K2(RUN)
  #undef RUN
  return cudaGetLastError();
}
extern "C" int hayate_glm_exl3_down(void* out,const void* product,const void* images,DenseWeight shared,
    unsigned m,unsigned gate_k2,unsigned k2,unsigned hidden,unsigned inter,cudaStream_t s) {
  if(!out||!product||!images||!shared.data||m<1||m>4||!supported(k2)||!supported(gate_k2)||!dims(hidden,inter))return cudaErrorInvalidValue;
  #define RUN(K) down<K,2><<<dim3(hidden/128,9),THREADS,0,s>>>((B*)out,(const B*)product,(const void* const*)images,shared,m,gate_k2,hidden,inter)
  EXL3_K2(RUN)
  #undef RUN
  return cudaGetLastError();
}
extern "C" int hayate_glm_grouped_exl3_gate_up(void* product,const void* x,const void* images,const void* ranges,
    unsigned count,unsigned k2,unsigned hidden,unsigned inter,cudaStream_t s) {
  if(!product||!x||!images||!ranges||!count||count>288||!supported(k2)||!dims(hidden,inter))return cudaErrorInvalidValue;
  #define RUN(K) grouped::gemm<K,2,false><<<grouped::BLOCKS,THREADS,0,s>>>((B*)product,(const B*)x,(const void* const*)images,(const unsigned*)ranges,count,k2,hidden,inter)
  EXL3_K2(RUN)
  #undef RUN
  return cudaGetLastError();
}
extern "C" int hayate_glm_grouped_exl3_down(void* out,const void* product,const void* images,const void* ranges,
    unsigned count,unsigned gate_k2,unsigned k2,unsigned hidden,unsigned inter,cudaStream_t s) {
  if(!out||!product||!images||!ranges||!count||count>288||!supported(k2)||!supported(gate_k2)||!dims(hidden,inter))
    return cudaErrorInvalidValue;
  #define RUN(K) grouped::gemm<K,2,true><<<grouped::BLOCKS,THREADS,0,s>>>((B*)out,(const B*)product,(const void* const*)images,(const unsigned*)ranges,count,gate_k2,hidden,inter)
  EXL3_K2(RUN)
  #undef RUN
  return cudaGetLastError();
}
extern "C" int hayate_glm_exl3_restore(void* out,const void* image,unsigned n,unsigned k,unsigned k2,cudaStream_t s) {
  if(!out||!image||!n||!k||n%128||k%128||!supported(k2))return cudaErrorInvalidValue;
  constexpr int shared=128*129*4;
  #define RUN(K) { \
    static const cudaError_t configured=cudaFuncSetAttribute(restore<K,2>,cudaFuncAttributeMaxDynamicSharedMemorySize,shared); \
    if(configured!=cudaSuccess)return configured; \
    restore<K,2><<<dim3(n/128,k/128),THREADS,shared,s>>>((B*)out,(const unsigned char*)image,n,k); }
  EXL3_K2(RUN)
  #undef RUN
  return cudaGetLastError();
}
extern "C" int hayate_glm_exl3_fetch(const void* dst,const void* src,size_t bytes,unsigned blocks,cudaStream_t s) {
  if(!dst||!src||!bytes||bytes%16||!blocks)return cudaErrorInvalidValue;
  fetch<<<dim3(blocks,8),THREADS,0,s>>>((void* const*)dst,(const void* const*)src,bytes);
  return cudaGetLastError();
}
