#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "math.cuh"
#pragma nv_diag_suppress 20013
#pragma nv_diag_suppress 20015
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include <cutlass/gemm/device/gemm_grouped.h>

namespace {
using B=cutlass::bfloat16_t;
template<class O> struct MetadataT {
  cutlass::gemm::GemmCoord sizes[8];
  B *a[8],*b[8]; O *c[8],*d[8];
  int64_t lda[8],ldb[8],ldc[8],ldd[8];
};
using Metadata=MetadataT<B>;
static_assert(sizeof(Metadata)==608);
template<class O> using KernelT=typename cutlass::gemm::kernel::DefaultGemmGrouped<
    B,cutlass::layout::RowMajor,cutlass::ComplexTransform::kNone,8,
    B,cutlass::layout::ColumnMajor,cutlass::ComplexTransform::kNone,8,
    O,cutlass::layout::RowMajor,float,cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,cutlass::gemm::GemmShape<128,128,32>,
    cutlass::gemm::GemmShape<64,64,32>,cutlass::gemm::GemmShape<16,8,16>,
    cutlass::epilogue::thread::LinearCombination<O,8,float,float>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    2,cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly>::GemmKernel;
using Gemm=cutlass::gemm::device::GemmGrouped<KernelT<B>>;
using FloatGemm=cutlass::gemm::device::GemmGrouped<KernelT<float>>;
template<class O> __global__ void describe(MetadataT<O>* meta,O* out,const B* x,const B* w,
    const unsigned* ranges,unsigned groups,unsigned n,unsigned k) {
  unsigned g=threadIdx.x;if(g>=groups)return;
  unsigned rows=ranges[g*2],offset=ranges[g*2+1];
  meta->sizes[g]={int(rows),int(n),int(k)};
  meta->a[g]=const_cast<B*>(x)+size_t(offset)*k;
  meta->b[g]=const_cast<B*>(w)+size_t(g)*n*k;
  meta->c[g]=meta->d[g]=out+size_t(offset)*n;
  meta->lda[g]=meta->ldb[g]=k;meta->ldc[g]=meta->ldd[g]=n;
}
__global__ void restore_nv(B* out,const unsigned char* image,unsigned n,unsigned k,bool scaled=true,float* global_out=nullptr) {
  if(global_out && blockIdx.x==0 && threadIdx.x==0) {
    size_t at=size_t(n)*k/2+size_t((n+127)/128*128)*((k/16+3)/4*4);
    *global_out=*reinterpret_cast<const float*>(image+at);
  }
  size_t i=size_t(blockIdx.x)*256+threadIdx.x;
  if(i<size_t(n)*k)out[i]=B(glm53::nvfp4(image,i/k,i%k,n,k,scaled));
}
__global__ void gather(B* out,const B* x,const unsigned* assignments,unsigned rows,unsigned hidden) {
  size_t i=size_t(blockIdx.x)*256+threadIdx.x;
  if(i<size_t(rows)*hidden)out[i]=x[size_t(assignments[i/hidden]/8)*hidden+i%hidden];
}
// Each expert runs once per token. Ascending expert launches retain BF16
// multiply/index_add rounding without atomics or reassociation.
__global__ void scatter(B* out,const B* x,const float* weights,const unsigned* assignments,unsigned rows,unsigned hidden) {
  size_t i=size_t(blockIdx.x)*256+threadIdx.x;if(i>=size_t(rows)*hidden)return;
  unsigned a=assignments[i/hidden];size_t dst=size_t(a/8)*hidden+i%hidden;
  out[dst]=B(float(out[dst])+glm53::bf16(float(x[i])*weights[a]));
}
}
extern "C" int hayate_glm_grouped(void* out,const void* x,const void* w,const void* ranges,
    unsigned groups,unsigned n,unsigned k,void* metadata,cudaStream_t stream) {
  if(!out||!x||!w||!ranges||!metadata||!groups||groups>8||
     !((n==2048&&k==4096)||(n==4096&&k==2048)))return cudaErrorInvalidValue;
  auto* meta=static_cast<Metadata*>(metadata);
  describe<<<1,8,0,stream>>>(meta,(B*)out,(const B*)x,(const B*)w,(const unsigned*)ranges,groups,n,k);
  auto error=cudaGetLastError();if(error)return error;
  static const int blocks=Gemm::sufficient();if(blocks<=0)return cudaErrorInvalidConfiguration;
  typename Gemm::Arguments args{meta->sizes,int(groups),blocks,{1.f,0.f},
      meta->a,meta->b,meta->c,meta->d,meta->lda,meta->ldb,meta->ldc,meta->ldd};
  Gemm gemm;auto status=gemm(args,nullptr,stream);
  if(status!=cutlass::Status::kSuccess){error=cudaGetLastError();return error?error:cudaErrorInvalidValue;}
  return cudaGetLastError();
}
extern "C" int hayate_glm_nvfp4_restore(void* out,const void* image,unsigned n,unsigned k,cudaStream_t stream) {
  if(!out||!image||!n||!k||k%16)return cudaErrorInvalidValue;
  restore_nv<<<(size_t(n)*k+255)/256,256,0,stream>>>((B*)out,(const unsigned char*)image,n,k);
  return cudaGetLastError();
}
extern "C" int hayate_glm_gather(void* out,const void* x,const void* assignments,unsigned rows,unsigned hidden,cudaStream_t stream) {
  if(!out||!x||!assignments||!rows||!hidden)return cudaErrorInvalidValue;
  gather<<<(size_t(rows)*hidden+255)/256,256,0,stream>>>((B*)out,(const B*)x,(const unsigned*)assignments,rows,hidden);
  return cudaGetLastError();
}
extern "C" int hayate_glm_scatter(void* out,const void* x,const void* weights,const void* assignments,unsigned rows,unsigned hidden,cudaStream_t stream) {
  if(!out||!x||!weights||!assignments||!rows||!hidden)return cudaErrorInvalidValue;
  scatter<<<(size_t(rows)*hidden+255)/256,256,0,stream>>>((B*)out,(const B*)x,(const float*)weights,(const unsigned*)assignments,rows,hidden);
  return cudaGetLastError();
}
extern "C" int hayate_glm_copy(void* out,const void* x,size_t bytes,cudaStream_t stream) {
  if(!out||!bytes)return cudaErrorInvalidValue;
  return x?cudaMemcpyAsync(out,x,bytes,cudaMemcpyDeviceToDevice,stream):cudaMemsetAsync(out,0,bytes,stream);
}

namespace {
__global__ void rescale(B* out,const float* x,const float* scales,const unsigned* ranges,
    unsigned n) {
  unsigned g=blockIdx.y,rows=ranges[g*2],offset=ranges[g*2+1];
  size_t i=size_t(blockIdx.x)*256+threadIdx.x;if(i>=size_t(rows)*n)return;
  float global=scales[g];
  out[size_t(offset)*n+i]=B(x[size_t(offset)*n+i]*global);
}
}
// E2M1 * E4M3 is exactly representable in BF16. Keep the matrix-wide FP32
// multiplier outside GEMM, avoiding BF16 rounding of the scaled NVFP4 weights.
extern "C" int hayate_glm_nvfp4_restore_unscaled(void* out,void* scale,const void* image,unsigned n,unsigned k,cudaStream_t stream) {
  if(!out||!scale||!image||!n||!k||k%16)return cudaErrorInvalidValue;
  restore_nv<<<(size_t(n)*k+255)/256,256,0,stream>>>((B*)out,(const unsigned char*)image,n,k,false,(float*)scale);
  return cudaGetLastError();
}
extern "C" int hayate_glm_grouped_nvfp4(void* out,void* accumulator,const void* x,const void* w,const void* ranges,
    unsigned groups,unsigned n,unsigned k,void* metadata,const void* scales,unsigned max_rows,cudaStream_t stream) {
  if(!out||!accumulator||!x||!w||!ranges||!metadata||!scales||!groups||groups>8||!max_rows||
     !((n==2048&&k==4096)||(n==4096&&k==2048)))return cudaErrorInvalidValue;
  auto* meta=static_cast<MetadataT<float>*>(metadata);
  describe<<<1,8,0,stream>>>(meta,(float*)accumulator,(const B*)x,(const B*)w,(const unsigned*)ranges,groups,n,k);
  auto error=cudaGetLastError();if(error)return error;
  static const int blocks=FloatGemm::sufficient();if(blocks<=0)return cudaErrorInvalidConfiguration;
  typename FloatGemm::Arguments args{meta->sizes,int(groups),blocks,{1.f,0.f},
      meta->a,meta->b,meta->c,meta->d,meta->lda,meta->ldb,meta->ldc,meta->ldd};
  FloatGemm gemm;auto status=gemm(args,nullptr,stream);
  if(status!=cutlass::Status::kSuccess){error=cudaGetLastError();return error?error:cudaErrorInvalidValue;}
  rescale<<<dim3((size_t(max_rows)*n+255)/256,groups),256,0,stream>>>((B*)out,(const float*)accumulator,
      (const float*)scales,(const unsigned*)ranges,n);
  return cudaGetLastError();
}
