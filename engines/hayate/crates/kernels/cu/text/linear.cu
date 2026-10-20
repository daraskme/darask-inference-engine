// Existing CUTLASS BF16 tensor-core GEMM, with the text FFN's K=25600.
#include <cuda_runtime.h>
#pragma nv_diag_suppress 20013
#pragma nv_diag_suppress 20015
#include <cutlass/gemm/device/gemm.h>
extern "C" int hayate_text_linear(void* out, const void* x, const void* w,
                                   unsigned m, unsigned n, unsigned k, cudaStream_t stream) {
  using B = cutlass::bfloat16_t;
  using Gemm = cutlass::gemm::device::Gemm<
      B, cutlass::layout::RowMajor, B, cutlass::layout::ColumnMajor,
      B, cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<128,128,32>,
      cutlass::gemm::GemmShape<64,64,32>, cutlass::gemm::GemmShape<16,8,16>>;
  typename Gemm::Arguments args{{int(m),int(n),int(k)},
      {static_cast<const B*>(x),k}, {static_cast<const B*>(w),k},
      {static_cast<const B*>(out),n}, {static_cast<B*>(out),n}, {1.f,0.f}};
  Gemm gemm;
  if (gemm.can_implement(args) != cutlass::Status::kSuccess) return cudaErrorInvalidValue;
  const auto status = gemm(args, nullptr, stream);
  if (status != cutlass::Status::kSuccess) {
    auto error = cudaGetLastError();
    return error == cudaSuccess ? cudaErrorInvalidValue : error;
  }
  return cudaGetLastError();
}
