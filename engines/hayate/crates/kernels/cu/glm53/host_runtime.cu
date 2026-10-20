#include <cuda.h>
#include <cuda_runtime.h>
namespace {
struct Entry {
  void* function=nullptr;
  cudaError_t error;
  Entry(const char* name) {
    cudaDriverEntryPointQueryResult status;
    error=cudaGetDriverEntryPointByVersion(name,&function,13000,cudaEnableDefault,&status);
    if(error==cudaSuccess && (status!=cudaDriverEntryPointSuccess || !function))error=cudaErrorNotSupported;
  }
};
}
extern "C" int hayate_host_wait(cudaStream_t s,void* address,unsigned value) {
  static Entry entry("cuStreamWaitValue32");
  if(entry.error)return entry.error;
  using Function=CUresult(CUDAAPI*)(CUstream,CUdeviceptr,cuuint32_t,unsigned);
  return reinterpret_cast<Function>(entry.function)((CUstream)s,(CUdeviceptr)address,value,CU_STREAM_WAIT_VALUE_EQ);
}
extern "C" int hayate_host_signal(cudaStream_t s,void* address,unsigned value) {
  static Entry entry("cuStreamWriteValue32");
  if(entry.error)return entry.error;
  using Function=CUresult(CUDAAPI*)(CUstream,CUdeviceptr,cuuint32_t,unsigned);
  return reinterpret_cast<Function>(entry.function)((CUstream)s,(CUdeviceptr)address,value,CU_STREAM_WRITE_VALUE_DEFAULT);
}
