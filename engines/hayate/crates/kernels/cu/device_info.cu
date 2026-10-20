#include <cuda_runtime_api.h>
#include <cstring>

// Read through the official header rather than duplicating cudaDeviceProp's ABI.
extern "C" int hayate_device_info(int dev, char* name, int name_len,
                                   int* major, int* minor) {
  cudaDeviceProp prop{};
  const cudaError_t error = cudaGetDeviceProperties(&prop, dev);
  if (error != cudaSuccess) return static_cast<int>(error);
  std::strncpy(name, prop.name, name_len - 1);
  name[name_len - 1] = '\0';
  *major = prop.major;
  *minor = prop.minor;
  return static_cast<int>(cudaSuccess);
}
