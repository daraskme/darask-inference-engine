// Tensor-level experiment bindings for the unmodified upstream EXL3 kernels.
#include <torch/extension.h>
#include "quant/quantize.cuh"
#include "quant/pack.cuh"
#include "quant/reconstruct.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize_tiles", &quantize_tiles);
    m.def("decode", &decode);
    m.def("test_distribution", &test_distribution);
    m.def("pack_trellis", &pack_trellis);
    m.def("unpack_trellis", &unpack_trellis);
    m.def("pack_signs", &pack_signs);
    m.def("reconstruct", &reconstruct);
}
