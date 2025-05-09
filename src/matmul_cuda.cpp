#include <torch/extension.h>
#include <vector>

torch::Tensor matmul_forward(torch::Tensor a1, torch::Tensor a2);

// CUDA function declaration
void matmul_cuda_forward(torch::Tensor a1, torch::Tensor a2, torch::Tensor out);

torch::Tensor matmul_forward(torch::Tensor a1, torch::Tensor a2) {

    auto out = torch::zeros_like(a1);

    matmul_cuda_forward(a1, a2, out);

    return out;
}

// Python binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_forward", &matmul_forward, "Matrix multiplication forward (CUDA)");
}