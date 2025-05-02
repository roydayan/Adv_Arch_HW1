#include <torch/extension.h>
#include <vector>

torch::Tensor matmul_forward(torch::Tensor a1, torch::Tensor a2);

// Python binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_forward", &matmul_forward, "Matrix multiplication forward (CUDA)");
}

torch::Tensor matmul_forward(torch::Tensor a1, torch::Tensor a2) {
    TORCH_CHECK(a1.device().is_cuda(), "a1 must be a CUDA tensor");
    TORCH_CHECK(a2.device().is_cuda(), "a2 must be a CUDA tensor");
    TORCH_CHECK(a1.is_contiguous(), "a1 must be contiguous");
    TORCH_CHECK(a2.is_contiguous(), "a2 must be contiguous");

    auto out = torch::zeros_like(a1);

    const int batch_size = a1.size(0);

    matmul_cuda_forward(a1, a2, out, batch_size);

    return out;
}

// CUDA function declaration
void matmul_cuda_forward(torch::Tensor a1, torch::Tensor a2, torch::Tensor out, int batch_size);



// #include <torch/extension.h>
// #include <vector>

// // CUDA forward declaration

// torch::Tensor matmul_cuda_forward(
//     torch::Tensor a1,
//     torch::Tensor a2);

// #define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
// #define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
// #define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// // C++ interface

// torch::Tensor matmul_forward(
//     torch::Tensor a1,
//     torch::Tensor a2) {
//   CHECK_INPUT(a1);
//   CHECK_INPUT(a2);
//   return matmul_cuda_forward(a1, a2);
// }

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//   m.def("forward", &matmul_forward, "Batched 32x32 Matrix Multiplication (CUDA)");
// }
