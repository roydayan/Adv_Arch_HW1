#include <torch/extension.h>
#include <vector>

// CUDA forward declaration

torch::Tensor matmul_cuda_forward(
    torch::Tensor a1,
    torch::Tensor a2);

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// C++ interface

torch::Tensor matmul_forward(
    torch::Tensor a1,
    torch::Tensor a2) {
  CHECK_INPUT(a1);
  CHECK_INPUT(a2);
  return matmul_cuda_forward(a1, a2);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &matmul_forward, "Batched 32x32 Matrix Multiplication (CUDA)");
}
