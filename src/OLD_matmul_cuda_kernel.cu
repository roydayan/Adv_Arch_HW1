#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_DIM 32

__global__ void matmul_kernel_optimized(const float* __restrict__ a1, const float* __restrict__ a2, float* __restrict__ out, int batch_size) {
    int batch = blockIdx.z;
    int row = threadIdx.y;
    int col = threadIdx.x;

    if (batch >= batch_size) return;

    __shared__ float shared_a[TILE_DIM][TILE_DIM];
    __shared__ float shared_b[TILE_DIM][TILE_DIM];

    // Load A and B tiles into shared memory
    shared_a[row][col] = a1[batch * TILE_DIM * TILE_DIM + row * TILE_DIM + col];
    shared_b[row][col] = a2[batch * TILE_DIM * TILE_DIM + row * TILE_DIM + col];

    __syncthreads();

    float sum = 0.0f;

    for (int k = 0; k < TILE_DIM; ++k) {
        sum += shared_a[row][k] * shared_b[k][col];
    }

    out[batch * TILE_DIM * TILE_DIM + row * TILE_DIM + col] = sum;
}

void matmul_cuda_forward(torch::Tensor a1, torch::Tensor a2, torch::Tensor out, int batch_size) {
    const dim3 threads(TILE_DIM, TILE_DIM);
    const dim3 blocks(1, 1, batch_size);

    matmul_kernel_optimized<<<blocks, threads>>>(
        a1.data_ptr<float>(),
        a2.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size
    );
}













// #include <torch/extension.h>
// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <vector>

// #define TILE_SIZE 32

// // Efficient CUDA kernel for batched 32x32 matrix multiplication using shared memory
// template <typename scalar_t>
// __global__ void batched_matmul_kernel(
//     const scalar_t* __restrict__ a1,
//     const scalar_t* __restrict__ a2,
//     scalar_t* __restrict__ out,
//     int batch_size) {
//     // Each block computes one 32x32 matrix multiplication for a batch element
//     int batch = blockIdx.z;
//     int row = threadIdx.y;
//     int col = threadIdx.x;
//     if (batch >= batch_size) return;

//     // Shared memory for A and B tiles
//     __shared__ scalar_t As[TILE_SIZE][TILE_SIZE];
//     __shared__ scalar_t Bs[TILE_SIZE][TILE_SIZE];

//     // Load A and B tiles from global memory to shared memory
//     As[row][col] = a1[batch * 32 * 32 + row * 32 + col];
//     Bs[row][col] = a2[batch * 32 * 32 + row * 32 + col];
//     __syncthreads();

//     // Compute dot product for (row, col)
//     scalar_t sum = 0;
//     for (int k = 0; k < 32; ++k) {
//         sum += As[row][k] * Bs[k][col];
//     }
//     out[batch * 32 * 32 + row * 32 + col] = sum;
// }

// // Kernel launcher

// torch::Tensor matmul_cuda_forward(
//     torch::Tensor a1,
//     torch::Tensor a2) {
//     const int batch_size = a1.size(0);
//     auto out = torch::zeros_like(a1);
//     const dim3 threads(32, 32);
//     const dim3 blocks(1, 1, batch_size);

//     AT_DISPATCH_FLOATING_TYPES(a1.scalar_type(), "batched_matmul_cuda", ([&] {
//         batched_matmul_kernel<scalar_t><<<blocks, threads>>>(
//             a1.data_ptr<scalar_t>(),
//             a2.data_ptr<scalar_t>(),
//             out.data_ptr<scalar_t>(),
//             batch_size);
//     }));
//     return out;
// }
