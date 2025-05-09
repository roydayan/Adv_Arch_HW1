#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_DIM 32
#define BATCH_SIZE 10000
#define MATRICES_PER_BLOCK 2
#define TOTAL_BLOCKS 5000 // (BATCH_SIZE / MATRICES_PER_BLOCK)

__global__ void matmul_kernel_optimized(const float* __restrict__ a1, const float* __restrict__ a2, float* __restrict__ out) {

    int batch_idx = blockIdx.x * MATRICES_PER_BLOCK;
    int row = threadIdx.y;
    int col = threadIdx.x;    

    // Better memory coalescing (threads in a warp access consecutive memory):
    int matrix_offset = batch_idx * TILE_DIM * TILE_DIM;
    const float* batch_a1 = a1 + matrix_offset;
    const float* batch_a2 = a2 + matrix_offset;
    float* batch_out = out + matrix_offset;
    __shared__ float shared_a[TILE_DIM][TILE_DIM];
    __shared__ float shared_b[TILE_DIM][TILE_DIM];
    shared_a[row][col] = batch_a1[row * TILE_DIM + col];
    shared_b[row][col] = batch_a2[row * TILE_DIM + col];
    __syncthreads();
    float sum = 0.0f;
    //#pragma unroll
    for (int k = 0; k < TILE_DIM; ++k) {
        sum += shared_a[row][k] * shared_b[k][col];
    }
    batch_out[row * TILE_DIM + col] = sum;
    __syncthreads();

    // no need for loop when MATRICES_PER_BLOCK == 2
    batch_idx++; 
    matrix_offset = batch_idx * TILE_DIM * TILE_DIM;
    batch_a1 = a1 + matrix_offset;
    batch_a2 = a2 + matrix_offset;
    batch_out = out + matrix_offset;
    shared_a[row][col] = batch_a1[row * TILE_DIM + col];
    shared_b[row][col] = batch_a2[row * TILE_DIM + col];
    __syncthreads();
    sum = 0.0f;
    //#pragma unroll
    for (int k = 0; k < TILE_DIM; ++k) {
        sum += shared_a[row][k] * shared_b[k][col];
    }
    batch_out[row * TILE_DIM + col] = sum;

    /* no need for loop when MATRICES_PER_BLOCK == 2
    for (int i = 1; i < MATRICES_PER_BLOCK; i++) {
        batch_idx++; 
        matrix_offset = batch_idx * TILE_DIM * TILE_DIM;
        batch_a1 = a1 + matrix_offset;
        batch_a2 = a2 + matrix_offset;
        batch_out = out + matrix_offset;
        shared_a[row][col] = batch_a1[row * TILE_DIM + col];
        shared_b[row][col] = batch_a2[row * TILE_DIM + col];
        __syncthreads();
        sum = 0.0f;
        //#pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            sum += shared_a[row][k] * shared_b[k][col];
        }
        batch_out[row * TILE_DIM + col] = sum;
        __syncthreads();    //not necessary when MATRICES_PER_BLOCK == 2
    }
    */

    
    /* //No memory coalescing (threads in a warp access non-consecutive memory):
    __shared__ float shared_a[TILE_DIM][TILE_DIM];
    __shared__ float shared_b[TILE_DIM][TILE_DIM];


    shared_a[row][col] = a1[batch_idx * TILE_DIM * TILE_DIM + row * TILE_DIM + col];
    shared_b[row][col] = a2[batch_idx * TILE_DIM * TILE_DIM + row * TILE_DIM + col];

    __syncthreads();

    float sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < TILE_DIM; ++k) {
        sum += shared_a[row][k] * shared_b[k][col];
    }

    out[batch_idx * TILE_DIM * TILE_DIM + row * TILE_DIM + col] = sum;
    */
}

void matmul_cuda_forward(torch::Tensor a1, torch::Tensor a2, torch::Tensor out) {
    const dim3 threads(TILE_DIM, TILE_DIM);
    const dim3 blocks(TOTAL_BLOCKS, 1, 1);

    matmul_kernel_optimized<<<blocks, threads>>>(
        a1.data_ptr<float>(),
        a2.data_ptr<float>(),
        out.data_ptr<float>()
    );
}
