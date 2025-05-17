
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

constexpr int TILE              = 32;   // matrix side
constexpr int MAT_SIZE          = 1024;   // 32*32
constexpr int THREADS_PER_DIM   = 16;   // 16×16  = 256 threads/block
constexpr int MAT_PER_BLOCK     = 16;    // 8 matrices per block  ➜ high occupancy   //default=8
constexpr int BATCH_SIZE        = 10000;
constexpr int NUM_OF_BLOCKS     = 625; // BATCH_SIZE / MAT_PER_BLOCK                 //default=1250
constexpr int PREFETCHED_MATS   = 4;    // matrices calculated at once

/// Each thread computes a 2×2 patch => 4 results (= higher arithmetic per load) for PPREFETCHED_MATS matrices
__global__ void matmul_kernel_fast(const float *__restrict__ A,
                                   const float *__restrict__ B,
                                   float       *__restrict__ C) {
    
    __shared__ float As[PREFETCHED_MATS][TILE][TILE];
    __shared__ float Bs[PREFETCHED_MATS][TILE][TILE];

    const int tx = threadIdx.x;        // 0…15
    const int ty = threadIdx.y;        // 0…15

    // First matrix handled by this block
    const int block_batch_base = blockIdx.x * MAT_PER_BLOCK;
    
    int mat_base[PREFETCHED_MATS];
    // Handle 0 … MAT_PER_BLOCK-1 matrices in a loop to reuse the same block
    
    if (block_batch_base >= BATCH_SIZE) return; // guard for final block

    for (int local_m = 0; local_m < MAT_PER_BLOCK; local_m += PREFETCHED_MATS) {

        int batch_idx = block_batch_base + local_m;

        //if (batch_idx >= BATCH_SIZE) break;        // guard for final block
       
        int temp_offset = batch_idx * MAT_SIZE;     // flat offset for this matrix
        
        #pragma unroll
        for (int i = 0; i < PREFETCHED_MATS; ++i) {
            mat_base[i] = temp_offset;
            temp_offset += MAT_SIZE;
        }

        // -------- 1. load 32×32 tiles of A and B to shared memory -------- 
        const int row0 = ty * 2;    // *2 b/c 16x16 threads, 32x32 tiles
        const int col0 = tx * 2;

        #pragma unroll
        for (int i = 0; i < PREFETCHED_MATS; ++i) { // for each matrix, each thread loads 4 entries (16x16 thread blocks, 32x32 matrices)

            const int entry_idx = mat_base[i] + ty * TILE + tx;   // entry in global memory

            As[i][row0][col0] = A[entry_idx];
            As[i][row0][col0 + 1] = A[entry_idx + 1];
            As[i][row0 + 1][col0] = A[entry_idx + TILE];
            As[i][row0 + 1][col0 + 1] = A[entry_idx + TILE + 1];
            Bs[i][row0][col0] = B[entry_idx];
            Bs[i][row0][col0 + 1] = B[entry_idx + 1];
            Bs[i][row0 + 1][col0] = B[entry_idx + TILE];
            Bs[i][row0 + 1][col0 + 1] = B[entry_idx + TILE + 1];
        }

        __syncthreads();   // wait for all threads to load


        // -------- 2. compute a 2×2 output patch in registers --------        
        
        for (int i = 0; i < PREFETCHED_MATS; ++i) {
            
            float c_sum[4] = {0.f, 0.f, 0.f, 0.f};
            float a0;
            float a1;
            float b0;
            float b1;
                        
            #pragma unroll 32
            for (int k = 0; k < TILE; ++k) {
                a0 = As[i][row0    ][k];
                a1 = As[i][row0 + 1][k];
                b0 = Bs[i][k][col0    ];
                b1 = Bs[i][k][col0 + 1];

                c_sum[0] += a0 * b0;
                c_sum[1] += a0 * b1;
                c_sum[2] += a1 * b0;
                c_sum[3] += a1 * b1;
            }
            // -------- 3. write back -------- 

            //int out_base = mat_base[i];
            int out_entry = mat_base[i] + row0 * TILE + col0;
            C[out_entry] = c_sum[0];
            C[out_entry + 1] = c_sum[1];
            C[out_entry + TILE] = c_sum[2];
            C[out_entry + TILE + 1] = c_sum[3];
        }

        local_m += PREFETCHED_MATS;   // next matrices for this thread block
        
        __syncthreads();   // recycle shared memory for next matrix
    }
}

void matmul_cuda_forward(torch::Tensor a1,
                         torch::Tensor a2,
                         torch::Tensor out)
{
    const dim3 threads(THREADS_PER_DIM, THREADS_PER_DIM);   // 16×16 = 256
    const dim3 blocks(NUM_OF_BLOCKS);

    // launch on the current PyTorch stream to avoid synchronisation stalls
    matmul_kernel_fast<<<blocks, threads>>>(
        a1.data_ptr<float>(),
        a2.data_ptr<float>(),
        out.data_ptr<float>()
    );
}



















/*
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

constexpr int TILE              = 32;   // matrix side
constexpr int MAT_SIZE          = 1024;   // 32*32
constexpr int THREADS_PER_DIM   = 16;   // 16×16  = 256 threads/block
constexpr int MAT_PER_BLOCK     = 16;    // 8 matrices per block  ➜ high occupancy   //default=8
constexpr int BATCH_SIZE        = 10000;
constexpr int NUM_OF_BLOCKS     = 625; // BATCH_SIZE / MAT_PER_BLOCK                 //default=1250
constexpr int PREFETCHED_MATS   = 4;    // matrices calculated at once

/// Each thread computes a 2×2 patch => 4 results (= higher arithmetic per load) for PPREFETCHED_MATS matrices
__global__ void matmul_kernel_fast(const float *__restrict__ A,
                                   const float *__restrict__ B,
                                   float       *__restrict__ C) {
    __shared__ float As[PREFETCHED_MATS][TILE][TILE];
    __shared__ float Bs[PREFETCHED_MATS][TILE][TILE];

    const int tx = threadIdx.x;        // 0…15
    const int ty = threadIdx.y;        // 0…15

    // First matrix handled by this block
    const int block_batch_base = blockIdx.x * MAT_PER_BLOCK;
    int mat_base[PREFETCHED_MATS];
    // Handle 0 … MAT_PER_BLOCK-1 matrices in a loop to reuse the same block
    
    if (block_batch_base >= BATCH_SIZE) return; // guard for final block

    for (int local_m = 0; local_m < MAT_PER_BLOCK; local_m += PREFETCHED_MATS) {

        int batch_idx = block_batch_base + local_m;

        //if (batch_idx >= BATCH_SIZE) break;        // guard for final block
       
        int temp_offset = batch_idx * MAT_SIZE;     // flat offset for this matrix
        
        #pragma unroll
        for (int i = 0; i < PREFETCHED_MATS; ++i) {
            mat_base[i] = temp_offset;
            temp_offset += MAT_SIZE;
        }

        //const int mat_base[] = batch_idx * MAT_SIZE;     // flat offset for this matrix
        //const int mat2_base = mat1_base + MAT_SIZE;     // flat offset for second matrix

        //-------- 1. load 32×32 tiles of A and B to shared memory -------- 
        const int row0 = ty * 2;
        const int col0 = tx * 2;

        #pragma unroll
        for (int i = 0; i < PREFETCHED_MATS; ++i) { // for each matrix, each thread loads 4 entries (16x16 thread blocks, 32x32 matrices)

            const int entry_idx = mat_base[i] + ty * TILE + tx;   // entry in global memory

            As[i][row0][col0] = A[entry_idx];
            As[i][row0][col0 + 1] = A[entry_idx + 1];
            Bs[i][row0][col0] = B[entry_idx];
            Bs[i][row0][col0 + 1] = B[entry_idx + 1];
            As[i][row0 + 1][col0] = A[entry_idx + TILE];
            As[i][row0 + 1][col0 + 1] = A[entry_idx + TILE + 1];
            Bs[i][row0 + 1][col0] = B[entry_idx + TILE];
            Bs[i][row0 + 1][col0 + 1] = B[entry_idx + TILE + 1];
        }

        __syncthreads();   // wait for all threads to load


        //-------- 2. compute a 2×2 output patch in registers --------            
        
        for (int i = 0; i < PREFETCHED_MATS; ++i) {
            
            float c_sum[4] = {0.f, 0.f, 0.f, 0.f};
            float a0;
            float a1;
            float b0;
            float b1;
                        
            #pragma unroll 32
            for (int k = 0; k < TILE; ++k) {
                a0 = As[i][row0    ][k];
                a1 = As[i][row0 + 1][k];
                b0 = Bs[i][k][col0    ];
                b1 = Bs[i][k][col0 + 1];

                c_sum[0] += a0 * b0;
                c_sum[1] += a0 * b1;
                c_sum[2] += a1 * b0;
                c_sum[3] += a1 * b1;
            }
            //-------- 3. write back -------- 

            int out_base = mat_base[i];
            int out_entry = mat_base[i] + row0 * TILE + col0;
            C[out_entry] = c_sum[0];
            C[out_entry + 1] = c_sum[1];
            C[out_entry + TILE] = c_sum[2];
            C[out_entry + TILE + 1] = c_sum[3];
        }

        local_m += PREFETCHED_MATS;   // next matrices for this thread block
        
        __syncthreads();   // recycle shared memory for next matrix
    }
}

void matmul_cuda_forward(torch::Tensor a1,
                         torch::Tensor a2,
                         torch::Tensor out)
{
    const dim3 threads(THREADS_PER_DIM, THREADS_PER_DIM);   // 16×16 = 256
    const dim3 blocks(NUM_OF_BLOCKS);

    // launch on the current PyTorch stream to avoid synchronisation stalls
    matmul_kernel_fast<<<blocks, threads>>>(
        a1.data_ptr<float>(),
        a2.data_ptr<float>(),
        out.data_ptr<float>()
    );
}
*/






/*
#define TILE_DIM 32
#define BATCH_SIZE 10000
#define MATRICES_PER_BLOCK 1
#define TOTAL_BLOCKS 10000 // (BATCH_SIZE / MATRICES_PER_BLOCK)

__global__ void matmul_kernel_optimized(const float* __restrict__ a1, const float* __restrict__ a2, float* __restrict__ out) {

    int batch_idx = blockIdx.x;// * MATRICES_PER_BLOCK;
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
    #pragma unroll
    for (int k = 0; k < TILE_DIM; ++k) {
        sum += shared_a[row][k] * shared_b[k][col];
    }
    batch_out[row * TILE_DIM + col] = sum;

    //__syncthreads();
    
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
    */ 
    /* // no need for loop when MATRICES_PER_BLOCK == 2
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

    
    /*  //No memory coalescing (threads in a warp access non-consecutive memory):
    __shared__ float shared_a[TILE_DIM][TILE_DIM];
    __shared__ float shared_b[TILE_DIM][TILE_DIM];


    shared_a[row][col] = a1[batch_idx * TILE_DIM * TILE_DIM + row * TILE_DIM + col];
    shared_b[row][col] = a2[batch_idx * TILE_DIM * TILE_DIM + row * TILE_DIM + col];

    __syncthreads();

    float sum = 0.0f;
    //#pragma unroll
    for (int k = 0; k < TILE_DIM; ++k) {
        sum += shared_a[row][k] * shared_b[k][col];
    }

    out[batch_idx * TILE_DIM * TILE_DIM + row * TILE_DIM + col] = sum;
    
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
*/

















/*
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

*/