#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

constexpr int TILE              = 32;   // matrix side
constexpr int MAT_SIZE          = 1024;   // 32*32
constexpr int THREADS_PER_DIM   = 16;   // 16×16  = 256 threads/block
constexpr int MAT_PER_BLOCK     = 8;    // 8 matrices per block  ➜ high occupancy
constexpr int BATCH_SIZE        = 10000;
constexpr int NUM_OF_BLOCKS     = 1250; // 10000 / 8 = 1250 blocks
constexpr int PREFETCHED_MATS   = 2;    // 2 matrices calculated at once

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
    
    if (block_batch_base >= BATCH_SIZE) return;
    
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

        /* -------- 1. load 32×32 tiles of A and B to shared memory -------- */
        const int row0 = ty * 2;
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


        /* -------- 2. compute a 2×2 output patch in registers -------- */
        float c_sum[PREFETCHED_MATS][4] = {0.f}; // 2x2 patch for each matrix
        

        for (int i = 0; i < PREFETCHED_MATS; ++i) {
            
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

                c_sum[i][0] += a0 * b0;
                c_sum[i][1] += a0 * b1;
                c_sum[i][2] += a1 * b0;
                c_sum[i][3] += a1 * b1;
            }
            /* -------- 3. write back -------- */
            C[mat_base[i] +  row0      * TILE +  col0    ] = c_sum[i][0];
            C[mat_base[i] +  row0      * TILE + (col0+1) ] = c_sum[i][1];
            C[mat_base[i] + (row0 + 1) * TILE +  col0    ] = c_sum[i][2];
            C[mat_base[i] + (row0 + 1) * TILE + (col0+1) ] = c_sum[i][3];
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
    //const int  grid_x = (BATCH_SIZE + MAT_PER_BLOCK - 1) / MAT_PER_BLOCK;
    const dim3 blocks(NUM_OF_BLOCKS);

    // launch on the current PyTorch stream to avoid synchronisation stalls
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    matmul_kernel_fast<<<blocks, threads, 0, stream>>>(
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
constexpr int MAT_PER_BLOCK     = 8;    // 8 matrices per block  ➜ high occupancy
constexpr int BATCH_SIZE        = 10000;
constexpr int NUM_OF_BLOCKS     = 1250; // 10000 / 8 = 1250 blocks
constexpr int PREFETCHED_MATS   = 2;    // 2 matrices calculated at once

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
    #pragma unroll
    for (int local_m = 0; local_m < MAT_PER_BLOCK; local_m += PREFETCHED_MATS) {

        int batch_idx = block_batch_base + local_m;

        if (batch_idx >= BATCH_SIZE) break;        // guard for final block
       
        mat_base[0] = batch_idx * MAT_SIZE;     // flat offset for this matrix
        for (int i = 1; i < PREFETCHED_MATS; ++i) {
            mat_base[i] = mat_base[i-1] + MAT_SIZE;
        }

        //const int mat_base[] = batch_idx * MAT_SIZE;     // flat offset for this matrix
        //const int mat2_base = mat1_base + MAT_SIZE;     // flat offset for second matrix

        
        const int row0 = ty * 2;
        const int col0 = tx * 2;

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


        
        float c_sum[PREFETCHED_MATS][4] = {0.f}; // 2x2 patch for each matrix

        #pragma unroll
        for (int i = 0; i < PREFETCHED_MATS; ++i) {
            #pragma unroll 32
            for (int k = 0; k < TILE; ++k) {
                float a0 = As[i][row0    ][k];
                float a1 = As[i][row0 + 1][k];
                float b0 = Bs[i][k][col0    ];
                float b1 = Bs[i][k][col0 + 1];

                c_sum[i][0] += a0 * b0;
                c_sum[i][1] += a0 * b1;
                c_sum[i][2] += a1 * b0;
                c_sum[i][3] += a1 * b1;
            }

            

            C[mat_base[i] +  row0      *TILE +  col0     ] = c_sum[i][0];
            C[mat_base[i] +  row0      *TILE + (col0+1) ] = c_sum[i][1];
            C[mat_base[i] + (row0 + 1)*TILE +  col0     ] = c_sum[i][2];
            C[mat_base[i] + (row0 + 1)*TILE + (col0+1) ] = c_sum[i][3];
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
    //const int  grid_x = (BATCH_SIZE + MAT_PER_BLOCK - 1) / MAT_PER_BLOCK;
    const dim3 blocks(NUM_OF_BLOCKS);

    // launch on the current PyTorch stream to avoid synchronisation stalls
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    matmul_kernel_fast<<<blocks, threads, 0, stream>>>(
        a1.data_ptr<float>(),
        a2.data_ptr<float>(),
        out.data_ptr<float>()
    );
}
*/








/* 0.085 ms
constexpr int TILE              = 32;   // matrix side
constexpr int MAT_SIZE          = 1024;   // 32*32
constexpr int THREADS_PER_DIM   = 16;   // 16×16  = 256 threads/block
constexpr int MAT_PER_BLOCK     = 8;    // 8 matrices per block  ➜ high occupancy
constexpr int BATCH_SIZE        = 10000;
constexpr int NUM_OF_BLOCKS  = 1250; 
constexpr int PREFETCHED_MATS    = 2;    // 2 matrices calculated at once

/// Each thread computes a 2×2 patch => 4 results (= higher arithmetic per load)
__global__ void matmul_kernel_fast(const float *__restrict__ A,
                                   const float *__restrict__ B,
                                   float       *__restrict__ C) {
    __shared__ float As1[TILE][TILE];
    __shared__ float Bs1[TILE][TILE];
    __shared__ float As2[TILE][TILE];
    __shared__ float Bs2[TILE][TILE];

    const int tx = threadIdx.x;        // 0…15
    const int ty = threadIdx.y;        // 0…15

    // First matrix handled by this block
    const int block_batch_base = blockIdx.x * MAT_PER_BLOCK;

    // Handle 0 … MAT_PER_BLOCK-1 matrices in a loop to reuse the same block
    int local_m = 0;
    #pragma unroll
    while (local_m < MAT_PER_BLOCK) {
        int batch_idx = block_batch_base + local_m;
        if (batch_idx >= BATCH_SIZE) break;        // guard for final block

        const int mat1_base = batch_idx * MAT_SIZE;     // flat offset for this matrix
        const int mat2_base = mat1_base + MAT_SIZE;     // flat offset for second matrix

       
        const int entry1 = mat1_base + ty * TILE + tx;   // entry in global memory
        const int entry2 = entry1 + MAT_SIZE;   // entry in global memory for second matrix
        As1[ty][tx] = A[entry1];
        Bs1[ty][tx] = B[entry1];
        As2[ty][tx] = A[entry2];
        Bs2[ty][tx] = B[entry2];
        __syncthreads();   // wait for all threads to load


        
        const int row0 = ty * 2;
        const int col0 = tx * 2;

        float c00 = 0.f, c01 = 0.f, c10 = 0.f, c11 = 0.f;
        float c2_00 = 0.f, c2_01 = 0.f, c2_10 = 0.f, c2_11 = 0.f;

        #pragma unroll 32
        for (int k = 0; k < TILE; ++k) {
            float a0 = As1[row0    ][k];
            float a1 = As1[row0 + 1][k];
            float b0 = Bs1[k][col0    ];
            float b1 = Bs1[k][col0 + 1];

            c00 += a0 * b0;
            c01 += a0 * b1;
            c10 += a1 * b0;
            c11 += a1 * b1;
        }
        
        C[mat1_base +  row0      *TILE +  col0     ] = c00;
        C[mat1_base +  row0      *TILE + (col0+1) ] = c01;
        C[mat1_base + (row0 + 1)*TILE +  col0     ] = c10;
        C[mat1_base + (row0 + 1)*TILE + (col0+1) ] = c11;



        #pragma unroll 32
        for (int k = 0; k < TILE; ++k) {
            float a0 = As2[row0    ][k];
            float a1 = As2[row0 + 1][k];
            float b0 = Bs2[k][col0    ];
            float b1 = Bs2[k][col0 + 1];

            c2_00 += a0 * b0;
            c2_01 += a0 * b1;
            c2_10 += a1 * b0;
            c2_11 += a1 * b1;
        }

       
        
        C[mat2_base +  row0      *TILE +  col0     ] = c2_00;
        C[mat2_base +  row0      *TILE + (col0+1) ] = c2_01;
        C[mat2_base + (row0 + 1)*TILE +  col0     ] = c2_10;
        C[mat2_base + (row0 + 1)*TILE + (col0+1) ] = c2_11;

        local_m += PREFETCHED_MATS;   // next matrices for this thread block

        __syncthreads();   // recycle shared memory for next matrix
    }
}
*/



/* 0.09 ms
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_DIM 32
#define BATCH_SIZE 10000
#define MATRICES_PER_BLOCK 1
#define TOTAL_BLOCKS 10000 // (BATCH_SIZE / MATRICES_PER_BLOCK)

__global__ void matmul_kernel_optimized(const float* __restrict__ a1, const float* __restrict__ a2, float* __restrict__ out) {

    int batch_idx = blockIdx.x;;
    int row = threadIdx.y;
    int col = threadIdx.x;    

    // Better memory coalescing (threads in a warp access consecutive memory):
    int entry_offset = batch_idx * TILE_DIM * TILE_DIM + row * TILE_DIM + col;
    __shared__ float shared_a[TILE_DIM][TILE_DIM];
    __shared__ float shared_b[TILE_DIM][TILE_DIM];
    shared_a[row][col] = a1[entry_offset];
    shared_b[row][col] = a2[entry_offset];
    __syncthreads();
    float sum = 0.0f;
    for (int k = 0; k < TILE_DIM; ++k) {
        sum += shared_a[row][k] * shared_b[k][col];
    }
    out[entry_offset] = sum;
}

void matmul_cuda_forward(torch::Tensor a1, torch::Tensor a2, torch::Tensor out) {
    const dim3 threads(TILE_DIM, TILE_DIM);
    const dim3 blocks(TOTAL_BLOCKS);

    matmul_kernel_optimized<<<blocks, threads>>>(
        a1.data_ptr<float>(),
        a2.data_ptr<float>(),
        out.data_ptr<float>()
    );
}
*/