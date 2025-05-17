
//~0.061 ms
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_DIM 32
#define THREADS_PER_DIM 16
#define MAT_SIZE 1024
#define BATCH_SIZE 10000
#define TOTAL_BLOCKS 10000

__global__ void matmul_kernel_optimized(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out) {

    int batch_idx = blockIdx.x;;
    int row = threadIdx.y * 2;
    int col = threadIdx.x * 2;    

    int entry_offset = batch_idx * MAT_SIZE + row * TILE_DIM + col;
    __shared__ float shared_a[TILE_DIM][TILE_DIM];
    __shared__ float shared_b[TILE_DIM][TILE_DIM];

    shared_a[row][col] = a[entry_offset];
    shared_a[row][col + 1] = a[entry_offset + 1];
    shared_a[row + 1][col] = a[entry_offset + TILE_DIM];
    shared_a[row + 1][col + 1] = a[entry_offset + TILE_DIM + 1];
    shared_b[row][col] = b[entry_offset];
    shared_b[row][col + 1] = b[entry_offset + 1];
    shared_b[row + 1][col] = b[entry_offset + TILE_DIM];
    shared_b[row + 1][col + 1] = b[entry_offset + TILE_DIM + 1];

    __syncthreads();

    /* for one entry:
    float sum = 0.0f;
    for (int k = 0; k < TILE_DIM; ++k) {
        sum += shared_a[row][k] * shared_b[k][col];
    }
    out[entry_offset] = sum;
    */
    // for 4 entries:

    float c_sum[4] = {0.f, 0.f, 0.f, 0.f};
    float a0;
    float a1;
    float b0;
    float b1;
    #pragma unroll 32
    for (int k = 0; k < TILE_DIM; ++k) {
        a0 = shared_a[row][k];
        a1 = shared_a[row + 1][k];
        b0 = shared_b[k][col];
        b1 = shared_b[k][col + 1];

        c_sum[0] += a0 * b0;
        c_sum[1] += a0 * b1;
        c_sum[2] += a1 * b0;
        c_sum[3] += a1 * b1;
    }

    // write back
    out[entry_offset] = c_sum[0];
    out[entry_offset + 1] = c_sum[1];
    out[entry_offset + TILE_DIM] = c_sum[2];
    out[entry_offset + TILE_DIM + 1] = c_sum[3];
}

void matmul_cuda_forward(torch::Tensor a1, torch::Tensor a2, torch::Tensor out) {
    const dim3 threads(THREADS_PER_DIM, THREADS_PER_DIM);   // 16×16 = 256
    const dim3 blocks(TOTAL_BLOCKS);

    matmul_kernel_optimized<<<blocks, threads>>>(
        a1.data_ptr<float>(),
        a2.data_ptr<float>(),
        out.data_ptr<float>()
    );
}








/* MODULAR VERSION: (less time efficient ~0.063ms)
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

//#include <assert.h> //TODO - remove

#define TILE_DIM 32
#define THREADS_PER_DIM 16
#define MAT_SIZE 1024
#define BATCH_SIZE 10000
#define MAT_PER_BLOCK 16
#define NUM_OF_BLOCKS 625 // (BATCH_SIZE / MAT_PER_BLOCK)
#define PREFETCHED_MATS 4

__global__ void matmul_kernel_optimized(const float* __restrict__ a, 
                                        const float* __restrict__ b, 
                                        float* __restrict__ out) {

    int batch_idx = blockIdx.x * MAT_PER_BLOCK;

    //assert(batch_idx < BATCH_SIZE); // guard for final block // TODO - remove
    //assert(BATCH_SIZE % MAT_PER_BLOCK == 0); 
    //assert(BATCH_SIZE / MAT_PER_BLOCK == NUM_OF_BLOCKS);
    //assert(MAT_PER_BLOCK % PREFETCHED_MATS == 0); // TODO - remove

    int row = threadIdx.y * 2;
    int col = threadIdx.x * 2;    

    __shared__ float shared_a[PREFETCHED_MATS][TILE_DIM][TILE_DIM];
    __shared__ float shared_b[PREFETCHED_MATS][TILE_DIM][TILE_DIM];
    
    
    // Better memory coalescing (threads in a warp access consecutive memory):
    //int start_offset = batch_idx * MAT_SIZE + row * TILE_DIM + col;
    int entry_offset = batch_idx * MAT_SIZE + row * TILE_DIM + col;
    int out_entry = entry_offset; //batch_idx * MAT_SIZE + row * TILE_DIM + col;

    for (int local_m = 0; local_m < MAT_PER_BLOCK; local_m += PREFETCHED_MATS) {
        
        #pragma unroll
        for (int i = 0; i < PREFETCHED_MATS; ++i) {
            
            shared_a[i][row][col] = a[entry_offset];
            shared_a[i][row][col + 1] = a[entry_offset + 1];
            shared_a[i][row + 1][col] = a[entry_offset + TILE_DIM];
            shared_a[i][row + 1][col + 1] = a[entry_offset + TILE_DIM + 1];
            shared_b[i][row][col] = b[entry_offset];
            shared_b[i][row][col + 1] = b[entry_offset + 1];
            shared_b[i][row + 1][col] = b[entry_offset + TILE_DIM];
            shared_b[i][row + 1][col + 1] = b[entry_offset + TILE_DIM + 1];
            
            entry_offset += MAT_SIZE; // move to next matrix prefetch

        }   
    
        __syncthreads();

        for (int i = 0; i < PREFETCHED_MATS; ++i) {
            
            float c_sum[4] = {0.f, 0.f, 0.f, 0.f};  // needs to reset for each matrix
            float a0;
            float a1;
            float b0;
            float b1;
            
            #pragma unroll 32
            for (int k = 0; k < TILE_DIM; ++k) {
                a0 = shared_a[i][row][k];
                a1 = shared_a[i][row + 1][k];
                b0 = shared_b[i][k][col];
                b1 = shared_b[i][k][col + 1];

                c_sum[0] += a0 * b0;
                c_sum[1] += a0 * b1;
                c_sum[2] += a1 * b0;
                c_sum[3] += a1 * b1;
            }

            // write back
            
            out[out_entry] = c_sum[0];
            out[out_entry + 1] = c_sum[1];
            out[out_entry + TILE_DIM] = c_sum[2];
            out[out_entry + TILE_DIM + 1] = c_sum[3];

            out_entry += MAT_SIZE;

        }
        
        //assert(out_entry == entry_offset); // TODO - remove
        
        __syncthreads();

    }

    //assert(out_entry == start_offset + MAT_PER_BLOCK * MAT_SIZE); // TODO - remove
}


void matmul_cuda_forward(torch::Tensor a1, torch::Tensor a2, torch::Tensor out) {
    const dim3 threads(THREADS_PER_DIM, THREADS_PER_DIM);   // 16×16 = 256
    const dim3 blocks(NUM_OF_BLOCKS);

    matmul_kernel_optimized<<<blocks, threads>>>(
        a1.data_ptr<float>(),
        a2.data_ptr<float>(),
        out.data_ptr<float>()
    );
    //for (int i=0; i<500000000;i++){}; // busy wait
}
*/