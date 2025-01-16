#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <cfloat>
#include <sys/time.h>

// Optimized kernel using shared memory and loop unrolling
__global__ void fused_relu_0_cuda(const float* __restrict__ mat, 
                                 const float* __restrict__ vec, 
                                 const float* __restrict__ bias, 
                                 float* __restrict__ output, 
                                 const int rows, 
                                 const int cols) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Shared memory for vector data
    extern __shared__ float shared_vec[];
    
    // Collaborative loading of vector into shared memory
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        shared_vec[i] = vec[i];
    }
    __syncthreads();
    
    if (row < rows) {
        float sum = bias[row];
        const float* row_ptr = mat + row * cols;
        
        // Manual loop unrolling for better instruction-level parallelism
        #pragma unroll 4
        for (int col = 0; col < (cols / 4) * 4; col += 4) {
            sum += row_ptr[col] * shared_vec[col]
                 + row_ptr[col + 1] * shared_vec[col + 1]
                 + row_ptr[col + 2] * shared_vec[col + 2]
                 + row_ptr[col + 3] * shared_vec[col + 3];
        }
        
        // Handle remaining elements
        for (int col = (cols / 4) * 4; col < cols; ++col) {
            sum += row_ptr[col] * shared_vec[col];
        }
        
        output[row] = max(sum, 0.0f); // ReLU activation using intrinsic
    }
}

// Optimized matrix-vector multiplication with bias
__global__ void addmm_1_cuda(const float* __restrict__ mat,
                            const float* __restrict__ vec,
                            const float* __restrict__ bias,
                            float* __restrict__ output,
                            const int rows,
                            const int cols) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Shared memory for vector data
    extern __shared__ float shared_vec[];
    
    // Collaborative loading of vector into shared memory
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        shared_vec[i] = vec[i];
    }
    __syncthreads();
    
    if (row < rows) {
        float sum = bias[row];
        const float* row_ptr = mat + row * cols;
        
        // Manual loop unrolling
        #pragma unroll 4
        for (int col = 0; col < (cols / 4) * 4; col += 4) {
            sum += row_ptr[col] * shared_vec[col]
                 + row_ptr[col + 1] * shared_vec[col + 1]
                 + row_ptr[col + 2] * shared_vec[col + 2]
                 + row_ptr[col + 3] * shared_vec[col + 3];
        }
        
        // Handle remaining elements
        for (int col = (cols / 4) * 4; col < cols; ++col) {
            sum += row_ptr[col] * shared_vec[col];
        }
        
        output[row] = sum;
    }
}

// Optimized softmax implementation using warp-level primitives
__global__ void softmax(float* __restrict__ input,
                       float* __restrict__ output,
                       const int N) {
    // Use warp shuffle operations for better performance
    const int warp_size = 32;
    const int lid = threadIdx.x % warp_size;
    const int wid = threadIdx.x / warp_size;
    const int warps_per_block = blockDim.x / warp_size;
    const int num_warps = warps_per_block * gridDim.x;
    const int warp_id = blockIdx.x * warps_per_block + wid;
    
    // Shared memory for partial results
    __shared__ float shared_max[32];
    __shared__ float shared_sum[32];
    
    // Find max value
    float local_max = -FLT_MAX;
    for (int i = warp_id; i < N; i += num_warps) {
        local_max = max(local_max, input[i]);
    }
    
    // Warp reduction for maximum
    #pragma unroll
    for (int offset = warp_size/2; offset > 0; offset /= 2) {
        local_max = max(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }
    
    if (lid == 0) shared_max[wid] = local_max;
    __syncthreads();
    
    // Block reduction for maximum
    if (lid < warps_per_block) {
        local_max = shared_max[lid];
    }
    if (wid == 0) {
        #pragma unroll
        for (int offset = warps_per_block/2; offset > 0; offset /= 2) {
            local_max = max(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
        }
    }
    
    // Broadcast max to all threads
    if (wid == 0 && lid == 0) {
        shared_max[0] = local_max;
    }
    __syncthreads();
    local_max = shared_max[0];
    
    // Compute exp and sum
    float local_sum = 0.0f;
    for (int i = warp_id; i < N; i += num_warps) {
        float val = expf(input[i] - local_max);
        local_sum += val;
        if (blockIdx.x == 0) {
            output[i] = val;  // Store exp values temporarily
        }
    }
    
    // Warp reduction for sum
    #pragma unroll
    for (int offset = warp_size/2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    
    if (lid == 0) shared_sum[wid] = local_sum;
    __syncthreads();
    
    // Block reduction for sum
    if (lid < warps_per_block) {
        local_sum = shared_sum[lid];
    }
    if (wid == 0) {
        #pragma unroll
        for (int offset = warps_per_block/2; offset > 0; offset /= 2) {
            local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        }
    }
    
    // Broadcast sum to all threads
    if (wid == 0 && lid == 0) {
        shared_sum[0] = local_sum;
    }
    __syncthreads();
    local_sum = shared_sum[0];
    
    // Normalize
    for (int i = warp_id; i < N; i += num_warps) {
        output[i] /= local_sum;  // Divide stored exp values by sum
    }
}

int main() {
    // Constants
    const int input_size = 10000;
    const int hidden_size = 20000;
    const int output_size = 100;
    
    // Use cudaStream_t for asynchronous operations
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // Allocate page-locked memory for better transfer speeds
    float *h_input, *h_hidden, *h_output;
    float *h_linear1_weights, *h_linear1_bias;
    float *h_linear2_weights, *h_linear2_bias;
    
    cudaMallocHost(&h_input, input_size * sizeof(float));
    cudaMallocHost(&h_hidden, hidden_size * sizeof(float));
    cudaMallocHost(&h_output, output_size * sizeof(float));
    cudaMallocHost(&h_linear1_weights, input_size * hidden_size * sizeof(float));
    cudaMallocHost(&h_linear1_bias, hidden_size * sizeof(float));
    cudaMallocHost(&h_linear2_weights, hidden_size * output_size * sizeof(float));
    cudaMallocHost(&h_linear2_bias, output_size * sizeof(float));
    
    // Initialize data
    initializeData(h_input, h_linear1_weights, h_linear1_bias, 
                  h_linear2_weights, h_linear2_bias, 
                  input_size, hidden_size, output_size);
    
    // Device memory allocation
    float *d_input, *d_hidden, *d_output, *d_softmax_out;
    float *d_linear1_weights, *d_linear1_bias;
    float *d_linear2_weights, *d_linear2_bias;
    
    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_hidden, hidden_size * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));
    cudaMalloc(&d_softmax_out, output_size * sizeof(float));
    cudaMalloc(&d_linear1_weights, input_size * hidden_size * sizeof(float));
    cudaMalloc(&d_linear1_bias, hidden_size * sizeof(float));
    cudaMalloc(&d_linear2_weights, hidden_size * output_size * sizeof(float));
    cudaMalloc(&d_linear2_bias, output_size * sizeof(float));
    
    // Asynchronous memory transfers
    cudaMemcpyAsync(d_input, h_input, input_size * sizeof(float),
                   cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_linear1_weights, h_linear1_weights,
                   input_size * hidden_size * sizeof(float),
                   cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_linear1_bias, h_linear1_bias,
                   hidden_size * sizeof(float),
                   cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_linear2_weights, h_linear2_weights,
                   hidden_size * output_size * sizeof(float),
                   cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_linear2_bias, h_linear2_bias,
                   output_size * sizeof(float),
                   cudaMemcpyHostToDevice, stream);
    
    // Kernel configurations
    const int BLOCK_SIZE = 256;
    dim3 fused_relu_block(BLOCK_SIZE);
    dim3 fused_relu_grid((hidden_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    dim3 addmm_block(BLOCK_SIZE);
    dim3 addmm_grid((output_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    dim3 softmax_block(256);
    dim3 softmax_grid(1);
    
    // Launch kernels with shared memory
    fused_relu_0_cuda<<<fused_relu_grid, fused_relu_block, 
                       input_size * sizeof(float), stream>>>
        (d_linear1_weights, d_input, d_linear1_bias, 
         d_hidden, hidden_size, input_size);
    
    addmm_1_cuda<<<addmm_grid, addmm_block,
                   hidden_size * sizeof(float), stream>>>
        (d_linear2_weights, d_hidden, d_linear2_bias,
         d_output, output_size, hidden_size);
    
    softmax<<<softmax_grid, softmax_block, 0, stream>>>
        (d_output, d_softmax_out, output_size);
    
    // Asynchronous copy back to host
    cudaMemcpyAsync(h_output, d_softmax_out,
                   output_size * sizeof(float),
                   cudaMemcpyDeviceToHost, stream);
    
    // Synchronize and check for errors
    cudaStreamSynchronize(stream);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }
    
    // Output results
    std::cout << "\nOutput: " << std::endl;
    for (int i = 0; i < output_size; ++i) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;
    
    // Cleanup
    cudaFreeHost(h_input);
    cudaFreeHost(h_hidden);
    cudaFreeHost(h_output);
    cudaFreeHost(h_linear1_weights);
    cudaFreeHost(h_linear1_bias);
    cudaFreeHost(h_linear2_weights);
    cudaFreeHost(h_linear2_bias);
    
    cudaFree(d_input);
    cudaFree(d_hidden);
    cudaFree(d_output);
    cudaFree(d_softmax_out);
    cudaFree(d_linear1_weights);
    cudaFree(d_linear1_bias);
    cudaFree(d_linear2_weights);
    cudaFree(d_linear2_bias);
    
    cudaStreamDestroy(stream);
    
    return 0;
}
