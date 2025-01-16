#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <sys/time.h>

// CUDA Kernel for matrix-vector multiplication with bias
__global__ void mm_cuda(const float* mat, const float* vec, float* output, int rows, int cols) {
    extern __shared__ float shared_vec[];
    int tx = threadIdx.x;
    int row = blockIdx.x * blockDim.y + threadIdx.y;

    if (threadIdx.y == 0 && tx < cols) {
        shared_vec[tx] = vec[tx];
    }
    __syncthreads();

    if (row < rows) {
        float mul = 0.0f;
        for (int col = tx; col < cols; col += blockDim.x) {
            mul += mat[row * cols + col] * shared_vec[col];
        }

        // Reduction within the warp
        __shared__ float warp_sum[32];
        warp_sum[threadIdx.y] = mul;
        __syncthreads();

        if (threadIdx.y == 0) {
            mul = 0.0f;
            for (int i = 0; i < blockDim.y; ++i) {
                mul += warp_sum[i];
            }
            output[row] = mul;
        }
    }
}

// CUDA Kernel for ReLU activation
__global__ void ReLU_cuda(float* data, const float* bias, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = data[idx] + bias[idx];
        data[idx] = max(0.0f, val);
    }
}

// CUDA Kernel for matrix-vector addition with bias
__global__ void addmm_cuda(const float* mat, const float* vec, const float* bias, float* output, int rows, int cols) {
    extern __shared__ float shared_vec[];
    int tx = threadIdx.x;
    int row = blockIdx.x * blockDim.y + threadIdx.y;

    if (threadIdx.y == 0 && tx < cols) {
        shared_vec[tx] = vec[tx];
    }
    __syncthreads();

    if (row < rows) {
        float sum = bias[row];
        for (int col = tx; col < cols; col += blockDim.x) {
            sum += mat[row * cols + col] * shared_vec[col];
        }

        // Reduction within the warp
        __shared__ float warp_sum[32];
        warp_sum[threadIdx.y] = sum;
        __syncthreads();

        if (threadIdx.y == 0) {
            sum = 0.0f;
            for (int i = 0; i < blockDim.y; ++i) {
                sum += warp_sum[i];
            }
            output[row] = sum;
        }
    }
}

// CUDA Kernel for Softmax computation
__global__ void softmax_1_cuda(float* output, float* softmax_out, int output_size) {
    __shared__ float max_val_shared;
    __shared__ float sum_exp_shared;

    float max_val = -INFINITY;
    float sum_exp = 0.0f;

    for (int i = threadIdx.x; i < output_size; i += blockDim.x) {
        max_val = max(max_val, output[i]);
    }

    atomicMax(&max_val_shared, max_val);
    __syncthreads();

    for (int i = threadIdx.x; i < output_size; i += blockDim.x) {
        sum_exp += expf(output[i] - max_val_shared);
    }

    atomicAdd(&sum_exp_shared, sum_exp);
    __syncthreads();

    for (int i = threadIdx.x; i < output_size; i += blockDim.x) {
        softmax_out[i] = expf(output[i] - max_val_shared) / sum_exp_shared;
    }
}
