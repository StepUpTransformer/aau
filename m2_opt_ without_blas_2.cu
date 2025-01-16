#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <sys/time.h>

// CUDA Kernel for matrix-vector multiplication with bias
// Grid size: (5000,1,1), Block size: (32,4,1)
__global__ void mm_cuda(const float* __restrict__ mat, const float* __restrict__ vec, float* __restrict__ output, int rows, int cols) {
    extern __shared__ float shared_vec[];
    int row = (blockIdx.x * blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    if (threadIdx.x < cols) {
        shared_vec[threadIdx.x] = vec[threadIdx.x];
    }
    __syncthreads();

    if (row < rows) {
        float mul = 0.0f;
        for (int col = 0; col < cols; ++col) {
            mul += mat[row * cols + col] * shared_vec[col];
        }
        output[row] = mul;
    }
}

// CUDA Kernel for ReLU activation
// Grid size: (79,1,1), Block size: (128,1,1)
__global__ void ReLU_cuda(float* __restrict__ data, const float* __restrict__ bias, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = data[idx] + bias[idx];
        data[idx] = fmaxf(val, 0.0f);
    }
}

// CUDA Kernel for matrix-vector addition with bias
// Grid size: (25,1,4), Block size: (34,4,1)
__global__ void addmm_cuda(const float* __restrict__ mat, const float* __restrict__ vec, const float* __restrict__ bias, float* __restrict__ output, int rows, int cols) {
    extern __shared__ float shared_data[];
    float* shared_vec = shared_data;
    float* shared_bias = shared_vec + blockDim.x;

    int tid = threadIdx.x;
    if (tid < cols) {
        shared_vec[tid] = vec[tid];
    }
    if (tid < rows) {
        shared_bias[tid] = bias[tid];
    }
    __syncthreads();

    int row = (blockIdx.x * blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    row += blockIdx.z * (gridDim.x * blockDim.x * blockDim.y);

    if (row < rows) {
        float sum = shared_bias[row];
        for (int col = 0; col < cols; ++col) {
            sum += mat[row * cols + col] * shared_vec[col];
        }
        output[row] = sum;
    }
}

// CUDA Kernel for Softmax computation
// Grid size: (1,1,1), Block size: (64,1,1)
__global__ void softmax_1_cuda(float* __restrict__ output, float* __restrict__ softmax_out, int output_size) {
    int idx = threadIdx.x;

    // Compute maximum using warp reduction
    float max_val = -FLT_MAX;
    if (idx < output_size) {
        max_val = output[idx];
    }
    for (int offset = 32 / 2; offset > 0; offset /= 2) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }

    __shared__ float shared_max;
    if (idx == 0) shared_max = max_val;
    __syncthreads();

    // Compute sum of exponentials
    float sum_exp = 0.0f;
    if (idx < output_size) {
        sum_exp = expf(output[idx] - shared_max);
    }
    for (int offset = 32 / 2; offset > 0; offset /= 2) {
        sum_exp += __shfl_down_sync(0xffffffff, sum_exp, offset);
    }

    __shared__ float shared_sum_exp;
    if (idx == 0) shared_sum_exp = sum_exp;
    __syncthreads();

    // Compute softmax
    if (idx < output_size) {
        softmax_out[idx] = expf(output[idx] - shared_max) / shared_sum_exp;
    }
}

// Function to initialize data arrays with specific formulas
void initializeData(float* input, float* W1, float* b1, float* W2, float* b2, int input_size, int hidden_size, int output_size) {
    for (int i = 0; i < input_size; ++i) {
        input[i] = i * 5e-10f;
    }
    for (int i = 0; i < hidden_size; ++i) {
        b1[i] = 0.002;
        for (int j = 0; j < input_size; ++j) {
            W1[i * input_size + j] = i * j * 7e-9f;
        }
    }
    for (int i = 0; i < output_size; ++i) {
        b2[i] = 0.002;
        for (int j = 0; j < hidden_size; ++j) {
            W2[i * hidden_size + j] = i * j * 9e-9f;
        }
    }
}

int main() {
    const int input_size = 10000;
    const int hidden_size = 20000;
    const int output_size = 100;

    // Allocate host memory
    float *h_input = new float[input_size];
    float *h_hidden = new float[hidden_size];
    float *h_output = new float[output_size];
    float *h_linear1_weights = new float[input_size * hidden_size];
    float *h_linear1_bias = new float[hidden_size];
    float *h_linear2_weights = new float[hidden_size * output_size];
    float *h_linear2_bias = new float[output_size];

    initializeData(h_input, h_linear1_weights, h_linear1_bias, h_linear2_weights, h_linear2_bias, 
                  input_size, hidden_size, output_size);

    // Allocate device memory
    float *d_input, *d_hidden, *d_output, *d_softmax_out;
    float *d_linear1_weights, *d_linear1_bias;
    float *d_linear2_weights, *d_linear2_bias;

    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_hidden, hidden_size * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));
    cudaMalloc(&d_linear1_weights, input_size * hidden_size * sizeof(float));
    cudaMalloc(&d_linear1_bias, hidden_size * sizeof(float));
    cudaMalloc(&d_linear2_weights, hidden_size * output_size * sizeof(float));
    cudaMalloc(&d_linear2_bias, output_size * sizeof(float));
    cudaMalloc(&d_softmax_out, output_size * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_linear1_weights, h_linear1_weights, input_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_linear1_bias, h_linear1_bias, hidden_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_linear2_weights, h_linear2_weights, hidden_size * output_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_linear2_bias, h_linear2_bias, output_size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernels with specified configurations
    dim3 mm_gridDim(5000, 1, 1);
    dim3 mm_blockDim(32, 4, 1);
    mm_cuda<<<mm_gridDim, mm_blockDim, input_size * sizeof(float)>>>(d_linear1_weights, d_input, d_hidden, hidden_size, input_size);

    dim3 relu_gridDim(79, 1, 1);
    dim3 relu_blockDim(128, 1, 1);
    ReLU_cuda<<<relu_gridDim, relu_blockDim>>>(d_hidden, d_linear1_bias, hidden_size);

    dim3 addmm_gridDim(25, 1, 4);
    dim3 addmm_blockDim(34, 4, 1);
    addmm_cuda<<<addmm_gridDim, addmm_blockDim, (input_size + hidden_size) * sizeof(float)>>>(d_linear2_weights, d_hidden, d_linear2_bias, d_output, output_size, hidden_size);

    dim3 softmax_gridDim(1, 1, 1);
    dim3 softmax_blockDim(64, 1, 1);
    softmax_1_cuda<<<softmax_gridDim, softmax_blockDim>>>(d_output, d_softmax_out, output_size);

    // Copy results back to host
    cudaMemcpy(h_output, d_softmax_out, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Print results
    std::cout << "\nOutput: " << std::endl;
    for (int i = 0; i < output_size; ++i) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    // Cleanup
    delete[] h_input;
    delete[] h_hidden;
    delete[] h_output;
    delete[] h_linear1_weights;
    delete[] h_linear1_bias;
    delete[] h_linear2_weights;
    delete[] h_linear2_bias;

    cudaFree(d_input);
    cudaFree(d_hidden);
    cudaFree(d_output);
    cudaFree(d_linear1_weights);
    cudaFree(d_linear1_bias);
    cudaFree(d_linear2_weights);
    cudaFree(d_linear2_bias);
    cudaFree(d_softmax_out);

    return 0;
}
