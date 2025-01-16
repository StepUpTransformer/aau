#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <cfloat>
#include <sys/time.h>

// CUDA Kernel for fused matrix-vector multiplication with bias and ReLU activation
__global__ void fused_relu_0_cuda(const float* __restrict__ mat, const float* __restrict__ vec, const float* __restrict__ bias, float* output, int rows, int cols) {
    extern __shared__ float shared_vec[];

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load vector into shared memory
    for (int i = tid; i < cols; i += blockDim.x) {
        shared_vec[i] = vec[i];
    }
    __syncthreads();

    if (row < rows) {
        float mul = bias[row];
        for (int col = 0; col < cols; ++col) {
            mul += mat[row * cols + col] * shared_vec[col];
        }
        output[row] = fmaxf(mul, 0.0f); // Apply ReLU activation
    }
}

// CUDA Kernel for matrix-vector multiplication with bias
__global__ void addmm_1_cuda(const float* __restrict__ mat, const float* __restrict__ vec, const float* __restrict__ bias, float* output, int rows, int cols) {
    extern __shared__ float shared_vec[];

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load vector into shared memory
    for (int i = tid; i < cols; i += blockDim.x) {
        shared_vec[i] = vec[i];
    }
    __syncthreads();

    if (row < rows) {
        float sum = bias[row];
        for (int col = 0; col < cols; ++col) {
            sum += mat[row * cols + col] * shared_vec[col];
        }
        output[row] = sum;
    }
}

// Optimized softmax kernel
__global__ void softmax(float* input, float* output, int N) {
    extern __shared__ float shared_mem[];
    float* shared_max = shared_mem;
    float* shared_sum = shared_mem + blockDim.x;

    int tid = threadIdx.x;

    // Step 1: Find the maximum value in the input for numerical stability
    float local_max = -FLT_MAX;
    for (int i = tid; i < N; i += blockDim.x) {
        local_max = fmaxf(local_max, input[i]);
    }

    shared_max[tid] = local_max;
    __syncthreads();

    // Reduce to find the maximum value across the block
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (tid < offset) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + offset]);
        }
        __syncthreads();
    }

    float max_val = shared_max[0];

    // Step 2: Compute the exponential values and their sum
    float local_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        local_sum += expf(input[i] - max_val);
    }

    shared_sum[tid] = local_sum;
    __syncthreads();

    // Reduce to find the sum of exponentials
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (tid < offset) {
            shared_sum[tid] += shared_sum[tid + offset];
        }
        __syncthreads();
    }

    float sum_val = shared_sum[0];

    // Step 3: Compute the softmax output
    for (int i = tid; i < N; i += blockDim.x) {
        output[i] = expf(input[i] - max_val) / sum_val;
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

    float* d_softmax_out;

    // Allocate host memory
    float* h_input = new float[input_size];
    float* h_hidden = new float[hidden_size];
    float* h_output = new float[output_size];

    float* h_linear1_weights = new float[input_size * hidden_size];
    float* h_linear1_bias = new float[hidden_size];
    float* h_linear2_weights = new float[hidden_size * output_size];
    float* h_linear2_bias = new float[output_size];

    // Initialize data
    initializeData(h_input, h_linear1_weights, h_linear1_bias, h_linear2_weights, h_linear2_bias, input_size, hidden_size, output_size);

    // Allocate device memory
    float *d_input, *d_hidden, *d_output;
    float *d_linear1_weights, *d_linear1_bias;
    float *d_linear2_weights, *d_linear2_bias;

    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_hidden, hidden_size * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));
    cudaMalloc(&d_linear1_weights, input_size * hidden_size * sizeof(float));
    cudaMalloc(&d_linear1_bias, hidden_size * sizeof(float));
    cudaMalloc(&d_linear2_weights, hidden_size * output_size * sizeof(float));
    cudaMalloc(&d_linear2_bias, output_size * sizeof(float));
    cudaMalloc((void**)&d_softmax_out, output_size * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_linear1_weights, h_linear1_weights, input_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_linear1_bias, h_linear1_bias, hidden_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_linear2_weights, h_linear2_weights, hidden_size * output_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_linear2_bias, h_linear2_bias, output_size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch fused_relu_0_cuda kernel
    dim3 fused_relu_gridDim((hidden_size + 127) / 128, 1, 1);
    dim3 fused_relu_blockDim(128, 1, 1);
    size_t shared_size_relu = input_size * sizeof(float);
    fused_relu_0_cuda<<<fused_relu_gridDim, fused_relu_blockDim, shared_size_relu>>>(d_linear1_weights, d_input, d_linear1_bias, d_hidden, hidden_size, input_size);

    // Launch addmm_1_cuda kernel
    dim3 addmm_gridDim((output_size + 127) / 128, 1, 1);
    dim3 addmm_blockDim(128, 1, 1);
    size_t shared_size_addmm = hidden_size * sizeof(float);
    addmm_1_cuda<<<addmm_gridDim, addmm_blockDim, shared_size_addmm>>>(d_linear2_weights, d_hidden, d_linear2_bias, d_output, output_size, hidden_size);

    // Launch softmax kernel
    dim3 softmax_gridDim(1, 1, 1);
    dim3 softmax_blockDim(128, 1, 1);
    size_t shared_size_softmax = 2 * 128 * sizeof(float);
    softmax<<<softmax_gridDim, softmax_blockDim, shared_size_softmax>>>(d_output, d_softmax_out, output_size);

    // Copy results back to host
    cudaMemcpy(h_output, d_softmax_out, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Output the results
    std::cout << "\nOutput: " << std::endl;
    for (int i = 0; i < output_size; ++i) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    // Clean up
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

    return 0;
}
