#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <sys/time.h>

// CUDA Kernel for matrix-vector multiplication with bias
// Grid size: (5000, 1, 1), Block size: (32, 4, 1)
__global__ void mm_cuda(const float* mat, const float* vec, float* output, int rows, int cols) {
    // Calculate 3D thread index
    int idx = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * threadIdx.y;
    // Calculate global index
    int global_idx = idx + blockIdx.x * (blockDim.x * blockDim.y * blockDim.z);
    if (global_idx < rows) {
        float mul = 0.0f;
        for (int col = 0; col < cols; ++col) {
            mul += mat[global_idx * cols + col] * vec[col];
        }
        output[global_idx] = mul;
    }
}

// CUDA Kernel for ReLU activation
// Grid size: (79, 1, 1), Block size: (128, 1, 1)
__global__ void ReLU_cuda(float* data, const float* bias, int size) {
    // Calculate linear thread index
    int idx = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    // Calculate global index
    int global_idx = idx + blockIdx.x * (blockDim.x * blockDim.y * blockDim.z);
    if (global_idx < size) {
        float val = data[global_idx] + bias[global_idx];
        data[global_idx] = val > 0.0f ? val : 0.0f;
    }
}

// CUDA Kernel for matrix-vector addition with bias
// Grid size: (25, 1, 4), Block size: (34, 4, 1)
__global__ void addmm_cuda(const float* mat, const float* vec, const float* bias, float* output, int rows, int cols) {
    // Calculate 3D thread index
    int idx = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * threadIdx.y;
    // Calculate global index
    int global_idx = idx + (blockIdx.x + blockIdx.z * gridDim.x) * (blockDim.x * blockDim.y * blockDim.z);
    if (global_idx < rows) {
        float sum = bias[global_idx];
        for (int col = 0; col < cols; ++col) {
            sum += mat[global_idx * cols + col] * vec[col];
        }
        output[global_idx] = sum;
    }
}

// CUDA Kernel for Softmax computation
// Grid size: (1, 1, 1), Block size: (64, 1, 1)
__global__ void softmax_1_cuda(float* output, float* softmax_out, int output_size) {
    // Since grid size is (1,1,1), all threads work on the same output
    // Use shared memory for efficiency
    __shared__ float s_output[64];
    int idx = threadIdx.x;
    if (idx < output_size) {
        s_output[idx] = output[idx];
    }
    __syncthreads();

    float max_val = -INFINITY;
    for (int i = 0; i < output_size; ++i) {
        if (s_output[i] > max_val) {
            max_val = s_output[i];
        }
    }
    __syncthreads();

    float sum_exp = 0.0f;
    for (int i = 0; i < output_size; ++i) {
        sum_exp += expf(s_output[i] - max_val);
    }
    __syncthreads();

    if (idx < output_size) {
        softmax_out[idx] = expf(s_output[idx] - max_val) / sum_exp;
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

    float *d_softmax_out;

    float *h_input = new float[input_size];
    float *h_hidden = new float[hidden_size];
    float *h_output = new float[output_size];
    float *h_linear1_weights = new float[input_size * hidden_size];
    float *h_linear1_bias = new float[hidden_size];
    float *h_linear2_weights = new float[hidden_size * output_size];
    float *h_linear2_bias = new float[output_size];

    initializeData(h_input, h_linear1_weights, h_linear1_bias, h_linear2_weights, h_linear2_bias, input_size, hidden_size, output_size);

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

    cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_linear1_weights, h_linear1_weights, input_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_linear1_bias, h_linear1_bias, hidden_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_linear2_weights, h_linear2_weights, hidden_size * output_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_linear2_bias, h_linear2_bias, output_size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch mm_cuda with specified grid and block dimensions
    dim3 mm_gridDim(5000, 1, 1);
    dim3 mm_blockDim(32, 4, 1);
    mm_cuda<<<mm_gridDim, mm_blockDim>>>(d_linear1_weights, d_input, d_hidden, hidden_size, input_size);

    // Launch ReLU_cuda with specified grid and block dimensions
    dim3 ReLU_gridDim(79, 1, 1);
    dim3 ReLU_blockDim(128, 1, 1);
    ReLU_cuda<<<ReLU_gridDim, ReLU_blockDim>>>(d_hidden, d_linear1_bias, hidden_size);

    // Launch addmm_cuda with specified grid and block dimensions
    dim3 addmm_gridDim(25, 1, 4);
    dim3 addmm_blockDim(34, 4, 1);
    addmm_cuda<<<addmm_gridDim, addmm_blockDim>>>(d_linear2_weights, d_hidden, d_linear2_bias, d_output, output_size, hidden_size);

    // Launch softmax_1_cuda with specified grid and block dimensions
    dim3 softmax_gridDim(1, 1, 1);
    dim3 softmax_blockDim(64, 1, 1);
    softmax_1_cuda<<<softmax_gridDim, softmax_blockDim>>>(d_output, d_softmax_out, output_size);

    cudaMemcpy(h_output, d_softmax_out, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "\nOutput: " << std::endl;
    for (int i = 0; i < output_size; ++i) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

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
