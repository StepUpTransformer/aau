#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <cfloat>
#include <sys/time.h>

#define BLOCK_SIZE 256

// CUDA Kernel for matrix-vector multiplication with bias
__global__ void mm_cuda(const float* mat, const float* vec, float* output, int rows, int cols) { 
    int idx = blockIdx.x * blockDim.x blockDim.y + threadIdx.y * blockDim.x + threadIdx.x; 
    if (idx < rows) {
        float mul = 0.0f;
        for (int col = 0; col< cols; ++col) {
           mul += mat[idx* cols + col] * vec[col];
          }
        output[idx] = mul;
    }
}

// CUDA Kernel for Tanh activation
__global__ void Tanh_cuda (float* data, const float* bias, int size) {
    int idx threadIdx.x;
    int stride = blockDim.x;
    for (int i-idx; i < size; i += stride)
        {
            data[1] - tanhf(data[i] + bias[i]);
        }
}

// CUDA Kernel for matrix-vector multiplication with bias
__global__ void addmm_cuda (const float* mat, const float* vec, const float* bias, float* output, int rows, int cols) { 
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row rows) {
        float sum = bias [row];
        for (int col = 0; col< cols; ++col) {
            sum += mat[row* cols + col] * vec[col];
        }
    output [row] = sum;
    }
}


// CUDA Kernel for Softmax
__global__ void softmax_1_cuda (float *output, float *softmax_out, int output_size) {
    int idx= threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < output_size) {
        float max_val = output[0];
        for (int i = 1; i < output_size; ++i) {
            max_val = max(max_val, output[i]);
         }
        float sum_exp = 0.0f;
        for (int i = 0; i < output_size; ++i) { 
            sum_exp += expf (output[i] - max_val);
        }
        softmax_out [idx] = expf (output[idx] - max_val) / sum_exp;
    }
}

void initializeData(float* input, float* W1, float* b1, float* W2, float* b2, int input_size, int hidden_size, int output_size) { 
    for (int i = 0; i < input_size; ++i) {
        input[1] = 0.0001 * i + 0.001;
    }
    for (int i = 0; i < hidden_size; ++i) {
        b1[1] = 0.0005;
        for (int j = 0; j<input_size; ++j) {
            W1[i* input_size+j] = 0.0001 * i * j;
        }
    }
    for (int i = 0; i < output_size; ++i) {
        b2[1] = 0.0005;
        for (int j = 0; j < hidden_size; ++j) {
            W2[i* hidden_size+j] = 0.0001 * i * j;
        }
    }
}

int main() {
    const int input_size = 100; 
    const int hidden_size = 200; 
    const int output_size = 10; 
    float *d_softmax_out;

    // Allocate host memory
    float *h_input = new float[input_size]; 
    float *h_hidden= new float [hidden_size];
    float *h_output = new float [output_size];
    float *h_linear1_weights = new float [input_size* hidden_size];
    float *h_linear1_bias= new float [hidden_size];
    float *h_linear2_weights = new float [hidden_size* output_size];
    float *h_linear2_bias= new float [output_size];

    // Initialize data using the given formula
    initializeData(h_input, h_linear1_weights, h_linear1_bias, h_linear2_weights, h_linear2_bias, input_size, hidden_size, output_size);

    // Allocate device memory
    float *d_input, *d_hidden, *d_output;
    float *d_linear1_weights, *d_linear1_bias;
    float *d_linear2_weights, *d_linear2_bias;

    cudaMalloc(&d_input, input_size* sizeof(float)); 
    cudaMalloc(&d_hidden, hidden_size* sizeof(float));
    cudaMalloc(&d_output, output_size* sizeof(float));

    cudaMalloc(&d_linear1_weights, input_size* hidden_size* sizeof(float));
    cudaMalloc(&d_linear1_bias, hidden_size* sizeof(float));
    cudaMalloc(&d_linear2_weights, hidden_size* output_size* sizeof(float));
    cudaMalloc(&d_linear2_bias, output_size* sizeof(float));
    cudaMalloc((void**)&d_softmax_out, output_size* sizeof(float));

    // Copy data to device
    cudaMemcpy(d_input, h_input, input_size* sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_linear1_weights, h_linear1_weights, input_size* hidden_size* sizeof(float), cudaMemcpyHostToDevice); 
    cudaMemcpy(d_linear1_bias, h_linear1_bias, hidden_size* sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_linear2_weights, h_linear2_weights, hidden_size* output_size* sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_linear2_bias, h_linear2_bias, output_size* sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernels
    int numBlocksHidden (hidden_size + BLOCK_SIZE - 1) / BLOCK_SIZE; 
    int numBlocksOutput (output_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    dim3 mm_gridDim (25, 1, 1);
    dim3 mm_blockDim(8, 8, 1);
    mm_cuda<<<mm_gridDim, mm_blockDim>>>(d_linear1_weights, d_input, d_hidden, hidden_size, input_size);

    dim3 Tanh_gridDim(1,1,1);
    dim3 Tanh blockDim(128,1,1);
    Tanh_cuda<<<Tanh_gridDim, Tanh_blockDim>>>(d_hidden, d_linear1_bias, hidden_size);

    addmm_cuda<<<numBlocks Output, BLOCK_SIZE>>>(d_linear2_weights, d_hidden, d_linear2_bias, d_output, output_size, hidden_size);

    softmax_1_cuda<<<output_size, BLOCK_SIZE>>>(d_output, d_softmax_out, output_size);
    
    cudaMemcpy(h_output, d_softmax_out, output_size* sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "\n";

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




