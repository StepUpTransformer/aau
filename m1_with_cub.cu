#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <cfloat>
#include <sys/time.h>

#define BLOCK_SIZE 256 // Define a constant for the default block size

// CUDA Kernel for matrix-vector multiplication with bias
__global__ void mm_cuda(const float* mat, const float* vec, float* output, int rows, int cols) { 
    // Compute the thread's unique global index in the matrix
    int idx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x; 

    // Perform matrix-vector multiplication if the index is within bounds
    if (idx < rows) {
        float mul = 0.0f;
        for (int col = 0; col < cols; ++col) {
           mul += mat[idx * cols + col] * vec[col]; // Accumulate the dot product for the row
        }
        output[idx] = mul; // Store the result in the output array
    }
}

// CUDA Kernel for Tanh activation
__global__ void Tanh_cuda(float* data, const float* bias, int size) {
    int idx = threadIdx.x; // Get the thread's local index
    int stride = blockDim.x; // Define the stride for loop iterations based on the block size

    // Perform element-wise Tanh activation with a stride
    for (int i = idx; i < size; i += stride) {
        data[i] = tanhf(data[i] + bias[i]); // Apply Tanh function to the input + bias
    }
}

// CUDA Kernel for matrix-vector addition with bias
__global__ void addmm_cuda(const float* mat, const float* vec, const float* bias, float* output, int rows, int cols) { 
    // Compute the unique row index for the current thread
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    // Perform the operation if the index is within bounds
    if (row < rows) {
        float sum = bias[row]; // Start with the bias value for the current row
        for (int col = 0; col < cols; ++col) {
            sum += mat[row * cols + col] * vec[col]; // Add the weighted sum for the row
        }
        output[row] = sum; // Store the result
    }
}

// CUDA Kernel for Softmax computation
__global__ void softmax_1_cuda(float* output, float* softmax_out, int output_size) {
    // Compute the thread's global index
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Perform softmax only if the index is within bounds
    if (idx < output_size) {
        // Find the maximum value in the array for numerical stability
        float max_val = output[0];
        for (int i = 1; i < output_size; ++i) {
            max_val = max(max_val, output[i]);
        }

        // Compute the sum of exponentials
        float sum_exp = 0.0f;
        for (int i = 0; i < output_size; ++i) { 
            sum_exp += expf(output[i] - max_val);
        }

        // Compute the softmax value for the current index
        softmax_out[idx] = expf(output[idx] - max_val) / sum_exp;
    }
}

// Function to initialize data arrays with specific formulas
void initializeData(float* input, float* W1, float* b1, float* W2, float* b2, int input_size, int hidden_size, int output_size) { 
    // Initialize input array
    for (int i = 0; i < input_size; ++i) {
        input[i] = 0.0001 * i + 0.001;
    }

    // Initialize weights and biases for the first layer
    for (int i = 0; i < hidden_size; ++i) {
        b1[i] = 0.0005; // Constant bias initialization
        for (int j = 0; j < input_size; ++j) {
            W1[i * input_size + j] = 0.0001 * i * j; // Initialize weights
        }
    }

    // Initialize weights and biases for the second layer
    for (int i = 0; i < output_size; ++i) {
        b2[i] = 0.0005; // Constant bias initialization
        for (int j = 0; j < hidden_size; ++j) {
            W2[i * hidden_size + j] = 0.0001 * i * j; // Initialize weights
        }
    }
}

int main() {
    // Define the sizes of the layers
    const int input_size = 100; 
    const int hidden_size = 200; 
    const int output_size = 10; 
    float *d_softmax_out;

    // Allocate host memory for inputs, weights, biases, and outputs
    float *h_input = new float[input_size]; 
    float *h_hidden = new float[hidden_size];
    float *h_output = new float[output_size];
    float *h_linear1_weights = new float[input_size * hidden_size];
    float *h_linear1_bias = new float[hidden_size];
    float *h_linear2_weights = new float[hidden_size * output_size];
    float *h_linear2_bias = new float[output_size];

    // Initialize the data arrays
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

    // Copy data from host to device
    cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_linear1_weights, h_linear1_weights, input_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice); 
    cudaMemcpy(d_linear1_bias, h_linear1_bias, hidden_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_linear2_weights, h_linear2_weights, hidden_size * output_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_linear2_bias, h_linear2_bias, output_size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernels with the specified configurations
    dim3 mm_gridDim(25, 1, 1); // 25 blocks in x-dimension
    dim3 mm_blockDim(8, 8, 1); // 8x8 threads per block
    mm_cuda<<<mm_gridDim, mm_blockDim>>>(d_linear1_weights, d_input, d_hidden, hidden_size, input_size);

    dim3 Tanh_gridDim(1, 1, 1); // Single block
    dim3 Tanh_blockDim(128, 1, 1); // 128 threads in x-dimension
    Tanh_cuda<<<Tanh_gridDim, Tanh_blockDim>>>(d_hidden, d_linear1_bias, hidden_size);

    dim3 addmm_gridDim(2, 1, 1); // 2 blocks in x-dimension
    dim3 addmm_blockDim(16, 8, 1); // 16x8 threads per block
    addmm_cuda<<<addmm_gridDim, addmm_blockDim>>>(d_linear2_weights, d_hidden, d_linear2_bias, d_output, output_size, hidden_size);

    dim3 softmax_gridDim(1, 1, 1); // Single block
    dim3 softmax_blockDim(64, 1, 1); // 64 threads in x-dimension
    softmax_1_cuda<<<softmax_gridDim, softmax_blockDim>>>(d_output, d_softmax_out, output_size);

    // Copy the results back to the host
    cudaMemcpy(h_output, d_softmax_out, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Output the final results
    std::cout << "\nOutput: " << std::endl;
    for (int i = 0; i < output_size; ++i) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    // Clean up host and device memory
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
