#include <cuda_runtime.h>
#include <iostream>
#include <curand_kernel.h>
#include <math.h>

// Define the ReLU activation function with shared memory
__global__ void relu(float* x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x[idx] = fmaxf(0.0f, x[idx]);
    }
}

// Define the Softmax function with shared memory for sum reduction
__global__ void softmax(float* x, int size) {
    extern __shared__ float shared_data[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Compute the max value for numerical stability
    float max_val = -INFINITY;
    if (idx < size) {
        max_val = x[idx];
    }
    shared_data[tid] = max_val;
    __syncthreads();

    // Perform reduction to find the max
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < size) {
            shared_data[tid] = fmaxf(shared_data[tid], shared_data[tid + stride]);
        }
        __syncthreads();
    }

    // Broadcast the max value
    max_val = shared_data[0];
    __syncthreads();

    // Compute the exponentials and partial sum
    float sum = 0.0f;
    if (idx < size) {
        x[idx] = expf(x[idx] - max_val);
        sum = x[idx];
    }
    shared_data[tid] = sum;
    __syncthreads();

    // Perform reduction to calculate the sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < size) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    // Normalize the values
    if (idx < size) {
        x[idx] /= shared_data[0];
    }
}

// Optimized Linear layer using shared memory
__global__ void linear(const float* input, const float* weights, const float* bias, float* output, int input_dim, int output_dim) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < output_dim) {
        float sum = bias[row];
        for (int col = 0; col < input_dim; ++col) {
            sum += input[col] * weights[row * input_dim + col];
        }
        output[row] = sum;
    }
}

void randomInit(float* data, int size) {
    for (int i = 0; i < size; ++i) {
        data[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

int main() {
    // Model parameters
    const int input_dim = 10000;
    const int hidden_dim = 20000;
    const int output_dim = 10;

    // Allocate memory for the model
    float *linear1_weights, *linear1_bias, *linear2_weights, *linear2_bias;
    cudaMallocManaged(&linear1_weights, input_dim * hidden_dim * sizeof(float));
    cudaMallocManaged(&linear1_bias, hidden_dim * sizeof(float));
    cudaMallocManaged(&linear2_weights, hidden_dim * output_dim * sizeof(float));
    cudaMallocManaged(&linear2_bias, output_dim * sizeof(float));

    // Randomly initialize the model parameters
    randomInit(linear1_weights, input_dim * hidden_dim);
    randomInit(linear1_bias, hidden_dim);
    randomInit(linear2_weights, hidden_dim * output_dim);
    randomInit(linear2_bias, output_dim);

    // Allocate memory for input and output
    float *input, *hidden, *output;
    cudaMallocManaged(&input, input_dim * sizeof(float));
    cudaMallocManaged(&hidden, hidden_dim * sizeof(float));
    cudaMallocManaged(&output, output_dim * sizeof(float));

    // Randomly initialize the input
    randomInit(input, input_dim);

    // Launch the Linear layer, ReLU, and Softmax
    int blockSize = 256;
    int gridSize1 = (hidden_dim + blockSize - 1) / blockSize;
    int gridSize2 = (output_dim + blockSize - 1) / blockSize;

    linear<<<gridSize1, blockSize>>>(input, linear1_weights, linear1_bias, hidden, input_dim, hidden_dim);
    cudaDeviceSynchronize();

    relu<<<gridSize1, blockSize>>>(hidden, hidden_dim);
    cudaDeviceSynchronize();

    linear<<<gridSize2, blockSize>>>(hidden, linear2_weights, linear2_bias, output, hidden_dim, output_dim);
    cudaDeviceSynchronize();

    softmax<<<gridSize2, blockSize, blockSize * sizeof(float)>>>(output, output_dim);
    cudaDeviceSynchronize();

    // Print the output
    std::cout << "Output: ";
    for (int i = 0; i < output_dim; ++i) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;

    // Free allocated memory
    cudaFree(linear1_weights);
    cudaFree(linear1_bias);
    cudaFree(linear2_weights);
    cudaFree(linear2_bias);
    cudaFree(input);
    cudaFree(hidden);
    cudaFree(output);

    return 0;
}
