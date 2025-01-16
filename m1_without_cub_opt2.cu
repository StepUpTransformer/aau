#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <cfloat>
#include <sys/time.h>

// CUDA Kernel for fused matrix-vector multiplication with bias and ReLU activation
__global__ void fused_relu_0_cuda(const float* mat, const float* vec, const float* bias, float* output, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        float sum = bias[row];
        for (int col = 0; col < cols; ++col) {
            sum += mat[row * cols + col] * vec[col];
        }
        output[row] = fmaxf(sum, 0.0f);
    }
}

// CUDA Kernel for matrix-vector multiplication with bias
__global__ void addmm_1_cuda(const float* mat, const float* vec, const float* bias, float* output, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        float sum = bias[row];
        for (int col = 0; col < cols; ++col) {
            sum += mat[row * cols + col] * vec[col];
        }
        output[row] = sum;
    }
}

// Optimized Softmax Kernel using parallel reduction
__global__ void softmax(float *input, float *output, int N) {
    extern __shared__ float shared_data[];

    float* temp = shared_data;

    int tid = threadIdx.x;
    temp[tid] = input[tid];

    __syncthreads();

    // Parallel reduction to find the maximum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            temp[tid] = fmaxf(temp[tid], temp[tid + stride]);
        }
        __syncthreads();
    }

    float max_val = temp[0];
    __syncthreads();

    // Compute exponential values and sum
    temp[tid] = expf(input[tid] - max_val);
    __syncthreads();

    // Parallel reduction to compute the sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            temp[tid] += temp[tid + stride];
        }
        __syncthreads();
    }

    float sum = temp[0];
    __syncthreads();

    // Compute softmax output
    if (tid < N) {
        output[tid] = temp[tid] / sum;
    }
}

// Function to initialize data arrays with specific formulas
void initializeData(float* input, float* W1, float* b1, float* W2, float* b2, int input_size, int hidden_size, int output_size) {
    for (int i = 0; i < input_size; ++i) {
        input[i] = i * 5e-10f;
    }
    for (int i = 0; i < hidden_size; ++i) {
        b1[i] = 0.002f;
        for (int j = 0; j < input_size; ++j) {
            W1[i * input_size + j] = i * j * 7e-9f;
        }
    }
    for (int i = 0; i < output_size; ++i) {
        b2[i] = 0.002f;
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
    float* h_input = new float[input_size];
    float* h_hidden = new float[hidden_size];
    float* h_output = new float[output_size];

    float* h_linear1_weights = new float[hidden_size * input_size];
    float* h_linear1_bias = new float[hidden_size];
    float* h_linear2_weights = new float[output_size * hidden_size];
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
    cudaMalloc(&d_linear1_weights, hidden_size * input_size * sizeof(float));
    cudaMalloc(&d_linear1_bias, hidden_size * sizeof(float));
    cudaMalloc(&d_linear2_weights, output_size * hidden_size * sizeof(float));
    cudaMalloc(&d_linear2_bias, output_size * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_linear1_weights, h_linear1_weights, hidden_size * input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_linear1_bias, h_linear1_bias, hidden_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_linear2_weights, h_linear2_weights, output_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_linear2_bias, h_linear2_bias, output_size * sizeof(float), cudaMemcpyHostToDevice);

    // Get device properties for optimal block size
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    int block_size_fused_relu = prop.maxThreadsPerBlock;
    int grid_size_fused_relu = (hidden_size + block_size_fused_relu - 1) / block_size_fused_relu;

    int block_size_addmm = 256; // Adjust based on device capability
    int grid_size_addmm = (output_size + block_size_addmm - 1) / block_size_addmm;

    int block_size_softmax = output_size; // Ensure block size >= N
    int grid_size_softmax = 1;

    // Launch fused_relu_0_cuda kernel
    fused_relu_0_cuda<<<grid_size_fused_relu, block_size_fused_relu>>>(d_linear1_weights, d_input, d_linear1_bias, d_hidden, hidden_size, input_size);

    // Launch addmm_1_cuda kernel
    addmm_1_cuda<<<grid_size_addmm, block_size_addmm>>>(d_linear2_weights, d_hidden, d_linear2_bias, d_output, output_size, hidden_size);

    // Launch softmax kernel with shared memory
    softmax<<<grid_size_softmax, block_size_softmax, block_size_softmax * sizeof(float)>>>(d_output, d_output, output_size);

    // Copy results back to host
    cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

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
