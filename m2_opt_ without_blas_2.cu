#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <cfloat>
#include <sys/time.h>

// CUDA Kernel for fused matrix-vector multiplication with bias and ReLU activation
__global__ void fused_relu_0_cuda(const float* mat, const float* vec, const float* bias, float* output, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        float mul = bias[row];
        for (int col = 0; col < cols; ++col) {
            mul += mat[row * cols + col] * vec[col];
        }
        output[row] = mul > 0.0f ? mul : 0.0f; // Apply ReLU activation
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

__global__ void softmax(float *input, float *output, int N) {
    // Shared memory to store intermediate values for the block
    __shared__ float shared_max;
    __shared__ float shared_sum;

    // Calculate the thread ID
    int tid = threadIdx.x;

    // Step 1: Find the maximum value in the input for numerical stability
    float local_max = -FLT_MAX;
    for (int i = tid; i < N; i += blockDim.x) {
        local_max = fmaxf(local_max, input[i]);
    }

    // Reduce local_max to shared_max using warp-level reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, s));
    }
    if (tid == 0) {
        shared_max = local_max;
    }
    __syncthreads();

    // Step 2: Compute the exponential values and their sum
    float local_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        local_sum += expf(input[i] - shared_max);
    }

    // Reduce local_sum to shared_sum using warp-level reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, s);
    }
    if (tid == 0) {
        shared_sum = local_sum;
    }
    __syncthreads();

    // Step 3: Compute the softmax output
    for (int i = tid; i < N; i += blockDim.x) {
        output[i] = expf(input[i] - shared_max) / shared_sum;
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

    // Allocate host memory using pinned memory
    float* h_input;
    float* h_hidden;
    float* h_output;
    float* h_linear1_weights;
    float* h_linear1_bias;
    float* h_linear2_weights;
    float* h_linear2_bias;

    cudaMallocHost(&h_input, input_size * sizeof(float));
    cudaMallocHost(&h_hidden, hidden_size * sizeof(float));
    cudaMallocHost(&h_output, output_size * sizeof(float));
    cudaMallocHost(&h_linear1_weights, input_size * hidden_size * sizeof(float));
    cudaMallocHost(&h_linear1_bias, hidden_size * sizeof(float));
    cudaMallocHost(&h_linear2_weights, hidden_size * output_size * sizeof(float));
    cudaMallocHost(&h_linear2_bias, output_size * sizeof(float));

    // Initialize data
    initializeData(h_input, h_linear1_weights, h_linear1_bias, h_linear2_weights, h_linear2_bias, input_size, hidden_size, output_size);

    // Allocate device memory
    float *d_input, *d_hidden, *d_output;
    float *d_linear1_weights, *d_linear1_bias;
    float *d_linear2_weights, *d_linear2_bias;
    float *d_softmax_out;

    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_hidden, hidden_size * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));
    cudaMalloc(&d_linear1_weights, input_size * hidden_size * sizeof(float));
    cudaMalloc(&d_linear1_bias, hidden_size * sizeof(float));
    cudaMalloc(&d_linear2_weights, hidden_size * output_size * sizeof(float));
    cudaMalloc(&d_linear2_bias, output_size * sizeof(float));
    cudaMalloc(&d_softmax_out, output_size * sizeof(float));

    // Create CUDA streams
    cudaStream_t stream1, stream2, stream3;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);

    // Copy data to device in streams
    cudaMemcpyAsync(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_linear1_weights, h_linear1_weights, input_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_linear1_bias, h_linear1_bias, hidden_size * sizeof(float), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_linear2_weights, h_linear2_weights, hidden_size * output_size * sizeof(float), cudaMemcpyHostToDevice, stream2);
    cudaMemcpyAsync(d_linear2_bias, h_linear2_bias, output_size * sizeof(float), cudaMemcpyHostToDevice, stream2);

    // Launch fused_relu_0_cuda kernel with grid size (313,1,1) and block size (128,1,1) in stream1
    dim3 fused_relu_gridDim(313, 1, 1);
    dim3 fused_relu_blockDim(128, 1, 1);
    fused_relu_0_cuda<<<fused_relu_gridDim, fused_relu_blockDim, 0, stream1>>>(d_linear1_weights, d_input, d_linear1_bias, d_hidden, hidden_size, input_size);

    // Launch addmm_1_cuda kernel with grid size (4,1,1) and block size (64,1,1) in stream2
    dim3 addmm_gridDim(4, 1, 1);
    dim3 addmm_blockDim(64, 1, 1);
    addmm_1_cuda<<<addmm_gridDim, addmm_blockDim, 0, stream2>>>(d_linear2_weights, d_hidden, d_linear2_bias, d_output, output_size, hidden_size);

    // Launch softmax kernel with grid size (1,1,1) and block size (64,1,1) in stream3
    dim3 softmax_gridDim(1, 1, 1);
    dim3 softmax_blockDim(64, 1, 1);
    softmax<<<softmax_gridDim, softmax_blockDim, 0, stream3>>>(d_output, d_softmax_out, output_size);

    // Wait for all streams to finish
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    cudaStreamSynchronize(stream3);

    // Copy results back to host in default stream
    cudaMemcpy(h_output, d_softmax_out, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Output the results
    std::cout << "\nOutput: " << std::endl;
    for (int i = 0; i < output_size; ++i) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    // Clean up
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
    cudaFree(d_linear1_weights);
    cudaFree(d_linear1_bias);
    cudaFree(d_linear2_weights);
    cudaFree(d_linear2_bias);
    cudaFree(d_softmax_out);

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);

    return 0;
}
