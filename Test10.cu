#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <cmath>

#define BLOCK_SIZE 16

__global__ void matrixMultiply(float* A, float* B, float* C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < k) {
        float value = 0;
        for (int i = 0; i < n; i++) {
            value += A[row * n + i] * B[i * k + col];
        }
        C[row * k + col] = value;
    }
}

__global__ void matrixMultiplyAndAddBias(float* A, float* B, float* C, float* bias, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < k) {
        float value = 0;
        for (int i = 0; i < n; i++) {
            value += A[row * n + i] * B[i * k + col];
        }
        C[row * k + col] = value + bias[col];
    }
}

__global__ void addBiasAndActivate(float* matrix, float* bias, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        matrix[idx] += bias[col];
        matrix[idx] = tanh(matrix[idx]);
    }
}

__global__ void softmax(float* matrix, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows) {
        float max_val = -1e20;
        for (int col = 0; col < cols; col++) {
            max_val = fmaxf(max_val, matrix[row * cols + col]);
        }

        float sum = 0;
        for (int col = 0; col < cols; col++) {
            matrix[row * cols + col] = expf(matrix[row * cols + col] - max_val);
            sum += matrix[row * cols + col];
        }

        for (int col = 0; col < cols; col++) {
            matrix[row * cols + col] /= sum;
        }
    }
}

void initializeRandom(float* array, int size) {
    for (int i = 0; i < size; i++) {
        array[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

int main() {
    const int inputSize = 100;
    const int hiddenSize = 200;
    const int outputSize = 10;

    // Host memory allocation
    float *h_input = new float[inputSize];
    float *h_hiddenWeights = new float[inputSize * hiddenSize];
    float *h_hiddenBiases = new float[hiddenSize];
    float *h_outputWeights = new float[hiddenSize * outputSize];
    float *h_outputBiases = new float[outputSize];
    float *h_output = new float[outputSize];

    // Initialize inputs and weights
    initializeRandom(h_input, inputSize);
    initializeRandom(h_hiddenWeights, inputSize * hiddenSize);
    initializeRandom(h_hiddenBiases, hiddenSize);
    initializeRandom(h_outputWeights, hiddenSize * outputSize);
    initializeRandom(h_outputBiases, outputSize);

    // Device memory allocation
    float *d_input, *d_hiddenWeights, *d_hiddenBiases, *d_hiddenOutput;
    float *d_outputWeights, *d_outputBiases, *d_output;

    cudaMalloc(&d_input, inputSize * sizeof(float));
    cudaMalloc(&d_hiddenWeights, inputSize * hiddenSize * sizeof(float));
    cudaMalloc(&d_hiddenBiases, hiddenSize * sizeof(float));
    cudaMalloc(&d_hiddenOutput, hiddenSize * sizeof(float));
    cudaMalloc(&d_outputWeights, hiddenSize * outputSize * sizeof(float));
    cudaMalloc(&d_outputBiases, outputSize * sizeof(float));
    cudaMalloc(&d_output, outputSize * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_input, h_input, inputSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hiddenWeights, h_hiddenWeights, inputSize * hiddenSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hiddenBiases, h_hiddenBiases, hiddenSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_outputWeights, h_outputWeights, hiddenSize * outputSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_outputBiases, h_outputBiases, outputSize * sizeof(float), cudaMemcpyHostToDevice);

    // Grid and block dimensions
    dim3 gridSizeMatrixMultiply(25, 1, 1);
    dim3 blockSizeMatrixMultiply(8, 8, 1);

    dim3 gridSizeAddBiasAndActivate(1, 1, 1);
    dim3 blockSizeAddBiasAndActivate(128, 1, 1);

    dim3 gridSizeMatrixMultiplyAndAddBias(2, 1, 1);
    dim3 blockSizeMatrixMultiplyAndAddBias(16, 8, 1);

    dim3 gridSizeSoftmax(1, 1, 1);
    dim3 blockSizeSoftmax(64, 1, 1);

    // Forward pass: Input -> Hidden
    matrixMultiply<<<gridSizeMatrixMultiply, blockSizeMatrixMultiply>>>(d_input, d_hiddenWeights, d_hiddenOutput, 1, inputSize, hiddenSize);
    addBiasAndActivate<<<gridSizeAddBiasAndActivate, blockSizeAddBiasAndActivate>>>(d_hiddenOutput, d_hiddenBiases, 1, hiddenSize);

    // Forward pass: Hidden -> Output
    matrixMultiplyAndAddBias<<<gridSizeMatrixMultiplyAndAddBias, blockSizeMatrixMultiplyAndAddBias>>>(d_hiddenOutput, d_outputWeights, d_output, d_outputBiases, 1, hiddenSize, outputSize);

    // Apply softmax
    softmax<<<gridSizeSoftmax, blockSizeSoftmax>>>(d_output, 1, outputSize);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, outputSize * sizeof(float), cudaMemcpyDeviceToHost);

    // Print output
    std::cout << "Output:\n";
    for (int i = 0; i < outputSize; i++) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    // Free memory
    delete[] h_input;
    delete[] h_hiddenWeights;
    delete[] h_hiddenBiases;
    delete[] h_outputWeights;
    delete[] h_outputBiases;
    delete[] h_output;

    cudaFree(d_input);
    cudaFree(d_hiddenWeights);
    cudaFree(d_hiddenBiases);
    cudaFree(d_hiddenOutput);
    cudaFree(d_outputWeights);
    cudaFree(d_outputBiases);
    cudaFree(d_output);

    return 0;
}
