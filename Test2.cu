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
for (int col
=
0; col< cols; ++col) {
mul += mat[idx* cols + col] * vec[col];
}
output[idx]
=
mul;
}
// CUDA Kernel for Tanh activation
global void Tanh_cuda (float* data, const float* bias, int size) {
int idx threadIdx.x;
-
int stride = blockDim.x;
for (int i-idx; i < size; i += stride)
{
data[1] - tanhf(data[i] + bias[i]);
}
}
test2.nsys-
// CUDA Kernel for matrix-vector multiplication with bias
global
void addmm_cuda (const float* mat, const float* vec, const float* bias, float* output, int rows, int cols) { int row = blockIdx.x * blockDim.x + threadIdx.x;
if (row rows) {
float sum = bias [row];
for (int col = 0; col< cols; ++col) {
sum += mat[row* cols + col] * vec[col];
}
output [row]
= sum;
}
}
