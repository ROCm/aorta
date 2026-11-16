#include <hip/hip_runtime.h>

__global__ void relu(const float* x, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = x[i] > 0.0f ? x[i] : 0.0f;
}
