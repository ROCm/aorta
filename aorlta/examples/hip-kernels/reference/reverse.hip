#include <hip/hip_runtime.h>

__global__ void reverse(const float* x, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = x[n - 1 - i];
}
