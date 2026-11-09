#include <hip/hip_runtime.h>

__global__ void scale(float* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[tid] = x[i] * 2.0f;
}
