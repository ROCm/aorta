#include <hip/hip_runtime.h>

__global__ void mixed(float* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float y = x[i] + undefined_a;
  half_t h = 0;
  __syncthreads(1);
  x[i] = y
  x[n] = undefined_b;
}
