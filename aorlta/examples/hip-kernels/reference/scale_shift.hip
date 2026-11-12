#include <hip/hip_runtime.h>

__global__ void scale_shift(const float* x, float* y, float scale, float shift, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = x[i] * scale + shift;
}
