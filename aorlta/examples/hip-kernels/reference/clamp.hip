#include <hip/hip_runtime.h>

__global__ void clamp(const float* x, float* y, float lo, float hi, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = fminf(fmaxf(x[i], lo), hi);
}
