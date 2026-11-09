#include <hip/hip_runtime.h>

__global__ void add(float* a, const float* b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x
  if (i < n) a[i] += b[i];
}
