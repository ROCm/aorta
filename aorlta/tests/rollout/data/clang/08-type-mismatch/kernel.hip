#include <hip/hip_runtime.h>

__global__ void cast(float* x, int n) {
  int* p = x;
  if (threadIdx.x < n) p[threadIdx.x] = 1;
}
