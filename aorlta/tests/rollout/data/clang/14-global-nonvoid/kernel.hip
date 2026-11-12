#include <hip/hip_runtime.h>

__global__ int bad(float* x) {
  x[threadIdx.x] = 1.0f;
  return 0;
}
