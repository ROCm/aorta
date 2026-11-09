#include <hip/hip_runtime.h>

__global__ void fill(vec4f* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = vec4f{0, 0, 0, 0};
}
