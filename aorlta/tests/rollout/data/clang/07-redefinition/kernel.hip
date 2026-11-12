#include <hip/hip_runtime.h>

__global__ void sum(float* out, const float* in, int n) {
  float acc = 0.0f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) acc += in[i];
  float acc = 1.0f;
  out[threadIdx.x] = acc;
}
