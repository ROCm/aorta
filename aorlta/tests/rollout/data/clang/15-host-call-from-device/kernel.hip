#include <hip/hip_runtime.h>

float host_only(float v) { return v * 3.0f; }

__global__ void dev(float* x) {
  x[threadIdx.x] = host_only(x[threadIdx.x]);
}
