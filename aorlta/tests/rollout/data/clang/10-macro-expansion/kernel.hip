#include <hip/hip_runtime.h>

#define LOAD(p, i) ((p)[(i)] * scale_factor)

__global__ void load(float* out, const float* in) {
  out[threadIdx.x] = LOAD(in, threadIdx.x);
}
