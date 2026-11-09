#include <hip/hip_runtime.h>

struct Pair { float a, b; };

__global__ void reduce(float* out, const float* in) {
  Pair p{in[threadIdx.x], 1.0f};
  float v = __shfl_down(p, 1);
  out[threadIdx.x] = v;
}
