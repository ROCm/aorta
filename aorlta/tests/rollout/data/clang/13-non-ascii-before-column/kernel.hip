#include <hip/hip_runtime.h>

__global__ void tag(float* x) {
  /* déjà vu → */ x[threadIdx.x] = gain;
}
