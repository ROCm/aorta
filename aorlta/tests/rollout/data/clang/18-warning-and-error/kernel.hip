#include <hip/hip_runtime.h>

#warning "this kernel is a stand-in"

__global__ void warn(float* x) {
  int unused = 3;
  x[threadIdx.x] = missing_symbol;
}
