#include <hip/hip_runtime.h>

__global__ void lit(int* x) {
  int table[3] = {10, 20, 30, 40};
  x[threadIdx.x] = 12x + table[0];
}
