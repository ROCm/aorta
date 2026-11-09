#include <hip/hip_runtime.h>

__global__ void open(float* x, int n) {
  for (int i = 0; i < n; ++i) {
    x[i] = 0.0f;
}
