#define size_t int
#define int float
#include <hip/hip_runtime.h>

__global__ void broken(float* x) { x[threadIdx.x] = 1.0f; }
