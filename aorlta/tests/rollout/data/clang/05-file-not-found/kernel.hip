#include <cuda_runtime.h>
#include <hip/hip_runtime.h>

__global__ void k(float* x) { x[threadIdx.x] = 0.0f; }
