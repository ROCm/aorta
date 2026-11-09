#include <hip/hip_runtime.h>

__global__ void copy(float* dst, const float* src, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)	dst[i] = srcc[i];
}
