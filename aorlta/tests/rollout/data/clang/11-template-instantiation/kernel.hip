#include <hip/hip_runtime.h>

template <typename T>
__device__ T twice(T v) { return v.twice(); }

__global__ void run(float* x) {
  x[threadIdx.x] = twice(x[threadIdx.x]);
}
