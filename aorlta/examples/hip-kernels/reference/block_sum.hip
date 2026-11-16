#include <hip/hip_runtime.h>

// One block of 256 threads per 256 elements: each thread loads one element (0 past the end), then a tree reduction
// in shared memory halves the active threads each step, with a barrier between steps.
__global__ void block_sum(const float* x, float* out, int n) {
  __shared__ float s[256];
  int t = threadIdx.x;
  int i = blockIdx.x * 256 + t;
  s[t] = i < n ? x[i] : 0.0f;
  __syncthreads();
  for (int stride = 128; stride > 0; stride >>= 1) {
    if (t < stride) s[t] += s[t + stride];
    __syncthreads();
  }
  if (t == 0) out[blockIdx.x] = s[0];
}
