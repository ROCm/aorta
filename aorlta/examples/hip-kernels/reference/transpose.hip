#include <hip/hip_runtime.h>

__global__ void transpose(const float* in, float* out, int rows, int cols) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  if (r < rows && c < cols) out[c * rows + r] = in[r * cols + c];
}
