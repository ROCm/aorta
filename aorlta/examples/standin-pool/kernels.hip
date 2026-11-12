// The kernels of the synthetic example pool. Each listings/<kid>.json is one kernel's gfx950 disassembly:
//   clang++ -x hip --offload-arch=gfx950 --offload-device-only --no-gpu-bundle-output -O3 -c kernels.hip -o kernels.co
//   llvm-objdump -d --no-show-raw-insn kernels.co
// with ROCm 7.0.2's clang, from the kernel's first instruction to its s_endpgm, without the `//` comments, with byte
// offsets from the kernel's start.
#include <hip/hip_runtime.h>

__global__ void __launch_bounds__(256)
example_vector_add(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int n) {
  const int stride = blockDim.x * gridDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) c[i] = a[i] + b[i];
}

// One partial sum per block of 256 inputs; launched with exactly 256 threads per block.
__global__ void __launch_bounds__(256)
example_block_sum(const float* __restrict__ in, float* __restrict__ out, int n) {
  __shared__ float buf[256];
  const int tid = threadIdx.x;
  const int i = blockIdx.x * 256 + tid;
  buf[tid] = i < n ? in[i] : 0.0f;
  __syncthreads();
  for (int s = 128; s > 0; s >>= 1) {
    if (tid < s) buf[tid] += buf[tid + s];
    __syncthreads();
  }
  if (tid == 0) out[blockIdx.x] = buf[0];
}

constexpr int TILE = 32;
constexpr int ROWS = 8;

// out (width x height) is the transpose of in (height x width); launched with blockDim (32, 8) and
// gridDim (ceil(width / 32), ceil(height / 32)).
__global__ void __launch_bounds__(TILE * ROWS)
example_transpose(const float* __restrict__ in, float* __restrict__ out, int width, int height) {
  __shared__ float tile[TILE][TILE + 1];
  int x = blockIdx.x * TILE + threadIdx.x;
  int y = blockIdx.y * TILE + threadIdx.y;
  for (int j = 0; j < TILE; j += ROWS)
    if (x < width && y + j < height) tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * width + x];
  __syncthreads();
  x = blockIdx.y * TILE + threadIdx.x;
  y = blockIdx.x * TILE + threadIdx.y;
  for (int j = 0; j < TILE; j += ROWS)
    if (x < height && y + j < width) out[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
}
