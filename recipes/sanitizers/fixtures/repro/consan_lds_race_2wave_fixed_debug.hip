// Diagnostic build of consan_lds_race_2wave_fixed.hip.
//
// Identical kernel, but it reports which HIP call failed instead of collapsing
// everything into "return 1". Used to tell an instrumentation fault apart from a
// nonzero exit the ConSan hook produces on its own.
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

__global__ void lds_race_2wave(uint32_t *out) {
  __shared__ uint32_t value[128];
  const uint32_t tid = threadIdx.x;
  value[tid] = tid;
  __syncthreads();
  if (tid >= 64)
    out[tid] = value[tid - 64];
}

int main() {
  uint32_t *device = nullptr;
  hipError_t alloc = hipMalloc(&device, 128 * sizeof(uint32_t));
  fprintf(stderr, "[dbg] hipMalloc=%s\n", hipGetErrorString(alloc));

  hipError_t memset = hipMemset(device, 0, 128 * sizeof(uint32_t));
  fprintf(stderr, "[dbg] hipMemset=%s\n", hipGetErrorString(memset));

  lds_race_2wave<<<1, 128>>>(device);
  hipError_t launch = hipGetLastError();
  fprintf(stderr, "[dbg] launch=%s\n", hipGetErrorString(launch));

  hipError_t sync = hipDeviceSynchronize();
  fprintf(stderr, "[dbg] hipDeviceSynchronize=%s\n", hipGetErrorString(sync));

  hipError_t cleanup = hipFree(device);
  fprintf(stderr, "[dbg] hipFree=%s\n", hipGetErrorString(cleanup));

  fprintf(stderr, "[dbg] main returning 0 unconditionally\n");
  return 0;
}
