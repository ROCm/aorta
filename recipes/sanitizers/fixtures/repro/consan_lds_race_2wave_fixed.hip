// Fixed counterpart of consan_lds_race_2wave.hip.
//
// The racy version lets wave 1 read LDS slots wave 0 wrote with nothing
// ordering the two, so ConSan reports a write/read conflict on every lane.
// __syncthreads() is a workgroup-wide barrier: it holds all 128 threads until
// every LDS write has landed, which is what supplies the missing edge between
// wave 0's store and wave 1's load.
#include <hip/hip_runtime.h>
#include <cstdint>

__global__ void lds_race_2wave(uint32_t *out) {
  __shared__ uint32_t value[128];
  const uint32_t tid = threadIdx.x;
  value[tid] = tid;                 // every lane writes its own slot
  __syncthreads();                  // orders wave 0's writes before wave 1's reads
  if (tid >= 64)                    // wave 1 reads slots written by wave 0
    out[tid] = value[tid - 64];
}

int main() {
  uint32_t *device = nullptr;
  if (hipMalloc(&device, 128 * sizeof(uint32_t)) != hipSuccess)
    return 1;
  if (hipMemset(device, 0, 128 * sizeof(uint32_t)) != hipSuccess)
    return 1;
  lds_race_2wave<<<1, 128>>>(device);   // 1 block, 128 threads = 2 waves
  const hipError_t result = hipDeviceSynchronize();
  const hipError_t cleanup = hipFree(device);
  return result == hipSuccess && cleanup == hipSuccess ? 0 : 1;
}
