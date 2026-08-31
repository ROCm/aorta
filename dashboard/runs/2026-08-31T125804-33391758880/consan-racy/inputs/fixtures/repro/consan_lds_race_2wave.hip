// gfx950 ConSan repro that triggers a REAL cross-wave LDS race.
//
// gfx950 (CDNA4) has a 64-lane wavefront. To create an inter-wave hazard we
// launch 128 threads = 2 wavefronts in one block and let wave 1 read LDS that
// wave 0 wrote, with NO __syncthreads() between them. That is an unsynchronized
// write/read on the same LDS bytes across waves -> ConSan should report it.
#include <hip/hip_runtime.h>
#include <cstdint>

__global__ void lds_race_2wave(uint32_t *out) {
  __shared__ uint32_t value[128];
  const uint32_t tid = threadIdx.x;
  value[tid] = tid;                 // every lane writes its own slot
  // No barrier here on purpose.
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
