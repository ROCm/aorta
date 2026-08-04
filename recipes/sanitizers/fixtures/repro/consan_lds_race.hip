// gfx950 ConSan repro: deliberate cross-wave LDS write/read race + a
// missing-wait load. Mirrors the upstream sanitizers.md minimal repro but
// compiled natively for gfx950 (MI350) so ConSan runs on real hardware.
#include <hip/hip_runtime.h>
#include <cstdint>

__global__ void lds_race(uint32_t *out) {
  __shared__ uint32_t value[64];
  const uint32_t tid = threadIdx.x;
  value[tid] = tid;
  // Wave 1 reads what wave 0 wrote, with no barrier between them.
  if (tid >= 32)
    out[tid] = value[tid - 32];
}

int main() {
  uint32_t *device = nullptr;
  if (hipMalloc(&device, 64 * sizeof(uint32_t)) != hipSuccess)
    return 1;
  if (hipMemset(device, 0, 64 * sizeof(uint32_t)) != hipSuccess)
    return 1;
  lds_race<<<1, 64>>>(device);
  const hipError_t result = hipDeviceSynchronize();
  const hipError_t cleanup = hipFree(device);
  return result == hipSuccess && cleanup == hipSuccess ? 0 : 1;
}
