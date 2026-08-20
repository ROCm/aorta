// gfx950 ConSan "clean" baseline: a single-wavefront LDS write/read that is
// deliberately NOT a race. gfx950 (CDNA4) has a 64-lane wavefront, so launching
// 64 threads in one block keeps every access inside one wave; the write and the
// later read are ordered within that wave, so ConSan should report NO hazard
// (daily-consan-clean.yaml expects PASS). Contrast the two-wave variant in
// consan_lds_race_2wave.hip, which launches 2 waves to force a real cross-wave
// race.
#include <hip/hip_runtime.h>
#include <cstdint>

__global__ void lds_race(uint32_t *out) {
  __shared__ uint32_t value[64];
  const uint32_t tid = threadIdx.x;
  value[tid] = tid;
  // Single 64-lane wave: this read is ordered after the write within the wave,
  // so there is no cross-wave hazard.
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
