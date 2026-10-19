// Device AddressSanitizer repro: a kernel that stores one int past its
// hipMalloc'd buffer when built with -DASAN_REPRO_OVERFLOW=1, and stays in
// bounds (and checks its own output) otherwise.
//
// Both variants must be built for device ASan. The :xnack+ suffix is not
// optional: with a plain gfx950 target, -fsanitize=address is dropped for the
// GPU code with only a compiler warning, and the overflow then runs to exit 0.
//
//   hipcc --offload-arch=gfx950:xnack+ -fsanitize=address -shared-libsan -g -O1 \
//     asan_heap_overflow.hip -o ../bin/asan_clean
//   hipcc --offload-arch=gfx950:xnack+ -fsanitize=address -shared-libsan -g -O1 \
//     -DASAN_REPRO_OVERFLOW=1 asan_heap_overflow.hip -o ../bin/asan_heap_overflow
#include <hip/hip_runtime.h>

#include <cstdio>
#include <vector>

#ifndef ASAN_REPRO_OVERFLOW
#define ASAN_REPRO_OVERFLOW 0
#endif

// 4000 bytes: not a multiple of any allocation granularity, so the overflowing
// store lands in the redzone ASan places right after the requested size rather
// than in rounding slack.
constexpr int kElements = 1000;

__global__ void store_index(int* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n + ASAN_REPRO_OVERFLOW) out[i] = i;
}

int main() {
  int* device = nullptr;
  if (hipMalloc(&device, kElements * sizeof(int)) != hipSuccess) return 2;
  hipLaunchKernelGGL(store_index, dim3((kElements + 256) / 256), dim3(256), 0, 0, device,
                     kElements);
  hipError_t launched = hipGetLastError();
  hipError_t finished = hipDeviceSynchronize();
  if (launched != hipSuccess || finished != hipSuccess) {
    fprintf(stderr, "asan repro: kernel failed: %s\n",
            hipGetErrorString(launched != hipSuccess ? launched : finished));
    return 2;
  }
  // The clean variant proves the kernel really ran: a launch that silently did
  // nothing would otherwise read as a clean ASan pass.
  std::vector<int> host(kElements);
  if (hipMemcpy(host.data(), device, kElements * sizeof(int), hipMemcpyDeviceToHost) !=
      hipSuccess) {
    return 2;
  }
  for (int i = 0; i < kElements; ++i) {
    if (host[i] != i) {
      fprintf(stderr, "asan repro: element %d is %d, expected %d\n", i, host[i], i);
      return 3;
    }
  }
  hipFree(device);
  printf("asan repro: %d elements verified\n", kElements);
  return 0;
}
