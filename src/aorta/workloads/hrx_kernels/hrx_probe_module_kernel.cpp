// Device kernel for the dynamic-module-launch repro.
// Compiled to a SEPARATE code object via `hipcc --genco --offload-arch=gfx942`.
// extern "C" keeps the symbol name "add100" for hipModuleGetFunction.
#include <hip/hip_runtime.h>

extern "C" __global__ void add100(float* out, const float* in, unsigned long n) {
  unsigned long i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[i] + 100.0f;
}
