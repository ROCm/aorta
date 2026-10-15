// Minimal single-kernel code object to observe ConSan behavior on a small,
// clean object (contrast with the 16-20 MB production Tensile libraries).
#include <hip/hip_runtime.h>

extern "C" __global__ void tiny_vecadd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
