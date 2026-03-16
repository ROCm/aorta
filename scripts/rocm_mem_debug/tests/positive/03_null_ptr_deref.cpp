/*
 * P3: NULL Pointer Dereference
 *
 * Passes a nullptr device pointer to a kernel that writes to it.
 * Should produce an immediate GPU fault.
 */

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

#define HIP_CHECK(call)                                                    \
    do {                                                                   \
        hipError_t err = (call);                                           \
        if (err != hipSuccess) {                                           \
            fprintf(stderr, "HIP error %d (%s) at %s:%d\n",               \
                    err, hipGetErrorString(err), __FILE__, __LINE__);      \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

__global__ void null_deref_kernel(float* ptr) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    ptr[idx] = 42.0f;
}

int main() {
    float* d_ptr = nullptr;

    printf("[P3] NULL Pointer Dereference test\n");
    printf("[P3] Launching kernel with nullptr...\n");

    hipLaunchKernelGGL(null_deref_kernel, dim3(1), dim3(256), 0, 0, d_ptr);

    hipError_t sync_err = hipDeviceSynchronize();
    if (sync_err != hipSuccess) {
        printf("[P3] EXPECTED: hipDeviceSynchronize returned %d (%s)\n",
               sync_err, hipGetErrorString(sync_err));
    } else {
        printf("[P3] WARNING: No error detected\n");
    }

    return (sync_err != hipSuccess) ? 0 : 1;
}
