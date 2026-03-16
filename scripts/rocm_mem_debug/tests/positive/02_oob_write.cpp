/*
 * P2: Out-of-Bounds Write
 *
 * Allocates a buffer of N elements and launches a kernel with enough
 * threads to write far beyond the allocation.  Should trigger a GPU/VM
 * fault on OOB access.
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

__global__ void oob_kernel(float* buf, int alloc_n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // Intentionally write beyond alloc_n
    buf[idx] = static_cast<float>(idx);
}

int main() {
    const int ALLOC_N = 1024;
    const int LAUNCH_N = 1024 * 1024;  // 1000x the allocation
    const size_t alloc_bytes = ALLOC_N * sizeof(float);

    float* d_buf = nullptr;

    printf("[P2] Out-of-Bounds Write test\n");

    HIP_CHECK(hipMalloc(&d_buf, alloc_bytes));
    printf("[P2] Allocated %zu bytes (%d floats)\n", alloc_bytes, ALLOC_N);

    int threads = 256;
    int blocks = (LAUNCH_N + threads - 1) / threads;

    printf("[P2] Launching %d threads on %d-element buffer...\n",
           blocks * threads, ALLOC_N);
    hipLaunchKernelGGL(oob_kernel, dim3(blocks), dim3(threads), 0, 0,
                       d_buf, ALLOC_N);

    hipError_t sync_err = hipDeviceSynchronize();
    if (sync_err != hipSuccess) {
        printf("[P2] EXPECTED: hipDeviceSynchronize returned %d (%s)\n",
               sync_err, hipGetErrorString(sync_err));
    } else {
        printf("[P2] WARNING: No error detected (OOB may not fault on this HW)\n");
    }

    hipFree(d_buf);
    return (sync_err != hipSuccess) ? 0 : 1;
}
