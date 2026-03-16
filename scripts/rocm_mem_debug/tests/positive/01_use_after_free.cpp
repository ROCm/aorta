/*
 * P1: Use-After-Free
 *
 * Allocates device memory, writes to it, frees it, then launches a
 * kernel that reads the freed pointer.  Should produce
 * hipErrorIllegalAddress and a GPU/VM fault in dmesg.
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

__global__ void write_kernel(float* buf, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        buf[idx] = static_cast<float>(idx);
    }
}

__global__ void read_kernel(float* buf, int n, float* out) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        out[idx] = buf[idx] * 2.0f;
    }
}

int main() {
    const int N = 256 * 1024;  // 1 MB of floats
    const size_t bytes = N * sizeof(float);

    float* d_buf = nullptr;
    float* d_out = nullptr;

    printf("[P1] Use-After-Free test\n");

    HIP_CHECK(hipMalloc(&d_buf, bytes));
    HIP_CHECK(hipMalloc(&d_out, bytes));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // Write to buffer
    hipLaunchKernelGGL(write_kernel, dim3(blocks), dim3(threads), 0, 0,
                       d_buf, N);
    HIP_CHECK(hipDeviceSynchronize());
    printf("[P1] Write kernel completed\n");

    // Free the buffer
    HIP_CHECK(hipFree(d_buf));
    printf("[P1] Buffer freed\n");

    // Read from freed buffer -- THIS SHOULD FAULT
    printf("[P1] Launching kernel on freed pointer...\n");
    hipLaunchKernelGGL(read_kernel, dim3(blocks), dim3(threads), 0, 0,
                       d_buf, N, d_out);

    hipError_t sync_err = hipDeviceSynchronize();
    if (sync_err != hipSuccess) {
        printf("[P1] EXPECTED: hipDeviceSynchronize returned %d (%s)\n",
               sync_err, hipGetErrorString(sync_err));
    } else {
        printf("[P1] WARNING: No error detected (UAF may not fault on this HW)\n");
    }

    hipFree(d_out);
    return (sync_err != hipSuccess) ? 0 : 1;
}
