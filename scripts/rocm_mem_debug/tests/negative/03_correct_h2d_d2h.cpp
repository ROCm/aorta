/*
 * N3: Correct H2D/D2H Copy Pattern
 *
 * Standard host-to-device and device-to-host copy with verification.
 * Should produce clean BO move events, no faults, no anomalies.
 */

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define HIP_CHECK(call)                                                    \
    do {                                                                   \
        hipError_t err = (call);                                           \
        if (err != hipSuccess) {                                           \
            fprintf(stderr, "HIP error %d (%s) at %s:%d\n",               \
                    err, hipGetErrorString(err), __FILE__, __LINE__);      \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

__global__ void double_kernel(float* buf, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        buf[idx] *= 2.0f;
    }
}

int main() {
    const int N = 512 * 1024;
    const size_t bytes = N * sizeof(float);

    printf("[N3] Correct H2D/D2H Copy Pattern test\n");

    // Pinned host memory
    float* h_buf = nullptr;
    float* h_out = nullptr;
    HIP_CHECK(hipHostMalloc(&h_buf, bytes));
    HIP_CHECK(hipHostMalloc(&h_out, bytes));

    // Fill with known pattern
    for (int i = 0; i < N; i++) h_buf[i] = (float)(i % 1000);

    // Device memory
    float* d_buf = nullptr;
    HIP_CHECK(hipMalloc(&d_buf, bytes));

    // H2D
    HIP_CHECK(hipMemcpy(d_buf, h_buf, bytes, hipMemcpyHostToDevice));

    // Compute
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    hipLaunchKernelGGL(double_kernel, dim3(blocks), dim3(threads), 0, 0,
                       d_buf, N);
    HIP_CHECK(hipDeviceSynchronize());

    // D2H
    HIP_CHECK(hipMemcpy(h_out, d_buf, bytes, hipMemcpyDeviceToHost));

    // Verify
    int errors = 0;
    for (int i = 0; i < N; i++) {
        float expected = (float)(i % 1000) * 2.0f;
        if (h_out[i] != expected) {
            errors++;
            if (errors <= 3) {
                printf("[N3] Mismatch at %d: expected %f, got %f\n",
                       i, expected, h_out[i]);
            }
        }
    }

    HIP_CHECK(hipFree(d_buf));
    hipHostFree(h_buf);
    hipHostFree(h_out);

    if (errors > 0) {
        printf("[N3] UNEXPECTED: %d errors\n", errors);
        return 1;
    }

    printf("[N3] Clean pass -- H2D/D2H copy verified\n");
    return 0;
}
