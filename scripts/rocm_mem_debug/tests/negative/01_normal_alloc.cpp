/*
 * N1: Normal Allocation and Access
 *
 * Standard HIP alloc -> kernel -> sync -> free workflow.
 * Should produce zero faults, zero evictions, zero anomalies.
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

__global__ void compute_kernel(float* buf, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        buf[idx] = buf[idx] * 2.0f + 1.0f;
    }
}

int main() {
    const int N = 1024 * 1024;
    const size_t bytes = N * sizeof(float);

    printf("[N1] Normal Allocation and Access test\n");

    float* d_buf = nullptr;
    float* h_buf = (float*)malloc(bytes);

    HIP_CHECK(hipMalloc(&d_buf, bytes));

    // Init on host
    for (int i = 0; i < N; i++) h_buf[i] = (float)i;

    HIP_CHECK(hipMemcpy(d_buf, h_buf, bytes, hipMemcpyHostToDevice));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    hipLaunchKernelGGL(compute_kernel, dim3(blocks), dim3(threads), 0, 0,
                       d_buf, N);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(h_buf, d_buf, bytes, hipMemcpyDeviceToHost));

    // Verify
    int errors = 0;
    for (int i = 0; i < N; i++) {
        float expected = (float)i * 2.0f + 1.0f;
        if (h_buf[i] != expected) errors++;
    }

    HIP_CHECK(hipFree(d_buf));
    free(h_buf);

    if (errors > 0) {
        printf("[N1] UNEXPECTED: %d computation errors\n", errors);
        return 1;
    }

    printf("[N1] Clean pass -- all values correct\n");
    return 0;
}
