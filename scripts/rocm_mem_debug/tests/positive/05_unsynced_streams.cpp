/*
 * P5: Unsynchronized Multi-Stream Access
 *
 * Two HIP streams concurrently access the same device buffer without
 * synchronization.  Stream A writes, Stream B reads -- with no event
 * or stream sync between them.  This is a data race that may or may
 * not produce a fault depending on HW queue mapping and timing.
 *
 * The debug script should detect concurrent BO access / VM mapping
 * events even if no explicit GPU fault occurs.
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

__global__ void write_kernel(float* buf, int n, float val) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        for (int i = 0; i < 100; i++) {
            buf[idx] = val + static_cast<float>(i);
        }
    }
}

__global__ void read_kernel(float* buf, int n, float* out) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        float sum = 0.0f;
        for (int i = 0; i < 100; i++) {
            sum += buf[idx];
        }
        out[idx] = sum;
    }
}

int main() {
    const int N = 1024 * 1024;
    const size_t bytes = N * sizeof(float);

    printf("[P5] Unsynchronized Multi-Stream Access test\n");

    float* d_shared = nullptr;
    float* d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_shared, bytes));
    HIP_CHECK(hipMalloc(&d_out, bytes));

    hipStream_t stream_a, stream_b;
    HIP_CHECK(hipStreamCreate(&stream_a));
    HIP_CHECK(hipStreamCreate(&stream_b));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    printf("[P5] Launching concurrent unsynchronized kernels...\n");

    for (int iter = 0; iter < 50; iter++) {
        // Stream A: write (no sync)
        hipLaunchKernelGGL(write_kernel, dim3(blocks), dim3(threads),
                           0, stream_a, d_shared, N, (float)iter);

        // Stream B: read the same buffer (no sync, no event wait)
        hipLaunchKernelGGL(read_kernel, dim3(blocks), dim3(threads),
                           0, stream_b, d_shared, N, d_out);
    }

    hipError_t sync_err = hipDeviceSynchronize();
    if (sync_err != hipSuccess) {
        printf("[P5] hipDeviceSynchronize error: %d (%s)\n",
               sync_err, hipGetErrorString(sync_err));
    } else {
        printf("[P5] Completed (race condition present but may not fault)\n");
    }

    hipStreamDestroy(stream_a);
    hipStreamDestroy(stream_b);
    hipFree(d_shared);
    hipFree(d_out);

    return 0;
}
