/*
 * N2: Properly Synchronized Streams
 *
 * Two streams share a buffer, but stream B waits on an event recorded
 * by stream A before accessing it.  Should produce no faults.
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
        buf[idx] = val;
    }
}

__global__ void read_kernel(float* buf, int n, float* out) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        out[idx] = buf[idx] * 2.0f;
    }
}

int main() {
    const int N = 1024 * 1024;
    const size_t bytes = N * sizeof(float);

    printf("[N2] Properly Synchronized Streams test\n");

    float* d_shared = nullptr;
    float* d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_shared, bytes));
    HIP_CHECK(hipMalloc(&d_out, bytes));

    hipStream_t stream_a, stream_b;
    HIP_CHECK(hipStreamCreate(&stream_a));
    HIP_CHECK(hipStreamCreate(&stream_b));

    hipEvent_t sync_event;
    HIP_CHECK(hipEventCreate(&sync_event));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    for (int iter = 0; iter < 20; iter++) {
        // Stream A: write buffer, then record event
        hipLaunchKernelGGL(write_kernel, dim3(blocks), dim3(threads),
                           0, stream_a, d_shared, N, (float)iter);
        HIP_CHECK(hipEventRecord(sync_event, stream_a));

        // Stream B: wait for event, then read buffer
        HIP_CHECK(hipStreamWaitEvent(stream_b, sync_event, 0));
        hipLaunchKernelGGL(read_kernel, dim3(blocks), dim3(threads),
                           0, stream_b, d_shared, N, d_out);
    }

    HIP_CHECK(hipDeviceSynchronize());

    hipEventDestroy(sync_event);
    hipStreamDestroy(stream_a);
    hipStreamDestroy(stream_b);
    hipFree(d_shared);
    hipFree(d_out);

    printf("[N2] Clean pass -- all synchronized operations completed\n");
    return 0;
}
