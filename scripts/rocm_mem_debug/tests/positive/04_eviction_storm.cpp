/*
 * P4: Memory Pressure Eviction Storm
 *
 * Fills ~95% of available VRAM with allocations, launches kernels on all
 * of them, then allocates more to trigger evictions.  Should generate
 * kfd_evict_process / kfd_restore_process tracepoint events.
 *
 * May or may not produce an illegal memory access -- the goal is to
 * trigger the eviction pattern that the debug script flags.
 */

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define HIP_CHECK(call)                                                    \
    do {                                                                   \
        hipError_t err = (call);                                           \
        if (err != hipSuccess) {                                           \
            fprintf(stderr, "HIP error %d (%s) at %s:%d\n",               \
                    err, hipGetErrorString(err), __FILE__, __LINE__);      \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

__global__ void fill_kernel(float* buf, int n, float val) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        buf[idx] = val;
    }
}

int main() {
    printf("[P4] Memory Pressure Eviction Storm test\n");

    size_t free_mem = 0, total_mem = 0;
    HIP_CHECK(hipMemGetInfo(&free_mem, &total_mem));
    printf("[P4] VRAM: %zu MB free / %zu MB total\n",
           free_mem / (1024 * 1024), total_mem / (1024 * 1024));

    const size_t CHUNK = 128 * 1024 * 1024;  // 128 MB per allocation
    size_t target = (size_t)(free_mem * 0.95);
    int n_chunks = (int)(target / CHUNK);

    printf("[P4] Allocating %d chunks of %zu MB to fill 95%% VRAM\n",
           n_chunks, CHUNK / (1024 * 1024));

    std::vector<float*> buffers;
    std::vector<hipStream_t> streams;

    for (int i = 0; i < n_chunks; i++) {
        float* buf = nullptr;
        hipError_t err = hipMalloc(&buf, CHUNK);
        if (err != hipSuccess) {
            printf("[P4] Allocation stopped at chunk %d (%s)\n",
                   i, hipGetErrorString(err));
            break;
        }
        buffers.push_back(buf);

        hipStream_t stream;
        HIP_CHECK(hipStreamCreate(&stream));
        streams.push_back(stream);
    }

    printf("[P4] Allocated %zu chunks, launching kernels...\n", buffers.size());

    int elems = CHUNK / sizeof(float);
    int threads = 256;
    int blocks = (elems + threads - 1) / threads;

    for (size_t i = 0; i < buffers.size(); i++) {
        hipLaunchKernelGGL(fill_kernel, dim3(blocks), dim3(threads),
                           0, streams[i % streams.size()],
                           buffers[i], elems, (float)i);
    }

    // Trigger evictions by allocating more
    printf("[P4] Triggering evictions with extra allocations...\n");
    std::vector<float*> extra;
    for (int i = 0; i < 5; i++) {
        float* buf = nullptr;
        hipError_t err = hipMalloc(&buf, CHUNK);
        if (err != hipSuccess) {
            printf("[P4] Extra alloc %d failed: %s\n", i, hipGetErrorString(err));
            break;
        }
        extra.push_back(buf);
        hipLaunchKernelGGL(fill_kernel, dim3(blocks), dim3(threads),
                           0, streams[0], buf, elems, 999.0f);
    }

    HIP_CHECK(hipDeviceSynchronize());
    printf("[P4] All kernels completed\n");

    // Cleanup
    for (auto* b : extra) hipFree(b);
    for (auto* b : buffers) hipFree(b);
    for (auto& s : streams) hipStreamDestroy(s);

    printf("[P4] Eviction storm test complete\n");
    return 0;
}
