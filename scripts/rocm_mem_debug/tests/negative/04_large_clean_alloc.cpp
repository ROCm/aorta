/*
 * N4: Large Clean Allocation
 *
 * Allocates a 2 GB buffer (under the 4 GB boundary), fills via kernel,
 * copies back, and verifies.  Should produce normal large BO move events
 * but no data corruption and no faults.
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

__global__ void fill_pattern(int* buf, long long n) {
    long long idx = (long long)threadIdx.x + (long long)blockIdx.x * blockDim.x;
    if (idx < n) {
        buf[idx] = (int)(idx & 0x7FFFFFFF);
    }
}

int main() {
    printf("[N4] Large Clean Allocation test\n");

    size_t free_mem = 0, total_mem = 0;
    HIP_CHECK(hipMemGetInfo(&free_mem, &total_mem));
    printf("[N4] VRAM: %zu MB free / %zu MB total\n",
           free_mem / (1024 * 1024), total_mem / (1024 * 1024));

    // 2 GB allocation (stays under 4 GB boundary)
    size_t alloc_size = 2ULL * 1024 * 1024 * 1024;
    if (alloc_size > free_mem * 0.7) {
        alloc_size = 512ULL * 1024 * 1024;  // Fall back to 512 MB
        printf("[N4] Reduced allocation to %zu MB due to VRAM limits\n",
               alloc_size / (1024 * 1024));
    }

    long long n_ints = alloc_size / sizeof(int);

    // Device buffer
    int* d_buf = nullptr;
    HIP_CHECK(hipMalloc(&d_buf, alloc_size));

    // Fill on device
    int threads = 256;
    long long blocks = (n_ints + threads - 1) / threads;
    printf("[N4] Filling %zu MB on device...\n", alloc_size / (1024 * 1024));
    hipLaunchKernelGGL(fill_pattern, dim3((unsigned int)blocks), dim3(threads),
                       0, 0, d_buf, n_ints);
    HIP_CHECK(hipDeviceSynchronize());

    // Copy back
    int* h_buf = (int*)malloc(alloc_size);
    if (!h_buf) {
        printf("[N4] Host malloc failed for %zu bytes\n", alloc_size);
        hipFree(d_buf);
        return 1;
    }

    printf("[N4] Copying D2H...\n");
    HIP_CHECK(hipMemcpy(h_buf, d_buf, alloc_size, hipMemcpyDeviceToHost));

    // Spot-check verification (full check is very slow for 2 GB)
    printf("[N4] Verifying...\n");
    int errors = 0;
    long long check_stride = n_ints / 10000;
    if (check_stride < 1) check_stride = 1;

    for (long long i = 0; i < n_ints; i += check_stride) {
        int expected = (int)(i & 0x7FFFFFFF);
        if (h_buf[i] != expected) {
            errors++;
            if (errors <= 3) {
                printf("[N4] Mismatch at %lld: expected %d, got %d\n",
                       i, expected, h_buf[i]);
            }
        }
    }

    hipFree(d_buf);
    free(h_buf);

    if (errors > 0) {
        printf("[N4] UNEXPECTED: %d errors\n", errors);
        return 1;
    }

    printf("[N4] Clean pass -- large allocation verified\n");
    return 0;
}
