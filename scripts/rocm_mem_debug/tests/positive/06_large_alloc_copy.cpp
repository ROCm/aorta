/*
 * P6: Large Allocation Copy Boundary
 *
 * Allocates 5 GB on device and host, copies data back and forth, and
 * verifies all bytes.  Targets a known ROCm bug on MI300X where copies
 * >4 GB can produce data mismatches.
 *
 * Falls back to 2 GB if 5 GB is unavailable.  Debug script should
 * capture large BO move events and migration volume.
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

int main() {
    printf("[P6] Large Allocation Copy Boundary test\n");

    size_t free_mem = 0, total_mem = 0;
    HIP_CHECK(hipMemGetInfo(&free_mem, &total_mem));
    printf("[P6] VRAM: %zu MB free / %zu MB total\n",
           free_mem / (1024 * 1024), total_mem / (1024 * 1024));

    // Try 5 GB, fall back to 2 GB
    size_t target_gb = 5;
    size_t alloc_size = target_gb * 1024ULL * 1024ULL * 1024ULL;
    if (alloc_size > free_mem * 0.8) {
        target_gb = 2;
        alloc_size = target_gb * 1024ULL * 1024ULL * 1024ULL;
        if (alloc_size > free_mem * 0.8) {
            target_gb = 1;
            alloc_size = target_gb * 1024ULL * 1024ULL * 1024ULL;
        }
    }
    printf("[P6] Testing with %zu GB allocation\n", target_gb);

    // Device buffer
    char* d_buf = nullptr;
    HIP_CHECK(hipMalloc(&d_buf, alloc_size));

    // Host buffer (pinned)
    char* h_buf = nullptr;
    HIP_CHECK(hipHostMalloc(&h_buf, alloc_size));

    // Host verification buffer
    char* h_verify = nullptr;
    HIP_CHECK(hipHostMalloc(&h_verify, alloc_size));

    // Fill host with known pattern
    printf("[P6] Filling host buffer with pattern...\n");
    for (size_t i = 0; i < alloc_size; i++) {
        h_buf[i] = (char)(i & 0xFF);
    }

    // H2D copy
    printf("[P6] Copying H2D (%zu GB)...\n", target_gb);
    HIP_CHECK(hipMemcpy(d_buf, h_buf, alloc_size, hipMemcpyHostToDevice));

    // D2H copy
    printf("[P6] Copying D2H (%zu GB)...\n", target_gb);
    memset(h_verify, 0, alloc_size);
    HIP_CHECK(hipMemcpy(h_verify, d_buf, alloc_size, hipMemcpyDeviceToHost));

    // Verify
    printf("[P6] Verifying data integrity...\n");
    int mismatches = 0;
    size_t first_mismatch = 0;
    for (size_t i = 0; i < alloc_size; i++) {
        if (h_verify[i] != h_buf[i]) {
            if (mismatches == 0) first_mismatch = i;
            mismatches++;
            if (mismatches <= 5) {
                printf("[P6] MISMATCH at offset %zu: expected 0x%02X, got 0x%02X\n",
                       i, (unsigned char)h_buf[i], (unsigned char)h_verify[i]);
            }
        }
    }

    if (mismatches > 0) {
        printf("[P6] DATA CORRUPTION: %d mismatches, first at offset %zu (%.2f GB)\n",
               mismatches, first_mismatch, first_mismatch / (1024.0 * 1024.0 * 1024.0));
    } else {
        printf("[P6] All data verified OK\n");
    }

    hipHostFree(h_verify);
    hipHostFree(h_buf);
    hipFree(d_buf);

    return (mismatches > 0) ? 0 : 1;
}
