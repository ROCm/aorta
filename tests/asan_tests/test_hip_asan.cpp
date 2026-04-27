// test_hip_asan.cpp — Small HIP programs to verify ASAN instrumentation.
//
// Compile (inside the ASAN container):
//   $ROCM_HOME/llvm/bin/clang++ -fsanitize=address \
//       --offload-arch=$PYTORCH_ROCM_ARCH -x hip \
//       -I$ROCM_HOME/include -L$ROCM_HOME/lib -lamdhip64 \
//       -o test_hip_asan test_hip_asan.cpp
//
// Run:
//   ASAN_OPTIONS="detect_leaks=0:halt_on_error=0" ./test_hip_asan [test_name]
//
// Tests:
//   clean         — correct HIP code, should produce zero ASAN reports
//   heap_overflow — host heap-buffer-overflow, ASAN must catch it
//   use_after_free— host use-after-free, ASAN must catch it
//   event_query   — hipEventQuery stress test (the real debugging target)

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <thread>

#define HIP_CHECK(cmd)                                                       \
    do {                                                                     \
        hipError_t err = (cmd);                                              \
        if (err != hipSuccess) {                                             \
            fprintf(stderr, "HIP error: %s (%d) at %s:%d\n",                \
                    hipGetErrorString(err), err, __FILE__, __LINE__);        \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

// ---------------------------------------------------------------------------
// GPU kernel — simple vector add
// ---------------------------------------------------------------------------
__global__ void vec_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

// ---------------------------------------------------------------------------
// Test: clean — no memory errors expected
// ---------------------------------------------------------------------------
static int test_clean() {
    printf("=== test_clean: correct HIP code (expect 0 ASAN reports) ===\n");

    const int N = 1024;
    size_t bytes = N * sizeof(float);

    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) { h_a[i] = 1.0f; h_b[i] = 2.0f; }

    float *d_a, *d_b, *d_c;
    HIP_CHECK(hipMalloc(&d_a, bytes));
    HIP_CHECK(hipMalloc(&d_b, bytes));
    HIP_CHECK(hipMalloc(&d_c, bytes));

    HIP_CHECK(hipMemcpy(d_a, h_a, bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, h_b, bytes, hipMemcpyHostToDevice));

    vec_add<<<(N + 255) / 256, 256>>>(d_a, d_b, d_c, N);
    HIP_CHECK(hipGetLastError());

    HIP_CHECK(hipMemcpy(h_c, d_c, bytes, hipMemcpyDeviceToHost));

    int errors = 0;
    for (int i = 0; i < N; i++) {
        if (h_c[i] != 3.0f) { errors++; break; }
    }

    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_c));
    free(h_a); free(h_b); free(h_c);

    printf("test_clean: %s (compute result %s)\n",
           "PASSED", errors ? "WRONG" : "correct");
    return errors;
}

// ---------------------------------------------------------------------------
// Test: heap_overflow — intentional host heap-buffer-overflow
// ---------------------------------------------------------------------------
static int test_heap_overflow() {
    printf("=== test_heap_overflow: intentional OOB write (ASAN should report) ===\n");

    // Allocate 10 floats, write to index 10 (one past the end).
    float* buf = (float*)malloc(10 * sizeof(float));
    buf[10] = 42.0f;  // <-- heap-buffer-overflow
    printf("test_heap_overflow: wrote buf[10] = %f\n", buf[10]);
    free(buf);

    printf("test_heap_overflow: PASSED (ASAN should have printed a report above)\n");
    return 0;
}

// ---------------------------------------------------------------------------
// Test: use_after_free — intentional host use-after-free
// ---------------------------------------------------------------------------
static int test_use_after_free() {
    printf("=== test_use_after_free: intentional UAF read (ASAN should report) ===\n");

    float* buf = (float*)malloc(64 * sizeof(float));
    buf[0] = 99.0f;
    free(buf);

    // Read after free
    volatile float val = buf[0];  // <-- use-after-free
    printf("test_use_after_free: read buf[0] = %f after free\n", (float)val);

    printf("test_use_after_free: PASSED (ASAN should have printed a report above)\n");
    return 0;
}

// ---------------------------------------------------------------------------
// Test: event_query — hipEventQuery under ASAN (the real target)
//
// This exercises the host-side hipEventQuery path that we're trying to
// instrument.  If there are memory errors in the HIP runtime's event
// handling code, ASAN will catch them here.
// ---------------------------------------------------------------------------
static int test_event_query() {
    printf("=== test_event_query: hipEventQuery stress under ASAN ===\n");

    const int N = 1 << 20;  // 1M elements
    size_t bytes = N * sizeof(float);
    const int ITERATIONS = 20;

    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) { h_a[i] = 1.0f; h_b[i] = 2.0f; }

    float *d_a, *d_b, *d_c;
    HIP_CHECK(hipMalloc(&d_a, bytes));
    HIP_CHECK(hipMalloc(&d_b, bytes));
    HIP_CHECK(hipMalloc(&d_c, bytes));

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    int query_count = 0;

    for (int iter = 0; iter < ITERATIONS; iter++) {
        HIP_CHECK(hipEventRecord(start, stream));

        HIP_CHECK(hipMemcpyAsync(d_a, h_a, bytes, hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(d_b, h_b, bytes, hipMemcpyHostToDevice, stream));

        vec_add<<<(N + 255) / 256, 256, 0, stream>>>(d_a, d_b, d_c, N);
        HIP_CHECK(hipGetLastError());

        HIP_CHECK(hipMemcpyAsync(h_c, d_c, bytes, hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipEventRecord(stop, stream));

        // Poll hipEventQuery until the work is done — this is the code path
        // we want ASAN to instrument for memory-safety bugs.
        hipError_t status;
        do {
            status = hipEventQuery(stop);
            query_count++;
            if (status == hipErrorNotReady) {
                std::this_thread::sleep_for(std::chrono::microseconds(10));
            }
        } while (status == hipErrorNotReady);

        if (status != hipSuccess) {
            fprintf(stderr, "hipEventQuery returned unexpected error: %s\n",
                    hipGetErrorString(status));
            return 1;
        }
    }

    // Verify last iteration result
    int errors = 0;
    for (int i = 0; i < N; i++) {
        if (h_c[i] != 3.0f) { errors++; break; }
    }

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_c));
    free(h_a); free(h_b); free(h_c);

    printf("test_event_query: %d iterations, %d hipEventQuery polls, result %s\n",
           ITERATIONS, query_count, errors ? "WRONG" : "correct");
    return errors;
}

// ---------------------------------------------------------------------------
// Test: multi_stream — concurrent streams + events under ASAN
// ---------------------------------------------------------------------------
static int test_multi_stream() {
    printf("=== test_multi_stream: concurrent streams + events under ASAN ===\n");

    const int NUM_STREAMS = 4;
    const int N = 1 << 18;
    size_t bytes = N * sizeof(float);

    hipStream_t streams[NUM_STREAMS];
    hipEvent_t events[NUM_STREAMS];
    float *d_a[NUM_STREAMS], *d_b[NUM_STREAMS], *d_c[NUM_STREAMS];

    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) { h_a[i] = 1.0f; h_b[i] = 2.0f; }

    for (int s = 0; s < NUM_STREAMS; s++) {
        HIP_CHECK(hipStreamCreate(&streams[s]));
        HIP_CHECK(hipEventCreate(&events[s]));
        HIP_CHECK(hipMalloc(&d_a[s], bytes));
        HIP_CHECK(hipMalloc(&d_b[s], bytes));
        HIP_CHECK(hipMalloc(&d_c[s], bytes));
    }

    // Launch work on all streams
    for (int s = 0; s < NUM_STREAMS; s++) {
        HIP_CHECK(hipMemcpyAsync(d_a[s], h_a, bytes, hipMemcpyHostToDevice, streams[s]));
        HIP_CHECK(hipMemcpyAsync(d_b[s], h_b, bytes, hipMemcpyHostToDevice, streams[s]));
        vec_add<<<(N + 255) / 256, 256, 0, streams[s]>>>(d_a[s], d_b[s], d_c[s], N);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipEventRecord(events[s], streams[s]));
    }

    // Poll all events concurrently
    bool done[NUM_STREAMS] = {};
    int remaining = NUM_STREAMS;
    int polls = 0;
    while (remaining > 0) {
        for (int s = 0; s < NUM_STREAMS; s++) {
            if (done[s]) continue;
            hipError_t status = hipEventQuery(events[s]);
            polls++;
            if (status == hipSuccess) {
                done[s] = true;
                remaining--;
            } else if (status != hipErrorNotReady) {
                fprintf(stderr, "stream %d: unexpected error %s\n",
                        s, hipGetErrorString(status));
                return 1;
            }
        }
        if (remaining > 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    }

    for (int s = 0; s < NUM_STREAMS; s++) {
        HIP_CHECK(hipEventDestroy(events[s]));
        HIP_CHECK(hipStreamDestroy(streams[s]));
        HIP_CHECK(hipFree(d_a[s]));
        HIP_CHECK(hipFree(d_b[s]));
        HIP_CHECK(hipFree(d_c[s]));
    }
    free(h_a); free(h_b);

    printf("test_multi_stream: %d streams, %d total polls, PASSED\n",
           NUM_STREAMS, polls);
    return 0;
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    const char* test = (argc > 1) ? argv[1] : "all";

    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);
    printf("HIP devices: %d (hipGetDeviceCount: %s)\n",
           device_count, hipGetErrorString(err));

    bool need_gpu = true;

    if (strcmp(test, "heap_overflow") == 0) {
        return test_heap_overflow();
    } else if (strcmp(test, "use_after_free") == 0) {
        return test_use_after_free();
    } else if (strcmp(test, "host_only") == 0) {
        // Run only host-side ASAN tests (no GPU required)
        printf("\n--- Host-only ASAN tests (no GPU required) ---\n\n");
        test_heap_overflow();
        printf("\n");
        test_use_after_free();
        return 0;
    }

    // GPU-requiring tests
    if (device_count == 0) {
        fprintf(stderr, "No GPU found. Use 'host_only' to run host-side ASAN tests.\n");
        return 1;
    }

    if (strcmp(test, "clean") == 0) {
        return test_clean();
    } else if (strcmp(test, "event_query") == 0) {
        return test_event_query();
    } else if (strcmp(test, "multi_stream") == 0) {
        return test_multi_stream();
    } else if (strcmp(test, "all") == 0) {
        printf("\n--- Running all tests ---\n\n");
        int rc = 0;
        rc |= test_clean();         printf("\n");
        rc |= test_heap_overflow(); printf("\n");
        rc |= test_use_after_free();printf("\n");
        rc |= test_event_query();   printf("\n");
        rc |= test_multi_stream();
        printf("\n--- All tests done (rc=%d) ---\n", rc);
        return rc;
    } else {
        fprintf(stderr, "Unknown test: %s\n", test);
        fprintf(stderr, "Available: all, clean, heap_overflow, use_after_free, "
                        "event_query, multi_stream, host_only\n");
        return 1;
    }
}
