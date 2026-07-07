// HRX perf benchmark: bandwidth-bound STREAM triad (a[i] = b[i] + s*c[i]).
//
// Part of the `hrx_perf` aorta workload. A memory-bound counterpart to the
// GEMM bench: each timed iteration touches 3 * size * 4 bytes of HBM, so the
// reported GB/s reflects achieved bandwidth. Per-iteration time is measured
// host-side (steady_clock around launch + hipDeviceSynchronize) so it includes
// runtime/launch overhead -- the axis on which HRX can differ from stock HIP.
//
// Output (parsed by src/aorta/workloads/hrx_perf.py):
//   bench=triad size=<N> iters=<M> warmup=<W>
//   step_ms=<t>              (one per timed iteration)
//   GBPS=<x>                 (effective HBM bandwidth from the mean timed step)
//   checksum=<v> expected=<e>
//   RESULT=PERF_OK | RESULT=PERF_FAIL

#include <hip/hip_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#define BLOCK 256

#define HIP_CHECK(cmd)                                                        \
  do {                                                                        \
    hipError_t _e = (cmd);                                                    \
    if (_e != hipSuccess) {                                                   \
      printf("HIP_ERR %s at %s:%d\n", hipGetErrorString(_e), __FILE__,        \
             __LINE__);                                                       \
      printf("RESULT=PERF_FAIL\n");                                           \
      return 2;                                                               \
    }                                                                         \
  } while (0)

__global__ void fill(float* x, float v, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = v;
}

__global__ void triad(float* a, const float* b, const float* c, float s, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] = b[i] + s * c[i];
}

int main(int argc, char** argv) {
  size_t N = (argc > 1) ? (size_t)strtoull(argv[1], nullptr, 10) : 64000000ULL;
  int iters = (argc > 2) ? atoi(argv[2]) : 100;
  int warmup = (argc > 3) ? atoi(argv[3]) : 20;
  if (N == 0 || iters <= 0 || warmup < 0) {
    printf("bad args: size iters warmup must be positive\nRESULT=PERF_FAIL\n");
    return 2;
  }
  printf("bench=triad size=%zu iters=%d warmup=%d\n", N, iters, warmup);

  size_t bytes = N * sizeof(float);
  float *da = nullptr, *db = nullptr, *dc = nullptr;
  HIP_CHECK(hipMalloc(&da, bytes));
  HIP_CHECK(hipMalloc(&db, bytes));
  HIP_CHECK(hipMalloc(&dc, bytes));

  size_t grid = (N + BLOCK - 1) / BLOCK;
  const float scalar = 3.0f;
  fill<<<grid, BLOCK>>>(db, 1.0f, N);
  fill<<<grid, BLOCK>>>(dc, 2.0f, N);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  for (int i = 0; i < warmup; ++i) {
    triad<<<grid, BLOCK>>>(da, db, dc, scalar, N);
  }
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  double total_ms = 0.0;
  for (int i = 0; i < iters; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    triad<<<grid, BLOCK>>>(da, db, dc, scalar, N);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    total_ms += ms;
    printf("step_ms=%.4f\n", ms);
  }

  float a0 = 0.0f;
  HIP_CHECK(hipMemcpy(&a0, da, sizeof(float), hipMemcpyDeviceToHost));

  double mean_ms = total_ms / iters;
  // Triad moves 3 arrays per iteration: read b, read c, write a.
  double gbps = (3.0 * (double)N * sizeof(float)) / (mean_ms / 1e3) / 1e9;
  printf("GBPS=%.2f\n", gbps);
  float expected = 1.0f + scalar * 2.0f;  // 7.0
  printf("checksum=%.1f expected=%.1f\n", a0, expected);

  HIP_CHECK(hipFree(da));
  HIP_CHECK(hipFree(db));
  HIP_CHECK(hipFree(dc));

  bool ok = (fabsf(a0 - expected) < 1e-3f);
  printf("RESULT=%s\n", ok ? "PERF_OK" : "PERF_FAIL");
  return ok ? 0 : 1;
}
