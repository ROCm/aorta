// HRX perf benchmark: compute-bound tiled SGEMM (C = A * B, float, N x N).
//
// Part of the `hrx_perf` aorta workload. Runs `warmup` untimed iterations then
// `iters` timed iterations; each timed iteration is bracketed by a host-side
// steady_clock around the launch + hipDeviceSynchronize, so the reported
// per-step time includes kernel execution AND host/runtime launch overhead --
// the latter is where an alternate HIP runtime (HRX) can differ from stock HIP.
//
// Output (parsed by src/aorta/workloads/hrx_perf.py):
//   bench=gemm size=<N> iters=<M> warmup=<W>
//   step_ms=<t>              (one per timed iteration)
//   GFLOPS=<x>               (from the mean timed step)
//   checksum=<v> expected=<e>
//   RESULT=PERF_OK | RESULT=PERF_FAIL
//
// A=1.0, B=1.0 so every C[i] == N; the checksum guards against a silently
// wrong result (e.g. a broken runtime) turning a perf number meaningless.

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define TILE 16

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

__global__ void sgemm_tiled(const float* A, const float* B, float* C, int N) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];
  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;
  float acc = 0.0f;
  for (int t = 0; t < N; t += TILE) {
    As[threadIdx.y][threadIdx.x] =
        (row < N && t + threadIdx.x < N) ? A[row * N + t + threadIdx.x] : 0.0f;
    Bs[threadIdx.y][threadIdx.x] =
        (col < N && t + threadIdx.y < N) ? B[(t + threadIdx.y) * N + col] : 0.0f;
    __syncthreads();
    for (int k = 0; k < TILE; ++k) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    __syncthreads();
  }
  if (row < N && col < N) C[row * N + col] = acc;
}

int main(int argc, char** argv) {
  int N = (argc > 1) ? atoi(argv[1]) : 4096;
  int iters = (argc > 2) ? atoi(argv[2]) : 50;
  int warmup = (argc > 3) ? atoi(argv[3]) : 10;
  if (N <= 0 || iters <= 0 || warmup < 0) {
    printf("bad args: size iters warmup must be positive\nRESULT=PERF_FAIL\n");
    return 2;
  }
  printf("bench=gemm size=%d iters=%d warmup=%d\n", N, iters, warmup);

  size_t elems = (size_t)N * (size_t)N;
  size_t bytes = elems * sizeof(float);

  std::vector<float> hA(elems, 1.0f), hB(elems, 1.0f);
  float *dA = nullptr, *dB = nullptr, *dC = nullptr;
  HIP_CHECK(hipMalloc(&dA, bytes));
  HIP_CHECK(hipMalloc(&dB, bytes));
  HIP_CHECK(hipMalloc(&dC, bytes));
  HIP_CHECK(hipMemcpy(dA, hA.data(), bytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dB, hB.data(), bytes, hipMemcpyHostToDevice));

  dim3 block(TILE, TILE);
  dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

  for (int i = 0; i < warmup; ++i) {
    sgemm_tiled<<<grid, block>>>(dA, dB, dC, N);
  }
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  double total_ms = 0.0;
  for (int i = 0; i < iters; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    sgemm_tiled<<<grid, block>>>(dA, dB, dC, N);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    total_ms += ms;
    printf("step_ms=%.4f\n", ms);
  }

  std::vector<float> hC(elems);
  HIP_CHECK(hipMemcpy(hC.data(), dC, bytes, hipMemcpyDeviceToHost));

  double mean_ms = total_ms / iters;
  double gflops = (2.0 * (double)N * (double)N * (double)N) / (mean_ms / 1e3) / 1e9;
  printf("GFLOPS=%.2f\n", gflops);
  printf("checksum=%.1f expected=%d\n", hC[0], N);

  HIP_CHECK(hipFree(dA));
  HIP_CHECK(hipFree(dB));
  HIP_CHECK(hipFree(dC));

  bool ok = (hC[0] == (float)N);
  printf("RESULT=%s\n", ok ? "PERF_OK" : "PERF_FAIL");
  return ok ? 0 : 1;
}
