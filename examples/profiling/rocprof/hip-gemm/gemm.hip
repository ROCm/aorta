// Tiled single-precision GEMM (C = A * B) for AMD GPUs.
//
// A deliberately small, dependency-free HIP payload for profiling examples:
// it needs nothing but `hipcc`, launches exactly one named kernel per
// iteration, verifies its own output, and prints one `key=value` line per
// result so a wrapper can regex the numbers out of stdout.
//
// Build and run standalone:
//   hipcc -O3 -o /tmp/hip_gemm gemm.hip
//   /tmp/hip_gemm 512 20
//
// Size and iteration count come from argv, falling back to the
// AORTA_GEMM_N / AORTA_GEMM_ITERS environment variables, falling back to
// small defaults that finish in well under a second. Keep it cheap: the
// point is to produce a clean profile, not a benchmark record.

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

#define HIP_CHECK(expr)                                                                 \
    do {                                                                                \
        hipError_t err_ = (expr);                                                        \
        if (err_ != hipSuccess) {                                                        \
            std::fprintf(stderr, "gemm: HIP error at %s:%d: %s\n", __FILE__, __LINE__,   \
                         hipGetErrorString(err_));                                       \
            return 1;                                                                    \
        }                                                                                \
    } while (0)

namespace {

constexpr int kTile = 16;
constexpr int kDefaultN = 512;
constexpr int kDefaultIters = 20;
constexpr int kWarmupIters = 3;
// Output elements re-computed on the host to validate the kernel. Full
// verification would be O(n^3) on one CPU thread; a fixed sample of exact
// dot products catches indexing and tiling bugs at O(samples * n).
constexpr int kCheckSamples = 64;
constexpr float kRelTolerance = 1e-3f;

// xorshift32: a deterministic, self-contained generator so every run of the
// example profiles the identical arithmetic (no <random> engine differences
// across toolchains).
unsigned int NextRandom(unsigned int& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

// Fills with values in [-1, 1) so accumulated sums stay well inside float
// range for the sizes this example uses.
void FillDeterministic(std::vector<float>& out, unsigned int seed) {
    unsigned int state = seed;
    for (float& v : out) {
        v = static_cast<float>(NextRandom(state) >> 8) / 8388608.0f - 1.0f;
    }
}

int ReadPositive(const char* text, int fallback) {
    if (text == nullptr) {
        return fallback;
    }
    const long parsed = std::strtol(text, nullptr, 10);
    if (parsed <= 0 || parsed > 16384) {
        return fallback;
    }
    return static_cast<int>(parsed);
}

}  // namespace

__global__ void sgemm_tiled(const float* __restrict__ a, const float* __restrict__ b,
                            float* __restrict__ c, int n) {
    __shared__ float a_tile[kTile][kTile];
    __shared__ float b_tile[kTile][kTile];

    const int row = blockIdx.y * kTile + threadIdx.y;
    const int col = blockIdx.x * kTile + threadIdx.x;

    float acc = 0.0f;
    for (int t = 0; t < (n + kTile - 1) / kTile; ++t) {
        const int a_col = t * kTile + threadIdx.x;
        const int b_row = t * kTile + threadIdx.y;
        a_tile[threadIdx.y][threadIdx.x] = (row < n && a_col < n) ? a[row * n + a_col] : 0.0f;
        b_tile[threadIdx.y][threadIdx.x] = (b_row < n && col < n) ? b[b_row * n + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < kTile; ++k) {
            acc += a_tile[threadIdx.y][k] * b_tile[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < n && col < n) {
        c[row * n + col] = acc;
    }
}

int main(int argc, char** argv) {
    const int n = ReadPositive(argc > 1 ? argv[1] : std::getenv("AORTA_GEMM_N"), kDefaultN);
    const int iters =
        ReadPositive(argc > 2 ? argv[2] : std::getenv("AORTA_GEMM_ITERS"), kDefaultIters);

    int device = 0;
    HIP_CHECK(hipGetDevice(&device));
    hipDeviceProp_t props{};
    HIP_CHECK(hipGetDeviceProperties(&props, device));
    std::printf("gemm: device=%d arch=%s\n", device, props.gcnArchName);
    std::printf("gemm: n=%d iters=%d tile=%d\n", n, iters, kTile);

    const size_t elements = static_cast<size_t>(n) * static_cast<size_t>(n);
    const size_t bytes = elements * sizeof(float);

    std::vector<float> host_a(elements);
    std::vector<float> host_b(elements);
    std::vector<float> host_c(elements);
    FillDeterministic(host_a, 0x9e3779b9u);
    FillDeterministic(host_b, 0x85ebca6bu);

    float* dev_a = nullptr;
    float* dev_b = nullptr;
    float* dev_c = nullptr;
    HIP_CHECK(hipMalloc(&dev_a, bytes));
    HIP_CHECK(hipMalloc(&dev_b, bytes));
    HIP_CHECK(hipMalloc(&dev_c, bytes));
    HIP_CHECK(hipMemcpy(dev_a, host_a.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dev_b, host_b.data(), bytes, hipMemcpyHostToDevice));

    const dim3 block(kTile, kTile);
    const dim3 grid((n + kTile - 1) / kTile, (n + kTile - 1) / kTile);

    for (int i = 0; i < kWarmupIters; ++i) {
        hipLaunchKernelGGL(sgemm_tiled, grid, block, 0, nullptr, dev_a, dev_b, dev_c, n);
    }
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t start;
    hipEvent_t stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start, nullptr));
    for (int i = 0; i < iters; ++i) {
        hipLaunchKernelGGL(sgemm_tiled, grid, block, 0, nullptr, dev_a, dev_b, dev_c, n);
    }
    HIP_CHECK(hipEventRecord(stop, nullptr));
    HIP_CHECK(hipEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));

    HIP_CHECK(hipMemcpy(host_c.data(), dev_c, bytes, hipMemcpyDeviceToHost));

    float max_rel_err = 0.0f;
    unsigned int probe_state = 0xc2b2ae35u;
    for (int s = 0; s < kCheckSamples; ++s) {
        const int row = static_cast<int>(NextRandom(probe_state) % static_cast<unsigned int>(n));
        const int col = static_cast<int>(NextRandom(probe_state) % static_cast<unsigned int>(n));
        double reference = 0.0;
        for (int k = 0; k < n; ++k) {
            reference += static_cast<double>(host_a[static_cast<size_t>(row) * n + k]) *
                         static_cast<double>(host_b[static_cast<size_t>(k) * n + col]);
        }
        const double actual = host_c[static_cast<size_t>(row) * n + col];
        const double scale = std::fmax(std::fabs(reference), 1.0);
        const double rel = std::fabs(actual - reference) / scale;
        // std::fmax returns its non-NaN operand, so folding a NaN sample
        // through it would discard the NaN and still print PASS. Propagate it
        // instead: the negated tolerance comparison below rejects a NaN.
        if (!std::isfinite(rel)) {
            max_rel_err = std::numeric_limits<float>::quiet_NaN();
            break;
        }
        max_rel_err = std::fmax(max_rel_err, static_cast<float>(rel));
    }

    const double mean_kernel_ms = elapsed_ms / static_cast<double>(iters);
    const double gflops =
        (2.0 * static_cast<double>(n) * n * n) / (mean_kernel_ms * 1.0e-3) / 1.0e9;
    std::printf("gemm: mean_kernel_ms=%.4f\n", mean_kernel_ms);
    std::printf("gemm: gflops=%.2f\n", gflops);
    std::printf("gemm: max_rel_err=%.3e\n", static_cast<double>(max_rel_err));

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipFree(dev_a));
    HIP_CHECK(hipFree(dev_b));
    HIP_CHECK(hipFree(dev_c));

    if (!(max_rel_err < kRelTolerance)) {
        std::printf("gemm: FAIL correctness (max_rel_err=%.3e >= %.3e)\n",
                    static_cast<double>(max_rel_err), static_cast<double>(kRelTolerance));
        return 1;
    }
    std::printf("gemm: PASS\n");
    return 0;
}
