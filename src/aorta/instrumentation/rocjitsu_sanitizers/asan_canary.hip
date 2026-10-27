// The node-health canary of `aorta run --sanitizer asan`.
//
// Device ASan reads its shadow memory, which is ordinary anonymous host memory,
// through the GPU's XNACK page faults, and the driver may migrate those pages to
// VRAM and back. On a node where that migration corrupts data, every check reads
// a garbage shadow: a run reports memory errors that did not happen, and can miss
// ones that did. So on every visible GPU this sends known data down the same
// path, through fresh MAP_NORESERVE anonymous mappings without huge pages, as
// ASan's shadow is mapped. Every access is in bounds, and the program is built
// with device ASan, so on a sound node it reports nothing and exits 0.
//
// Prints "AORTA_ASAN_CANARY gpu=<n> check=<name> bad_words=<k> of=<m> ok|FAIL"
// per check, then "AORTA_ASAN_CANARY verdict=PASS|FAIL gpus=<n>". Exits 3 when a
// check failed and 2 on a HIP error.
#include <hip/hip_runtime.h>
#include <sys/mman.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define CHECK(call)                                                                      \
  do {                                                                                   \
    hipError_t status_ = (call);                                                         \
    if (status_ != hipSuccess) {                                                         \
      printf("AORTA_ASAN_CANARY hip_error %s: %s\n", #call, hipGetErrorString(status_)); \
      fflush(stdout);                                                                    \
      exit(2);                                                                           \
    }                                                                                    \
  } while (0)

constexpr size_t kBytes = size_t{64} << 20;
constexpr size_t kWords = kBytes / sizeof(uint64_t);
constexpr size_t kWordsPerPage = 4096 / sizeof(uint64_t);
constexpr uint64_t kSparseWord = 0xA5A5A5A5A5A5A5A5ull;

__host__ __device__ inline uint64_t pattern(uint64_t seed, uint64_t i) {
  uint64_t x = seed ^ (i * 0x9E3779B97F4A7C15ull);
  x ^= x >> 31;
  x *= 0xBF58476D1CE4E5B9ull;
  x ^= x >> 29;
  return x | 1;
}

__global__ void aorta_canary_count_nonzero(const uint64_t* p, size_t n, unsigned long long* bad) {
  unsigned long long found = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    found += p[i] != 0;
  if (found) atomicAdd(bad, found);
}

__global__ void aorta_canary_count_mismatch(const uint64_t* p, size_t n, uint64_t seed,
                                            unsigned long long* bad) {
  unsigned long long found = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    found += p[i] != pattern(seed, i);
  if (found) atomicAdd(bad, found);
}

__global__ void aorta_canary_write_pattern(uint64_t* p, size_t n, uint64_t seed) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    p[i] = pattern(seed, i);
}

// One word per 4 KiB page: a page the GPU wrote only partly.
__global__ void aorta_canary_write_sparse(uint64_t* p, size_t pages) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < pages; i += (size_t)gridDim.x * blockDim.x)
    p[i * kWordsPerPage] = kSparseWord;
}

static uint64_t* fresh_mapping() {
  void* p = mmap(nullptr, kBytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  if (p == MAP_FAILED) {
    perror("AORTA_ASAN_CANARY mmap");
    exit(2);
  }
  madvise(p, kBytes, MADV_NOHUGEPAGE);
  return static_cast<uint64_t*>(p);
}

static int report(int gpu, const char* check, unsigned long long bad) {
  printf("AORTA_ASAN_CANARY gpu=%d check=%s bad_words=%llu of=%zu %s\n", gpu, check, bad, kWords,
         bad ? "FAIL" : "ok");
  fflush(stdout);
  return bad ? 1 : 0;
}

static unsigned long long counted(unsigned long long* bad) {
  unsigned long long host = 0;
  CHECK(hipGetLastError());
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(&host, bad, sizeof host, hipMemcpyDeviceToHost));
  CHECK(hipMemset(bad, 0, sizeof *bad));
  return host;
}

int main() {
  int gpus = 0;
  CHECK(hipGetDeviceCount(&gpus));
  if (gpus < 1) {
    printf("AORTA_ASAN_CANARY hip_error no GPU is visible\n");
    return 2;
  }
  int failures = 0;
  for (int gpu = 0; gpu < gpus; ++gpu) {
    CHECK(hipSetDevice(gpu));
    int pageable = 0;
    CHECK(hipDeviceGetAttribute(&pageable, hipDeviceAttributePageableMemoryAccess, gpu));
    if (!pageable) {
      printf("AORTA_ASAN_CANARY gpu=%d check=pageable_memory_access FAIL\n", gpu);
      ++failures;
      continue;
    }
    unsigned long long* bad = nullptr;
    CHECK(hipMalloc(&bad, sizeof *bad));
    CHECK(hipMemset(bad, 0, sizeof *bad));

    // Pages nothing has written read as zero on the GPU, and on the CPU after the GPU has
    // touched them. Most of ASan's shadow is such pages.
    uint64_t* p = fresh_mapping();
    aorta_canary_count_nonzero<<<1024, 256>>>(p, kWords, bad);
    failures += report(gpu, "untouched_gpu_read", counted(bad));
    unsigned long long found = 0;
    for (size_t i = 0; i < kWords; ++i) found += p[i] != 0;
    failures += report(gpu, "untouched_cpu_after", found);

    for (size_t i = 0; i < kWords; ++i) p[i] = pattern(1, i);
    aorta_canary_count_mismatch<<<1024, 256>>>(p, kWords, 1, bad);
    failures += report(gpu, "cpu_pattern_gpu", counted(bad));
    aorta_canary_write_pattern<<<1024, 256>>>(p, kWords, 2);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    found = 0;
    for (size_t i = 0; i < kWords; ++i) found += p[i] != pattern(2, i);
    failures += report(gpu, "gpu_pattern_cpu", found);
    munmap(p, kBytes);

    p = fresh_mapping();
    aorta_canary_write_sparse<<<1024, 256>>>(p, kWords / kWordsPerPage);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    found = 0;
    for (size_t i = 0; i < kWords; ++i) found += p[i] != (i % kWordsPerPage ? 0 : kSparseWord);
    failures += report(gpu, "sparse_gpu_write", found);
    munmap(p, kBytes);
    CHECK(hipFree(bad));
  }
  printf("AORTA_ASAN_CANARY verdict=%s gpus=%d\n", failures ? "FAIL" : "PASS", gpus);
  return failures ? 3 : 0;
}
