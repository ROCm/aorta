// Self-contained HIP program that DISPATCHES a race-free LDS reduction kernel from
// an explicitly hipModuleLoad'ed code object, so the rocjitsu ConSan hook
// instruments the very module we launch and record/replay captures real runtime
// LDS accesses (dynamic_complete=true). Using hipModuleLaunchKernel (not static
// hipLaunchKernelGGL) is deliberate: the static fat-binary path did not route
// through the instrumented module, so 0 records were captured -> strict exit 86.
#include <hip/hip_runtime.h>
#include <cstdio>

// The code object to load/dispatch is provided at build time via -DLDS_HSACO=...
// (the CI fixture-build step passes the absolute path of the unbundled lds.hsaco).
#ifndef LDS_HSACO
#define LDS_HSACO "lds.hsaco"
#endif

static bool ck(hipError_t e, const char* what) {
    if (e != hipSuccess) {
        fprintf(stderr, "FATAL %s: %s\n", what, hipGetErrorString(e));
        return false;
    }
    return true;
}

int main() {
    const int N = 4096;
    const int BLOCK = 256;
    const int GRID = (N + BLOCK - 1) / BLOCK;

    hipModule_t mod;
    if (!ck(hipModuleLoad(&mod, LDS_HSACO), "hipModuleLoad")) return 1;
    hipFunction_t func;
    if (!ck(hipModuleGetFunction(&func, mod, "lds_reduce"), "hipModuleGetFunction")) return 1;

    float* d_in = nullptr;
    float* d_out = nullptr;
    if (!ck(hipMalloc(&d_in, N * sizeof(float)), "hipMalloc in")) return 1;
    if (!ck(hipMalloc(&d_out, GRID * sizeof(float)), "hipMalloc out")) return 1;
    ck(hipMemset(d_in, 0, N * sizeof(float)), "hipMemset");

    struct {
        const float* in;
        float* out;
        int n;
    } args = {d_in, d_out, N};
    size_t arg_size = sizeof(args);
    void* config[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
        HIP_LAUNCH_PARAM_BUFFER_SIZE, &arg_size,
        HIP_LAUNCH_PARAM_END,
    };

    // A few dispatches so record/replay has multiple generations to observe.
    for (int iter = 0; iter < 4; ++iter) {
        if (!ck(hipModuleLaunchKernel(func, GRID, 1, 1, BLOCK, 1, 1, 0, 0, nullptr, config),
                "hipModuleLaunchKernel")) {
            return 1;
        }
    }
    if (!ck(hipDeviceSynchronize(), "hipDeviceSynchronize")) return 1;

    printf("[lds_dispatch] module-launched lds_reduce grid=%d block=%d n=%d OK\n", GRID, BLOCK, N);
    hipFree(d_in);
    hipFree(d_out);
    hipModuleUnload(mod);
    return 0;
}
