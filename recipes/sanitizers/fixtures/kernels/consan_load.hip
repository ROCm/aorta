// Generic, self-contained load-only ConSan target: hipModuleLoad one code object
// (path baked in at build time via -DOBJECT="..."), so the combined rocjitsu ConSan
// hook instruments exactly that object at load time.
// No kernel dispatch (production StreamK GEMM kernels need hipBLASLt to launch); the
// interesting ConSan outcomes for large objects (timeout / report-buffer overflow) and
// small no-site objects occur at load. Used as `source.consan_command` in a recipe.
#include <hip/hip_runtime.h>
#include <cstdio>

#ifndef OBJECT
#define OBJECT "object.hsaco"
#endif

int main() {
    hipModule_t mod;
    hipError_t e = hipModuleLoad(&mod, OBJECT);
    if (e != hipSuccess) {
        fprintf(stderr, "hipModuleLoad(%s) failed: %s\n", OBJECT, hipGetErrorString(e));
        return 1;
    }
    printf("[consan_load] loaded+instrumented %s under the ConSan hook (no dispatch)\n", OBJECT);
    hipModuleUnload(mod);
    return 0;
}
