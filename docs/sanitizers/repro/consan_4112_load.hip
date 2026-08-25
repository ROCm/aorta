// Minimal load-only ConSan target for the status=4112 transform rejection.
//
// hipModuleLoad is enough to reproduce: the rocjitsu ConSan hook instruments the
// object at load time, and the overlapping-anchor-patch validation failure happens
// during that transform, before anything is dispatched.
//
// Build with -DOBJECT="\"/abs/path/to/object.hsaco\"".
#include <hip/hip_runtime.h>
#include <cstdio>

#ifndef OBJECT
#define OBJECT "object.hsaco"
#endif

int main() {
    hipModule_t mod;
    hipError_t err = hipModuleLoad(&mod, OBJECT);
    if (err != hipSuccess) {
        fprintf(stderr, "hipModuleLoad(%s) failed: %s\n", OBJECT, hipGetErrorString(err));
        return 1;
    }
    printf("[consan_4112_load] loaded and instrumented %s (no dispatch)\n", OBJECT);
    (void)hipModuleUnload(mod);
    return 0;
}
