# TheRock HOST_ASAN + PyTorch Docker Build Issues & Resolutions

This document tracks all build issues encountered while creating
`Dockerfile.therock-host-asan-pytorch` — a Docker image that builds ROCm/HIP
from source via [TheRock](https://github.com/ROCm/TheRock) with host-only
AddressSanitizer (HOST_ASAN), then builds PyTorch against that instrumented stack.

Reference Dockerfile: `docker/Dockerfile.rocm70_2-ubuntu-pytorch` (working
PyTorch + ROCm build using pre-built debian packages).

---

## Issue 1: Missing ROCm path environment variables

**Symptom:** PyTorch's CMake could not find ROCm components (rocblas, hipblas,
miopen, etc.) during configuration, even though `ROCM_HOME=/opt/rocm` was set.

**Root cause:** The working Dockerfile installs ROCm via debian packages that
ship CMake config files and pkg-config entries. TheRock's `cmake --install`
produces a different layout. PyTorch's `find_package()` calls need
`CMAKE_PREFIX_PATH` to locate components, and `HIP_PATH` / `ROCM_PATH` are
also consulted by various FindHIP/FindROCm scripts.

**Resolution:** Added three additional environment variables alongside `ROCM_HOME`:

```dockerfile
ENV ROCM_PATH=${ROCM_INSTALL_PREFIX}
ENV HIP_PATH=${ROCM_INSTALL_PREFIX}
ENV CMAKE_PREFIX_PATH=${ROCM_INSTALL_PREFIX}
```

---

## Issue 2: pip cannot uninstall debian-managed pip on Ubuntu 24.04

**Symptom:**
```
ERROR: Cannot uninstall pip 24.0, RECORD file not found.
Hint: The package was installed by debian.
```

**Root cause:** Ubuntu 24.04 enforces PEP 668 — system-installed pip has no
RECORD file, so `pip install --upgrade pip` fails when trying to uninstall
the existing version. Using `--break-system-packages` doesn't help with this
specific error.

**Resolution:** Use a Python venv for PyTorch build dependencies. The venv
has its own pip that can be freely upgraded:

```dockerfile
RUN python3 -m venv /opt/pytorch-venv && \
    /opt/pytorch-venv/bin/pip install --upgrade pip setuptools wheel && \
    /opt/pytorch-venv/bin/pip install --no-cache-dir pyyaml typing_extensions numpy pandas

ENV PATH=/opt/pytorch-venv/bin:${PATH}
```

---

## Issue 3: AOTriton configure fails — missing liblzma

**Symptom:**
```
CMake Error at /usr/share/cmake-4.2/Modules/FindPkgConfig.cmake:1203 (message):
    None of the required 'liblzma' found
FAILED: aotriton/src/aotriton_runtime-stamp/aotriton_runtime-configure
```

**Root cause:** PyTorch v2.11 configures the `aotriton_runtime` ExternalProject
even when `DISABLE_AOTRITON=1` is set (the flag only disables runtime usage,
not the build configure step). AOTriton's CMakeLists.txt requires `liblzma`
via `pkg_search_module(liblzma)`.

**Resolution:** Added `liblzma-dev` to the apt package list:

```dockerfile
RUN apt-get update && apt-get install -y \
    ...
    liblzma-dev \
    ...
```

---

## Issue 4: TheRock rocRoller link failure — ASAN symbol mismatch

**Symptom:**
```
[rocRoller] ld.lld: error: undefined symbol: __asan_option_detect_stack_use_after_return
>>> referenced by ... in archive /build/TheRock/build/third-party/yaml-cpp/dist/lib/libyaml-cpp.a
[rocRoller] ld.lld: error: undefined symbol: __asan_stack_malloc_4
[rocRoller] ld.lld: error: undefined symbol: __asan_set_shadow_f5
```

**Root cause:** The `linux-release-host-asan` preset sets
`THEROCK_SANITIZER=HOST_ASAN` globally, which adds `-fsanitize=address` to
the global `CMAKE_C_FLAGS` / `CMAKE_CXX_FLAGS`. This affects ALL compiled
code — including third-party dependencies like `yaml-cpp`.

We attempted to selectively disable ASAN on individual components (rocRoller,
composable_kernel, etc.) via `-D<component>_SANITIZER=OFF`. However,
per-component `_SANITIZER=OFF` only removes ASAN from TheRock's own managed
sub-projects' build flags. Third-party static libraries like `libyaml-cpp.a`
always inherit the global ASAN flags and are compiled with ASAN symbols.

When rocRoller (SANITIZER=OFF, no `-fsanitize=address` in link flags) tried to
link against the ASAN-instrumented `libyaml-cpp.a`, the ASAN runtime symbols
were unresolvable — the linker doesn't pull in `libclang_rt.asan` without
`-fsanitize=address`.

**Resolution:** Use the `linux-release-host-asan` preset as designed — let all
components get HOST_ASAN consistently. Do NOT try to selectively disable ASAN
on individual components. Only keep the minimal overrides that the upstream
`CMakePresets.json` already defines (amd-llvm, hipcc, hipify, etc. are
excluded from ASAN in the preset itself) plus `therock-SuiteSparse_SANITIZER=OFF`
and `amdsmi_SANITIZER=OFF` which were needed to avoid other build issues:

```dockerfile
cmake --preset linux-release-host-asan \
    -B build \
    -DTHEROCK_AMDGPU_FAMILIES=${THEROCK_AMDGPU_FAMILIES} \
    -DTHEROCK_ENABLE_DC_TOOLS=OFF \
    -DBUILD_TESTING=OFF \
    -Dtherock-SuiteSparse_SANITIZER=OFF \
    -Damdsmi_SANITIZER=OFF \
    ${MINIMAL_FLAGS}
```

**Key learning:** You cannot apply ASAN to only the HIP runtime when using
TheRock's build system. The global ASAN flags propagate to all code including
third-party deps, and any mismatch between ASAN-instrumented static libraries
and non-ASAN executables causes linker errors.

---

## Issue 5: PyTorch build fails — `nccl_device.h` not found

**Symptom:**
```
/build/pytorch/torch/csrc/distributed/c10d/symm_mem/nccl_dev_cap.hpp:14:10:
    fatal error: 'nccl_device.h' file not found
   14 | #include <nccl_device.h>
```

Build was at `[12174/12781]` — very near the end of the PyTorch compilation.

**Root cause:** PyTorch v2.11 requires `nccl_device.h` for the NCCL Symmetric
Memory feature (`NCCLSymmetricMemory.cu`). When ROCm is installed via debian
packages (as in the working Dockerfile), RCCL ships with proper CMake config
files that PyTorch auto-detects via `find_package()`. With TheRock's
`cmake --install`, the install layout and CMake config files differ — PyTorch's
NCCL detection falls back to building its own bundled NCCL submodule, which
doesn't provide `nccl_device.h`.

**Resolution (partial):** Initially added `USE_SYSTEM_NCCL=1` with RCCL paths:

```dockerfile
ENV USE_SYSTEM_NCCL=1
ENV NCCL_ROOT=/opt/rocm
ENV NCCL_INCLUDE_DIR=/opt/rocm/include
ENV NCCL_LIB_DIR=/opt/rocm/lib
```

These are placed after `git submodule update --init --recursive` to avoid
invalidating Docker layer cache for the clone/submodule steps.

However, this alone did not fix the issue — see Issue 8.

---

## Issue 6: PyTorch cloned into TheRock directory

**Symptom:** PyTorch source ended up at `/build/TheRock/pytorch/` instead of
`/build/pytorch/`, causing confusing paths in build logs.

**Root cause:** No `WORKDIR` directive between the TheRock build (which sets
`WORKDIR /build/TheRock`) and the PyTorch `git clone`.

**Resolution:** Added `WORKDIR /build` before the PyTorch clone:

```dockerfile
WORKDIR /build

RUN git clone --branch ${PYTORCH_GIT_REF} --single-branch \
    https://github.com/pytorch/pytorch.git
```

---

## Issue 7: Docker build log truncated at 2 MiB

**Symptom:**
```
[output clipped, log limit 2MiB reached]
```

The actual compilation error was hidden in the middle of the output, making
debugging impossible. The CK SDPA kernel compilation generates thousands of
repetitive warning lines per `.hip.o` file, quickly exhausting the 2 MiB
BuildKit log buffer.

**Root cause:** Docker BuildKit has a default per-step log buffer limit of
2 MiB. The verbose CK SDPA kernel warnings fill this buffer before the build
completes.

**Workaround:** Use `--progress=plain` to stream full output to the terminal
(bypasses the internal buffer), and pipe to `tee` to capture everything:

```bash
docker compose -f docker/docker-compose.build.yaml build --progress=plain 2>&1 \
    | tee stdout.asan_docker_build.log
```

The `tee` file captures the full terminal stream regardless of BuildKit's
internal buffer limit. The `[output clipped]` message only affects what
BuildKit stores internally — `--progress=plain` streams everything to stdout
as it happens.

To also increase the internal buffer (useful for `docker buildx`):

```bash
docker buildx create \
    --name biglog \
    --driver docker-container \
    --driver-opt env.BUILDKIT_STEP_LOG_MAX_SIZE=104857600 \
    --use
```

---

## Issue 8: `nccl_device.h` still not found despite `USE_SYSTEM_NCCL=1`

**Symptom:** Same error as Issue 5 — still `fatal error: 'nccl_device.h' file not found`
even after setting `USE_SYSTEM_NCCL=1` and `NCCL_INCLUDE_DIR=/opt/rocm/include`.
Build failed at `[12175/12781]` after ~75 minutes of compilation.

**Root cause:** `USE_SYSTEM_NCCL=1` told PyTorch to search `/opt/rocm/include`
for NCCL headers (visible in the `-I/opt/rocm/include` compiler flags), but
**`nccl_device.h` simply doesn't exist there**. TheRock's RCCL
`cmake --install` installs `nccl.h` and `rccl.h` but omits `nccl_device.h` —
a device-level header that lives in RCCL's source tree
(`src/include/nccl_device.h`) and is not part of the standard install targets.

The ROCm debian packages (used in `Dockerfile.rocm70_2-ubuntu-pytorch`)
include `nccl_device.h` as part of the `rccl-dev` package, which is why the
working Dockerfile doesn't need any special handling.

**Resolution:** After TheRock's `cmake --install`, copy `nccl_device.h` from
the RCCL source tree in TheRock's build directory to the install prefix:

```dockerfile
RUN find /build/TheRock -name "nccl_device.h" -type f | head -1 | \
    xargs -I{} cp {} ${ROCM_INSTALL_PREFIX}/include/ && \
    ls -la ${ROCM_INSTALL_PREFIX}/include/nccl_device.h
```

This step is placed immediately after `cmake --install` so it fails fast
(in seconds) if the header can't be found, rather than after a 75-minute
PyTorch build.

**Why not just `USE_SYSTEM_NCCL`?** The env var only controls which NCCL
library PyTorch links against (system vs bundled submodule). It doesn't
help if the header file is physically missing from the include path.
Both the system RCCL and the bundled NCCL need the header to be present;
the difference is that TheRock's install step doesn't place it there.

---

## Issue 9: `nccl_device/impl/comm__funcs.h` not found (incomplete RCCL header copy)

**Symptom:**
```
/opt/rocm/include/nccl_device.h:7:10: fatal error: 'nccl_device/impl/comm__funcs.h' file not found
    7 | #include "nccl_device/impl/comm__funcs.h"
```

Build was at `[12173/12781]` — same file (`NCCLSymmetricMemory.cu`) as Issue 8.

**Root cause:** The Issue 8 fix only copied the single `nccl_device.h` file
from RCCL's source tree. However, `nccl_device.h` uses `#include` directives
that reference a `nccl_device/` subdirectory containing implementation headers
(`nccl_device/impl/comm__funcs.h`, etc.). Without the entire subdirectory tree,
the compiler finds `nccl_device.h` but cannot resolve its transitive includes.

**Resolution:** Instead of copying just the single file, find the RCCL include
directory and copy both `nccl_device.h` and the entire `nccl_device/`
subdirectory:

```dockerfile
RUN RCCL_SRC=$(find /build/TheRock -path "*/rccl/src/include/nccl_device.h" -type f | head -1 | xargs dirname) && \
    cp ${RCCL_SRC}/nccl_device.h ${ROCM_INSTALL_PREFIX}/include/ && \
    cp -r ${RCCL_SRC}/nccl_device ${ROCM_INSTALL_PREFIX}/include/ && \
    ls -la ${ROCM_INSTALL_PREFIX}/include/nccl_device.h && \
    ls ${ROCM_INSTALL_PREFIX}/include/nccl_device/
```

**Key learning:** When copying headers from a source tree, always check whether
the header has transitive includes pointing to sibling directories. A single
`cp` of one file is rarely sufficient for C/C++ header hierarchies.

---

## Issue 10: `../comm_tmp.h` not found (incomplete RCCL header directory copy)

**Symptom:**
```
/opt/rocm/include/nccl_device/impl/comm__types.h:9:10: fatal error: '../comm_tmp.h' file not found
    9 | #include "../comm_tmp.h"
```

Two files failed: `nccl_extension.cu` and `NCCLSymmetricMemory.cu` at
`[11894/12781]` and `[11895/12781]`.

**Root cause:** The Issue 9 fix copied `nccl_device.h` and the `nccl_device/`
subdirectory, but RCCL's device headers use relative `#include` paths that
reference sibling files at various directory levels. Specifically,
`nccl_device/impl/comm__types.h` includes `../comm_tmp.h`, which resolves to
`nccl_device/comm_tmp.h` — a file that exists in RCCL's `src/include/`
directory but was either not part of the `nccl_device/` subdirectory or was
generated during the build.

Cherry-picking individual files or subdirectories from RCCL's source include
tree doesn't work because the headers have deep cross-references with relative
paths.

**Resolution:** Copy the **entire contents** of RCCL's `src/include/` directory
to the install prefix, preserving the full relative include structure:

```dockerfile
RUN RCCL_INC=$(find /build/TheRock -path "*/rccl/src/include/nccl_device.h" -type f | head -1 | xargs dirname) && \
    cp -r ${RCCL_INC}/* ${ROCM_INSTALL_PREFIX}/include/ && \
    ls -la ${ROCM_INSTALL_PREFIX}/include/nccl_device.h && \
    ls ${ROCM_INSTALL_PREFIX}/include/nccl_device/
```

**Key learning:** When a source tree has headers with relative `#include`
paths (`../`, `../../`, etc.), you must copy the entire include subtree — not
individual files or directories. The relative paths encode assumptions about
the directory layout.

---

## Issue 11: `comm_tmp.h` still not found — generated file not in source tree

**Symptom:** Same as Issue 10:
```
/opt/rocm/include/nccl_device/impl/comm__types.h:9:10: fatal error: '../comm_tmp.h' file not found
```

**Root cause:** The Issue 10 fix (`cp -r ${RCCL_INC}/*`) copies the entire
RCCL `src/include/` directory, but `comm_tmp.h` is a **generated** header —
it's created during RCCL's CMake build process (likely by a code generator
like `generate.py`) and placed in the **build** directory, not the source
tree. Since it never existed in `rccl/src/include/nccl_device/`, the
wildcard copy didn't include it.

**Resolution:** After copying the source headers, search the TheRock build
tree for build-generated RCCL device headers (like `comm_tmp.h`) and merge
them into the install prefix:

```dockerfile
RUN RCCL_INC=$(find /build/TheRock -path "*/rccl/src/include/nccl_device.h" \
        -type f | head -1 | xargs dirname) && \
    cp -r ${RCCL_INC}/* ${ROCM_INSTALL_PREFIX}/include/ && \
    for f in $(find /build/TheRock -path "*/rccl*" -name "comm_tmp.h" -type f); do \
        DEST_DIR=${ROCM_INSTALL_PREFIX}/include/$(dirname "${f}" | grep -oP 'nccl_device.*'); \
        mkdir -p "${DEST_DIR}" && cp -n "${f}" "${DEST_DIR}/"; \
    done && \
    test -f ${ROCM_INSTALL_PREFIX}/include/nccl_device/comm_tmp.h || \
        (echo "ERROR: comm_tmp.h not found" && exit 1)
```

The step now includes a validation check: if `comm_tmp.h` can't be found
anywhere in the TheRock build tree, the build fails immediately with a
diagnostic listing all `comm_tmp.h` locations, instead of failing 75 minutes
later during PyTorch compilation.

**Key learning:** Header directories for projects like RCCL contain a mix of
source-tree headers and build-generated headers. Copying only the source
include directory misses generated files. You must merge both source and
build directories.

---

## Issue 12: `core_tmp.h` not found — more generated headers missing

**Symptom:**
```
/opt/rocm/include/nccl_device/impl/../comm_tmp.h:9:10: fatal error: 'core_tmp.h' file not found
```

**Root cause:** Issues 8–11 attempted to copy RCCL device headers piecemeal:
first `nccl_device.h`, then `nccl_device/`, then `comm_tmp.h`. Each fix
uncovered another missing generated header. The RCCL build's hipify step
(`hipify-perl`) and code generator (`generate.py`) produce multiple `*_tmp.h`
files (`comm_tmp.h`, `core_tmp.h`, `ptr_tmp.h`, etc.) in the **hipified build
tree** at `${CMAKE_BINARY_DIR}/hipify/src/include/nccl_device/`. Copying from
the source tree or searching for individual files by name is a losing game.

**Resolution:** Copy from the **hipified build directory** instead of the
source tree. The path `*/hipify/src/include/` in TheRock's build tree contains
the complete set of RCCL device headers — both hipified source headers and
all generated `*_tmp.h` files:

```dockerfile
RUN HIPIFY_INC=$(find /build/TheRock -path "*/hipify/src/include/nccl_device.h" \
        -type f | head -1 | xargs dirname) && \
    cp -r ${HIPIFY_INC}/nccl_device.h ${ROCM_INSTALL_PREFIX}/include/ && \
    cp -r ${HIPIFY_INC}/nccl_device ${ROCM_INSTALL_PREFIX}/include/ && \
    find ${ROCM_INSTALL_PREFIX}/include/nccl_device -type f | sort
```

**Key learning:** RCCL's build produces a parallel "hipified" include tree
under `<build>/hipify/src/include/` that mirrors `src/include/` but with
generated files added. Always copy from this directory — not from the source
tree — to get the complete header set. This is exactly what RCCL's
`CMakeLists.txt` should do in its `install()` section (see analysis above).

---

## Issue 13: `nccl.h` not found — RCCL build-internal header not installed

**Symptom:**
```
/opt/rocm/include/nccl_device/impl/../core_tmp.h:9:10: fatal error: 'nccl.h' file not found
```

**Root cause:** RCCL's `CMakeLists.txt` generates two copies of the main
header from `src/nccl.h.in`:

```cmake
configure_file(src/nccl.h.in ${PROJECT_BINARY_DIR}/include/rccl/rccl.h)  # installed
configure_file(src/nccl.h.in ${PROJECT_BINARY_DIR}/include/nccl.h)       # internal only
```

Only `rccl.h` is installed (to `<prefix>/include/rccl/rccl.h`). The internal
`nccl.h` is used during RCCL's build via `-I${PROJECT_BINARY_DIR}/include`
but never installed. The device headers (e.g. `core_tmp.h`) `#include "nccl.h"`
by that name, which fails when they're placed in `/opt/rocm/include/` without
the internal `nccl.h`.

The ROCm debian packages ship both `rccl.h` and `nccl.h` in their `-dev`
package, so this problem doesn't appear with package-based installs.

**Resolution:** Find the generated `nccl.h` in the RCCL build directory and
copy it to the install prefix:

```dockerfile
RCCL_NCCL_H=$(find /build/TheRock -path "*/rccl*/include/nccl.h" \
    ! -path "*/src/*" -type f | head -1) && \
cp ${RCCL_NCCL_H} ${ROCM_INSTALL_PREFIX}/include/nccl.h
```

**Key learning:** RCCL has a split-header design: `rccl.h` (public, installed)
and `nccl.h` (internal, identical content, not installed). When exposing
internal device headers for consumers like PyTorch, the internal `nccl.h`
must also be installed.

---

## Issue 14: `core.h` not found — nccl_device headers reference parent includes

**Symptom:**
```
/opt/rocm/include/nccl_device/impl/../ptr.h:9:10: fatal error: 'core.h' file not found
```

**Root cause:** The file listing from the copy step revealed that `core.h`
and `comm.h` are **not** in the `hipify/src/include/nccl_device/` directory.
The generate.py script produces `core_tmp.h` and `comm_tmp.h` as replacements,
but the original `core.h` lives at `hipify/src/include/core.h` (one level up).

During RCCL's build, CMake adds both directories to the include path:
```cmake
target_include_directories(rccl PRIVATE ${HIPIFY_DIR}/src/include)
target_include_directories(rccl PRIVATE ${HIPIFY_DIR}/src/include/nccl_device)
```

So `#include "core.h"` in `nccl_device/ptr.h` resolves via the `-I` path to
`src/include/core.h`. When we only copied `nccl_device/` to `/opt/rocm/include/`,
the parent-level headers (`core.h`, `comm.h`, `device.h`, etc.) were missing.

**Resolution:** Copy the **entire** `hipify/src/include/` tree to the install
prefix (using `-n` to avoid overwriting existing files from TheRock's install):

```dockerfile
RUN HIPIFY_INC=$(find /build/TheRock \
        -path "*/hipify/src/include/nccl_device.h" \
        -type f | head -1 | xargs dirname) && \
    cp -rn ${HIPIFY_INC}/* ${ROCM_INSTALL_PREFIX}/include/
```

This adds RCCL's internal headers (core.h, comm.h, device.h, etc.) alongside
the device headers. The `-n` flag ensures already-installed headers from
TheRock's `cmake --install` are not overwritten.

---

## Issue 15: Pre-flight header check to catch all missing includes at once

**Symptom:** Each missing RCCL header was only discovered after a 75-minute
PyTorch build reached the `NCCLSymmetricMemory.cu` file at step `[12173/12781]`.
Issues 8–14 each required a full rebuild cycle.

**Root cause:** No validation step between installing RCCL headers and starting
the PyTorch build.

**Resolution:** Added two fixes:

1. **Symlinks for ambiguous includes**: RCCL's `nccl_device/ptr.h` includes
`"core.h"` which exists at the parent level (`src/include/core.h`) but not
in `nccl_device/`. During RCCL's build this works via `-I` paths, but PyTorch
doesn't have that path. Fixed by symlinking:

```dockerfile
for h in core.h comm.h device.h; do
    if [ -f ${ROCM_INSTALL_PREFIX}/include/${h} ] && \
       [ ! -f ${ROCM_INSTALL_PREFIX}/include/nccl_device/${h} ]; then
        ln -s ../${h} ${ROCM_INSTALL_PREFIX}/include/nccl_device/${h}
    fi
done
```

2. **Pre-flight compilation check**: A 2-second step that test-compiles
`#include "nccl_device.h"` with the same include paths PyTorch will use.
Any missing transitive include fails immediately:

```dockerfile
RUN echo '#include "nccl_device.h"' > /tmp/test_nccl_device.cpp && \
    ${ROCM_INSTALL_PREFIX}/llvm/bin/clang++ -fsyntax-only -x c++ \
        -I${ROCM_INSTALL_PREFIX}/include \
        -I${ROCM_INSTALL_PREFIX}/include/nccl_device \
        -ferror-limit=0 \
        /tmp/test_nccl_device.cpp
```

With `-ferror-limit=0`, the compiler reports ALL missing headers at once
instead of stopping at the first one.

---

## Issue 16: Symlinks pointed to wrong `core.h` — RCCL internal vs device header

**Symptom:** Pre-flight check fails with errors from deep RCCL internals:
```
/opt/rocm/include/core.h:39: In file included: alloc.h
/opt/rocm/include/core.h:45: In file included: nvtx.h → roctx.h → device.h
/opt/rocm/include/device.h:29:10: fatal error: 'nccl_tuner.h' file not found
```

**Root cause:** Issue 15's fix created symlinks:
```
nccl_device/core.h -> ../core.h    (WRONG!)
nccl_device/comm.h -> ../comm.h    (WRONG!)
```

RCCL has **two completely different files** named `core.h`:
- `src/include/nccl_device/core.h` — lightweight device header (~160 lines,
  includes only `coop.h` and `utility.h`)
- `src/include/core.h` — heavy RCCL internal header (~500+ lines, pulls in
  `alloc.h`, `nvtx.h`, `device.h`, `nccl_tuner.h`, and the entire RCCL
  internals)

The symlink pointed `nccl_device/core.h` at the wrong one, causing the
include chain to explode into RCCL's full internal header tree, eventually
failing on `nccl_tuner.h` (which lives in `src/include/plugin/`).

Similarly, Issue 14's approach of copying the entire `hipify/src/include/*`
was wrong — it dumped RCCL's internal headers (`core.h`, `alloc.h`, `nvtx.h`,
`device.h`, etc.) into `/opt/rocm/include/`, polluting the install prefix.

**Resolution:** 
1. Removed the symlinks and the bulk `cp -rn ${HIPIFY_INC}/*` approach
2. Copy only `nccl_device.h` and `nccl_device/` from the hipified build tree
   (which has the generated `*_tmp.h` files)
3. Copy `nccl_device/core.h` and `nccl_device/comm.h` from the **RCCL source
   tree** (not the hipified tree, which doesn't produce them; not the parent
   directory, which has different files with the same name)

```dockerfile
RCCL_SRC_INC=$(find /build/TheRock \
    -path "*/rccl/src/include/nccl_device/core.h" \
    -type f | head -1 | xargs dirname) && \
for h in core.h comm.h; do
    cp ${RCCL_SRC_INC}/${h} ${ROCM_INSTALL_PREFIX}/include/nccl_device/${h}
done
```

**Key learning:** When multiple directories contain files with the same name,
never use symlinks or bulk copies without verifying file identity. RCCL's
`src/include/core.h` and `src/include/nccl_device/core.h` have completely
different contents despite the same filename.

---

## Summary of all fixes applied

| # | Issue | Fix | Dockerfile location |
|---|-------|-----|---------------------|
| 1 | Missing ROCm path vars | Added `ROCM_PATH`, `HIP_PATH`, `CMAKE_PREFIX_PATH` | Stage 3: ROCm environment |
| 2 | pip uninstall failure on Ubuntu 24.04 | Use Python venv instead of system pip | Stage 4: before PyTorch build |
| 3 | AOTriton missing liblzma | Added `liblzma-dev` to apt packages | Stage 1: System packages |
| 4 | ASAN symbol mismatch in TheRock | Use preset as designed; don't selectively disable ASAN | Stage 2: TheRock configure |
| 5 | `nccl_device.h` not found (initial) | Set `USE_SYSTEM_NCCL=1` + RCCL paths (partial fix) | Stage 4: before `setup.py develop` |
| 6 | PyTorch cloned in wrong directory | Added `WORKDIR /build` before clone | Stage 4: before git clone |
| 7 | Docker log truncated at 2 MiB | Use `--progress=plain` with `tee` | Build command (not Dockerfile) |
| 8 | `nccl_device.h` still missing | Copy from RCCL source tree after TheRock install (partial) | Stage 2: after `cmake --install` |
| 9 | `nccl_device/` subdirectory missing | Copy entire `nccl_device/` dir alongside `nccl_device.h` (partial) | Stage 2: after `cmake --install` |
| 10 | `../comm_tmp.h` relative include missing | Copy entire RCCL `src/include/*` contents (partial) | Stage 4: after submodule init |
| 11 | `comm_tmp.h` is build-generated, not in source | Find & merge generated headers from RCCL build dir (partial) | Stage 4: after submodule init |
| 12 | `core_tmp.h` + more generated headers missing | Copy from `hipify/src/include/` build tree | Stage 4: after submodule init |
| 13 | `nccl.h` not installed (internal-only header) | Copy generated `nccl.h` from RCCL build dir | Stage 4: after submodule init |
| 14 | `core.h` missing (parent-level internal header) | Copy entire `hipify/src/include/*` with `cp -rn` | Stage 4: after submodule init |
| 15 | Repeated build cycles for each missing header | Pre-flight `clang++ -fsyntax-only` check | Stage 4: after header copy |
| 16 | Symlinks to wrong `core.h` (internal vs device) | Copy from RCCL source tree, not symlink to parent | Stage 4: after header copy |
| 17 | `railGinBarrierCount` missing in RCCL struct | Patch `nccl_dev_cap.hpp` to raise SYMMEM_DEVICE threshold | Stage 4: after PyTorch clone |
| 18 | `libdrm/drm.h` not found (`rocm_smi/kfd_ioctl.h`) | Added `libdrm-dev` to apt packages | Stage 1: System packages |
| 19 | `import torch` segfaults (ASAN not preloaded) | Do NOT use `LD_PRELOAD`; use NEEDED + `verify_asan_link_order=0` | Dockerfile ENV |
| 20 | `fbgemm_gpu_py.so: undefined symbol: rsmi_shut_down` | `patchelf --add-needed librocm_smi64.so` on the `.so` | `setup_repro_env.sh` post-build |
| 21 | `libclang-cpp.so.23.0git` not found at runtime | Add `llvm/lib` + `lib/rocm_sysdeps/lib` to `LD_LIBRARY_PATH` | Dockerfile ENV + setup script |
| 22 | roctracer `Finalize()` assert crash (exit 134) | PR [#4552](https://github.com/ROCm/rocm-systems/pull/4552); workaround: check stdout | `setup_repro_env.sh` |
| 23 | ASAN SEGV in `amd::Device::init()` with `LD_PRELOAD` | Load ASAN via NEEDED + `verify_asan_link_order=0` (no `LD_PRELOAD`) | Dockerfile ENV |
| 24 | `tensordict` missing deps (`pyvers`, `cloudpickle`) | Install explicitly alongside `--no-deps` | `setup_repro_env.sh` |

---

## Issue 17: RCCL/NCCL API version mismatch — `railGinBarrierCount`

**Error:**
```
/build/pytorch/torch/csrc/distributed/c10d/symm_mem/nccl_devcomm_manager.hpp:80:10:
  error: no member named 'railGinBarrierCount' in 'ncclDevCommRequirements'
    reqs.railGinBarrierCount = gin_barrier_count;
    ~~~  ^
```

**Root cause:** PyTorch v2.11.0-rc2 file
`torch/csrc/distributed/c10d/symm_mem/nccl_dev_cap.hpp` defines:

```cpp
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
#define NCCL_HAS_SYMMEM_DEVICE_SUPPORT
#include <nccl_device.h>
#endif
```

TheRock's RCCL (develop branch) reports **NCCL version 2.28.3**. This triggers
`NCCL_HAS_SYMMEM_DEVICE_SUPPORT`, which compiles `nccl_devcomm_manager.hpp`.
That file uses `reqs.railGinBarrierCount`, but that struct field was only added
in **NCCL 2.28.7**. RCCL 2.28.3 has the `ncclDevCommRequirements` struct but
without `railGinBarrierCount`.

The **working** Dockerfile (`Dockerfile.rocm70_2-ubuntu-pytorch`) uses ROCm
7.0.2 packages, which ship RCCL ~2.22.x. At that version, both
`NCCL_HAS_SYMMEM_SUPPORT` (>= 2.27.0) and `NCCL_HAS_SYMMEM_DEVICE_SUPPORT`
(>= 2.28.0) are disabled, so the entire symmetric-memory device code is skipped.

**Fix:** Patch `nccl_dev_cap.hpp` to raise the `NCCL_HAS_SYMMEM_DEVICE_SUPPORT`
threshold from 2.28.0 to 2.29.0. This prevents the device-communicator code
from compiling while keeping basic symmetric memory support (>= 2.27.0):

```dockerfile
RUN sed -i 's/NCCL_VERSION(2, 28, 0)/NCCL_VERSION(2, 29, 0)/' \
    pytorch/torch/csrc/distributed/c10d/symm_mem/nccl_dev_cap.hpp
```

Additionally, the complex RCCL device-header copy (nccl_device.h, nccl_device/,
generated *_tmp.h files, nccl.h from build tree) and pre-flight clang++ check
are no longer needed because `nccl_device.h` is only `#include`d inside the
`NCCL_HAS_SYMMEM_DEVICE_SUPPORT` guard. Replaced the entire header-copy block
with a simple symlink: `ln -s rccl.h nccl.h`.

**Key learning:** When building PyTorch against an RCCL version that tracks
NCCL but lags behind on specific features, the version-gated macros in
PyTorch may enable code that references API additions not yet in RCCL.
Always check the RCCL version's actual API surface against what PyTorch
expects, especially for features marked "available since NCCL X.Y.Z" in
the NCCL documentation.

---

## Issue 18: Missing `libdrm/drm.h` — `rocm_smi` dependency

**Error:**
```
/opt/rocm/include/rocm_smi/kfd_ioctl.h:26:10: fatal error: libdrm/drm.h: No such file or directory
```

**Root cause:** PyTorch's `intra_node_comm.cpp` (symmetric memory subsystem)
includes ROCm SMI headers which in turn include `kfd_ioctl.h`. That header
requires `libdrm/drm.h` from the `libdrm-dev` package, which was not installed.

**Fix:** Add `libdrm-dev` to the system packages in Stage 1:
```dockerfile
RUN apt-get update && apt-get install -y \
    ...
    libdrm-dev \
    ...
```

---

## Issue 19 (revised): ASAN SEGV in `amd::Device::init()` when using `LD_PRELOAD`

**Symptom:**
```
AddressSanitizer:DEADLYSIGNAL
==56792==ERROR: AddressSanitizer: SEGV on unknown address 0x000000000000
(pc 0x000000000000 bp 0x7fffffffcfa0 sp 0x7fffffffce28 T0)
==56792==Hint: pc points to the zero page.
```

This crash occurred on any `import torch` when `LD_PRELOAD` was set to
`libclang_rt.asan-x86_64.so` alongside the ASAN-built ROCm libraries.

**Root cause:** `LD_PRELOAD` of the ASAN runtime globally intercepts `dlopen`
and `dlsym`. HIP's internal initialization (`amd::Device::init()`) uses `dlopen`
to dynamically load `libhsa-runtime64.so` and resolves function pointers via
`dlsym`. With ASAN intercepting these calls, the resolved function pointers
were null, leading to a null-pointer dereference (SEGV at address 0x0).

**Resolution:** Do NOT use `LD_PRELOAD` for the ASAN runtime. Instead, the
ASAN-built ROCm libraries (in `/opt/rocm-asan/lib/`) already have
`libclang_rt.asan-x86_64.so` as a `NEEDED` dependency (linked at build time).
The dynamic linker loads it automatically when the ASAN runtime directory is
on `LD_LIBRARY_PATH`. Set `verify_asan_link_order=0` in `ASAN_OPTIONS` to
suppress the ASAN check that it must be the first loaded library:

```bash
unset LD_PRELOAD
ASAN_RT_DIR=$(dirname $(find /opt/rocm/llvm/lib/clang -name "libclang_rt.asan-x86_64.so" | head -1))
export LD_LIBRARY_PATH="/opt/rocm-asan/lib:${ASAN_RT_DIR}:/opt/rocm/lib:/opt/rocm/llvm/lib:/opt/rocm/lib/rocm_sysdeps/lib"
export ASAN_OPTIONS="detect_leaks=0:halt_on_error=0:symbolize=1:verify_asan_link_order=0"
```

**Key learning:** `LD_PRELOAD` is incompatible with libraries that internally
use `dlopen`/`dlsym` for lazy loading. The ASAN runtime's global interception
of these calls breaks the lazy-loading pattern. When the ASAN runtime is loaded
as a normal shared library dependency (NEEDED), it intercepts only the memory
operations of the libraries it's linked to, without breaking `dlopen`/`dlsym`.

---

## Issue 20: `fbgemm_gpu_py.so: undefined symbol: rsmi_shut_down`

**Symptom:**
```
OSError: /opt/pytorch-venv/lib/python3.12/site-packages/fbgemm_gpu/fbgemm_gpu_py.so:
    undefined symbol: rsmi_shut_down
```

**Root cause:** `fbgemm_gpu_py.so` references `rsmi_shut_down` (from ROCm SMI)
but does not list `librocm_smi64.so` in its ELF `NEEDED` entries. With
TheRock's install layout, the symbol isn't pulled in transitively — unlike the
debian-packaged ROCm where transitive dependencies happen to resolve it.

Verified with:
```bash
readelf -d fbgemm_gpu_py.so | grep NEEDED | grep -i smi   # empty
nm -D /opt/rocm/lib/librocm_smi64.so | grep rsmi_shut_down  # present
```

**Resolution:** After FBGEMM is built and installed, use `patchelf` to
explicitly add the missing dependency:

```bash
patchelf --add-needed librocm_smi64.so \
    /opt/pytorch-venv/lib/python3.12/site-packages/fbgemm_gpu/fbgemm_gpu_py.so
```

The script checks idempotently via `readelf -d` before patching.

---

## Issue 21: `libclang-cpp.so.23.0git` and `librocm_sysdeps_z.so.1` not found

**Symptom:**
```
ImportError: libclang-cpp.so.23.0git: cannot open shared object file
ImportError: librocm_sysdeps_z.so.1: cannot open shared object file
```

These failures appeared at runtime during `import torch` or when running
ASAN test binaries.

**Root cause:** TheRock installs LLVM libraries to `/opt/rocm/llvm/lib/` and
system dependency shims to `/opt/rocm/lib/rocm_sysdeps/lib/`. Neither of these
non-standard paths is on the default `LD_LIBRARY_PATH`.

**Resolution:** Include both paths in `LD_LIBRARY_PATH`:

```dockerfile
ENV LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/llvm/lib:/opt/rocm/lib/rocm_sysdeps/lib
```

Note: The correct path for sysdeps is `/opt/rocm/lib/rocm_sysdeps/lib` (not
`/opt/rocm/rocm_sysdeps/lib`).

---

## Issue 22: roctracer `Finalize()` assertion crash (exit code 134)

**Symptom:**
```
python3: .../roctracer/hsa_support.cpp:629: void roctracer::hsa_support::Finalize():
    Assertion `!"hsa_amd_profiling_async_copy_enable failed"' failed.
Aborted
```

Every Python process that imports `torch` and touches HIP exits with
`SIGABRT` (exit code 134) during shutdown. The import and execution succeed,
but the assertion fires during process teardown.

**Root cause:** In `hsa_support.cpp`, `Finalize()` calls
`hsa_amd_profiling_async_copy_enable_fn()` to restore the profiling state.
During process exit, the C++ static destruction order across shared libraries
is undefined. If the HSA runtime is already shut down when roctracer's
`Finalize()` runs, the function call returns an error status, and the
`assert()` crashes the process.

**Upstream fix:** Draft PR [#4552](https://github.com/ROCm/rocm-systems/pull/4552)
replaces the `assert()` with a graceful warning and null-pointer check.

**Workaround (until PR merges):** In scripts that verify Python imports,
check stdout for success strings rather than relying on exit codes:

```bash
OUTPUT=$($PYTHON -c "import fbgemm_gpu; print('[OK] fbgemm_gpu imported')" 2>&1)
if echo "$OUTPUT" | grep -q '\[OK\] fbgemm_gpu imported'; then
    echo "SUCCESS"
else
    echo "FAILED"
fi
```

---

## Issue 23: `LD_PRELOAD` ASAN + HIP `dlopen` conflict (detailed)

This is a more detailed analysis supplementing Issue 19.

**Investigation path:**

1. With `LD_PRELOAD=libclang_rt.asan-x86_64.so` and ASAN-built libraries on
   `LD_LIBRARY_PATH`: SEGV at null function pointer in `amd::Device::init()`
2. Without `LD_PRELOAD` but with ASAN-built libraries on path: `import torch`
   fails with `ImportError: libclang_rt.asan-x86_64.so` not found (because the
   ASAN-built libraries have it as a NEEDED dependency)
3. Without ASAN-built libraries on path (no `/opt/rocm-asan/lib`): `import torch`
   succeeds with normal (non-ASAN) libraries
4. With ASAN-built libraries + ASAN runtime dir on `LD_LIBRARY_PATH` + no
   `LD_PRELOAD` + `verify_asan_link_order=0`: works correctly, reports 8 devices

**Verification that ASAN is active:**

```bash
# Check process maps show ASAN-built library from /opt/rocm-asan/lib
python3 -c "
import torch; torch.cuda.device_count()
import subprocess, os
maps = open(f'/proc/{os.getpid()}/maps').read()
for line in maps.splitlines():
    if 'libamdhip64' in line or 'libclang_rt.asan' in line:
        print(line)
"
# Should show: /opt/rocm-asan/lib/libamdhip64.so (not /opt/rocm/lib/)
# Should show: libclang_rt.asan-x86_64.so loaded

# Confirm ASAN symbols present
nm -D /opt/rocm-asan/lib/libamdhip64.so | grep __asan | head -5
# Shows: U __asan_after_dynamic_init, __asan_alloca_poison, etc.
```

---

## Issue 24: Missing `pyvers` and `cloudpickle` for `tensordict`

**Symptom:**
```
ModuleNotFoundError: No module named 'pyvers'
ModuleNotFoundError: No module named 'cloudpickle'
```

**Root cause:** `tensordict` is installed with `--no-deps` (to avoid pulling
in an incompatible PyTorch wheel). This skips installing its transitive
dependencies `pyvers` and `cloudpickle`.

**Resolution:** Explicitly install both alongside `tensordict`:

```bash
pip install pyvers cloudpickle tensordict --no-deps
```

---
---

# Docker Setup Guide

This section describes the prerequisites, build instructions, and runtime
configuration for the TheRock ASAN + PyTorch Docker image.

## Prerequisites

### Hardware
- AMD Instinct GPU (MI300X / MI250X / MI210 or similar) with ROCm support
- Sufficient disk space: ~100 GB for the Docker image build (TheRock build
  artifacts are large)
- Recommended: >=64 GB RAM for TheRock compilation

### Software
- Docker Engine 20.10+ with BuildKit enabled
- `docker compose` v2 (or `docker-compose` v1.29+)
- NVIDIA Container Toolkit is **not** needed; this uses AMD ROCm
- Host must have the `amdgpu` kernel driver loaded (`modprobe amdgpu`)
- ROCm kernel-mode driver installed on the host (verify: `ls /dev/kfd /dev/dri/render*`)

### Network
- Internet access during build (clones TheRock, PyTorch, pip packages)
- GitHub access for `git clone` operations

## Building the Docker Image

### 1. Configure the environment file

Copy and edit the environment file for your GPU target:

```bash
cp docker/.env.example docker/.env
```

Set the following in `.env`:

```bash
DOCKERFILE=Dockerfile.therock-host-asan-pytorch
IMAGE_NAME=aorta:therock-host-asan-pytorch
CONTAINER_NAME=<your-username>-therock-asan
```

The Dockerfile accepts build `ARG`s with these defaults:

| ARG | Default | Description |
|-----|---------|-------------|
| `THEROCK_TARBALL_VERSION` | `7.12.0` | TheRock tarball version to download for the normal ROCm install |
| `THEROCK_RELEASE_TYPE` | `stable` | Tarball source: `stable`, `nightlies`, `prereleases`, or `devreleases` |
| `THEROCK_TARBALL_AMDGPU_FAMILY` | `gfx950-dcgpu` | GPU family for tarball download (note: includes `-dcgpu` suffix) |
| `THEROCK_GIT_REF` | `therock-7.12` | TheRock source tag for ASAN build (must match tarball version for ABI compat) |
| `THEROCK_AMDGPU_FAMILIES` | `gfx950` | GPU architecture family for TheRock ASAN cmake build |
| `PYTORCH_GIT_REF` | `v2.11.0-rc2` | PyTorch version/branch/tag to build |
| `PYTORCH_ROCM_ARCH` | `gfx950` | PyTorch ROCm GPU architecture |
| `ROCM_INSTALL_PREFIX` | `/opt/rocm` | ROCm install prefix inside the container |

To override build ARGs, pass them via `--build-arg`:

```bash
docker compose ... build \
    --build-arg THEROCK_TARBALL_AMDGPU_FAMILY=gfx94X-dcgpu \
    --build-arg THEROCK_AMDGPU_FAMILIES=gfx942 ...
```

For nightly tarballs:

```bash
docker compose ... build \
    --build-arg THEROCK_RELEASE_TYPE=nightlies \
    --build-arg THEROCK_TARBALL_VERSION=7.9.0rc20251008 \
    --build-arg THEROCK_GIT_REF=main ...
```

Available tarball families can be found at the
[TheRock nightly S3 index](https://therock-nightly-tarball.s3.amazonaws.com/index.html).
Common families: `gfx950-dcgpu` (MI350X), `gfx94X-dcgpu` (MI300X).

### 2. Build with docker compose

```bash
docker compose -f docker/docker-compose.build.yaml \
    --env-file docker/.env \
    build --progress=plain 2>&1 \
    | tee stdout.asan_docker_build.log
```

**Build time:** ~30 minutes on a 64-core machine (the normal ROCm stack is
downloaded as a pre-built tarball; only the ASAN overlay is built from source).
Subsequent rebuilds are even faster thanks to BuildKit ccache mounts.

**Using a custom `.env` file:**

```bash
docker compose -f docker/docker-compose.build.yaml \
    --env-file docker/.env.my-custom \
    build --progress=plain
```

### 3. Build troubleshooting

- **Tarball download fails**: Verify the tarball URL exists. Check
  [TheRock nightly tarballs](https://therock-nightly-tarball.s3.amazonaws.com/index.html)
  for available versions and GPU families. For gfx950, try `nightlies` release
  type if `stable` is not available yet.
- **ABI mismatch (ASAN overlay crashes)**: `THEROCK_GIT_REF` must correspond to
  `THEROCK_TARBALL_VERSION`. E.g. tarball `7.12.0` requires tag `therock-7.12`.
- **Log truncation** (`[output clipped, log limit 2MiB reached]`): Always use
  `--progress=plain` and pipe to `tee` (see Issue 7)
- **CMake preset errors**: TheRock presets change across branches. Run
  `cmake --list-presets` inside the TheRock directory to see available presets
- **Network timeouts**: Retry the build — Docker layer caching will skip
  completed stages

## Running the Docker Container

### Basic run

```bash
docker compose -f docker/docker-compose.build.yaml \
    --env-file docker/.env \
    run --rm torchenv
```

Or with plain `docker run` (using the image name from `.env`):

```bash
docker run --rm -it \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video --group-add render \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    --shm-size=64g \
    aorta:therock-host-asan-pytorch
```

### GPU device access

The container requires access to:
- `/dev/kfd` — ROCm kernel fusion driver
- `/dev/dri/render*` — GPU render nodes

The `docker-compose.build.yaml` maps these automatically. For `docker run`,
use `--device=/dev/kfd --device=/dev/dri`.

### ASAN activation

ASAN is baked into the image via `ENV` directives — no entrypoint script needed.
The Dockerfile sets `LD_LIBRARY_PATH` to prepend `/opt/rocm-asan/lib` (ASAN overlay)
and `/opt/rocm-asan/rt` (ASAN runtime symlink) so they shadow the normal ROCm libs.
This works with `docker run`, `docker exec`, and images loaded from tar files.

To disable ASAN at runtime:

```bash
docker run ... -e LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/llvm/lib:/opt/rocm/lib/rocm_sysdeps/lib aorta:therock-host-asan-pytorch
```

### Verifying ASAN is active

Inside the container:

```bash
# Quick check
python3 -c "import torch; print('devices:', torch.cuda.device_count())"

# Verify ASAN library is loaded
python3 -c "
import torch, os
torch.cuda.device_count()
maps = open(f'/proc/{os.getpid()}/maps').read()
for line in maps.splitlines():
    if 'rocm-asan' in line or 'libclang_rt.asan' in line:
        print(line)
"
```

Expected: lines showing `/opt/rocm-asan/lib/libamdhip64.so` and
`libclang_rt.asan-x86_64.so` loaded.

### Entering a running container

To open a second shell into an already-running container (sharing the same
`/tmp`, filesystem, and GPU access):

```bash
docker exec -it <container_id_or_name> bash
```

Find the container ID with `docker ps`.

### Environment variables reference

| Variable | Default | Description |
|----------|---------|-------------|
| `ASAN_DISABLE` | unset | Set to `1` to skip ASAN overlay activation |
| `ASAN_OPTIONS` | `detect_leaks=0:halt_on_error=0:symbolize=1:verify_asan_link_order=0` | ASAN runtime options |
| `ASAN_SYMBOLIZER_PATH` | `/opt/rocm/llvm/bin/llvm-symbolizer` | Path to the LLVM symbolizer for readable ASAN traces |
| `HSA_TOOLS_LIB` | `""` (empty) | Set empty to prevent roctracer from loading (workaround for Issue 22) |
| `LD_LIBRARY_PATH` | `/opt/rocm-asan/lib:...:/opt/rocm/lib:...` | Library search path; ASAN overlay dir comes first |
| `ROCM_HOME` | `/opt/rocm` | ROCm installation prefix |

### Running the NaN reproduction test

After entering the container and running `setup_repro_env.sh` (if using the
`nan_repro_by_jeremy` workflow):

```bash
bash run_nan_test.sh <num_trials> <num_steps> "<gpu_list>"
# Example: 6 trials, 3000 steps, GPUs 0-3
bash run_nan_test.sh 6 3000 "0,1,2,3"
```

Results are written to `/tmp/nan_test_<pid>/`.

## Architecture Overview

The Docker image uses a **tarball + ASAN overlay** architecture:

1. **Normal ROCm (tarball):** A pre-built TheRock tarball is downloaded
   and extracted to `/opt/rocm/`. This provides the complete ROCm stack
   (compiler, math libs, ML libs, RCCL, etc.) without any source compilation.
   PyTorch is built against these normal libraries.

2. **ASAN overlay (source build):** TheRock source (pinned to the same
   release tag as the tarball) is cloned and only core_runtime + hip_runtime
   are rebuilt with `linux-release-host-asan` preset. The ASAN-instrumented
   shared libraries (`libamdhip64.so`, `libhsa-runtime64.so`, `libhsakmt.so`,
   `libamd_comgr.so`) are extracted to `/opt/rocm-asan/lib/`.

3. **Runtime overlay:** The entrypoint prepends `/opt/rocm-asan/lib/` to
   `LD_LIBRARY_PATH`, so the ASAN-instrumented libraries shadow the normal
   ones. PyTorch (compiled against normal libs) transparently uses the ASAN
   versions at runtime. The ASAN runtime is loaded as a normal NEEDED
   dependency — not via `LD_PRELOAD`.

This avoids the problems of building PyTorch under ASAN (hipify hangs,
compilation crashes, massive slowdowns) while still catching host-side
memory errors at runtime. The tarball approach reduces initial build time
from ~100 minutes to ~30 minutes.
