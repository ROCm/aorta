# Device AddressSanitizer on gfx950: what it catches, what it needs

**Status: measured on MI350 (gfx950), 2026-10-19.** This is the evidence behind the
`asan` backend's rules in
[`rocjitsu_sanitizers/README.md`](../../src/aorta/instrumentation/rocjitsu_sanitizers/README.md#addresssanitizer-asan).
Every rule there exists because the naive version was measured to pass a run that
checked nothing.

## Configurations

One probe, [`repro/asan_probe.hip`](repro/asan_probe.hip), one defect per mode, built three
ways: `-fsanitize=address -shared-libsan -g -O1` with `--offload-arch=gfx950:xnack+`, the same
with plain `gfx950`, and with no sanitizer. Run on `meta64` nodes (gfx950, kernel 6.9, amdgpu
6.14.14):

| stack | HIP runtime | where |
|---|---|---|
| classic ROCm 7.0.2.2, CentOS 9 | stock | native |
| ROCm 10.0 wheel (TheRock), the pinned CI base `rocm/pytorch:rocm10.0_..._2.13.0@sha256:3174cb70…` | stock (it ships no `lib/asan`) | container |
| classic ROCm 7.0.2.2, CentOS 9 | ROCm 7.0.2 `*-asan` packages, unpacked, plus a `GLIBCXX_3.4.30` libstdc++ | native |
| `rocm/dev-ubuntu-22.04:7.0.2` | ROCm 7.0.2 `*-asan` packages | container |

## Results

With the ASan runtime (both of the last two rows, identical):

| mode | `xnack+` ASan build | plain-`gfx950` ASan build | no sanitizer |
|---|---|---|---|
| `device_oob` (store 1 past `hipMalloc`) | **report** | exit 0 | exit 0 |
| `read_oob` (load 1 past) | **report** | | exit 0 |
| `device_uaf` (store after `hipFree`) | **report** (`heap-use-after-free`, 64 lanes) | | exit 0 |
| `lds_oob` (store 1 past `__shared__`) | **report** (LDS is lowered to a guarded device-malloc buffer) | | exit 0 |
| `pinned_oob` (`hipHostMalloc`) | **report** | | exit 0 |
| `managed_oob` (`hipMallocManaged`) | missed, exit 0 | | exit 0 |
| `device_far` (store 1 MiB past) | missed, exit 0 | | exit 0 |
| `host_oob` (host `malloc`) | **report** | **report** | exit 0 |
| `inbounds` (control) | exit 0, no report | | exit 0 |

With the stock runtime (first two rows): every device detection aborts the process (exit 134)
with `Hostcall: no handler found for service ID 4` and no report. The ROCm 10 stack behaves
exactly like ROCm 7.0.2 here, and its `xnack+` build carries 76 ASan symbols in the device code
object against 0 without the suffix. `device_far` tripped the device check in one of the five
runs (the Ubuntu image with the stock runtime, so detected and unreported), and in none of the
runs that could report. A far access is caught only if it happens to land on poisoned memory.

## Five silent failures, and what the backend does about each

1. **No `:xnack+`**: `clang++: warning: ignoring '-fsanitize=address' option for offload arch
   'gfx950'`, and the build succeeds. The same happens with `-fno-gpu-sanitize` on an `xnack+`
   target: the device code object is `gfx950:xnack+` with 0 ASan symbols while the host half
   carries 20 `__asan_report_*` strings. So the backend reads the device ELF out of each clang
   offload bundle and requires the ASan marker *inside it* (`asan_device_not_instrumented`).
2. **Stock runtime**: detects, then aborts unreported. The resolver refuses a directory whose
   `libamdhip64` has no ASan instrumentation (stock: 0 `__asan_` strings; ASan build: 40), and
   the evaluator recognises the abort (`asan_device_report_lost`). Proven end to end by
   preloading the stock runtime over the ASan one.
3. **XNACK off**: the `xnack+` kernel does not load (`hipErrorInvalidDeviceFunction`). Loud, and
   `HSA_XNACK=1` is pinned anyway. `amdgpu.noretry=1` on the kernel command line does **not**
   prevent per-process XNACK on gfx950; `HSA_XNACK=1 rocminfo` reporting `xnack+` is the test.
4. **LeakSanitizer**: with the stock runtime it reports a correct program's runtime allocations,
   rewrites its exit code (a HIP error exit of 2 came back as 1), and its exit discards buffered
   stdout. `detect_leaks=0` is pinned.
5. **An uninstrumented launcher**: with `lib/asan` on the path, a plain executable (Python, a
   shell) fails with `ASan runtime does not come first in initial library list`. The host
   runtime is preloaded.

## Where the runtime comes from

- **ROCm 7.x**: `repo.radeon.com/rocm/apt/<ver>/pool/main/` carries `hip-runtime-amd-asan`,
  `hsa-rocr-asan`, `comgr-asan`, `rocm-core-asan` and ~40 library `*-asan` packages, for
  **Ubuntu 22.04 only**; the el9 repositories carry none. They need `GLIBCXX_3.4.30` (el9 has
  `.29`) and glibc 2.34.
- **ROCm 10 (TheRock)**: ASan builds for gfx94X and gfx950 at
  `nightly.repo.amd.com/rocm/core/tarball-asan/` (10.1/10.2 nightlies, ~10 GB for gfx950)
  and as native packages, per TheRock's `RELEASES.md`. None for the stable 10.0 the CI image
  pins, which is why the ASan recipes are not in the sanitizer nightly.

## End to end through aorta

`aorta sweep run` on `asan-heap-overflow.yaml` reports `fail` with one
`memory_error/heap-buffer-overflow` in `store_index` at `asan_heap_overflow.hip:29:43`,
allocated at `:34:7`; `asan-clean.yaml` reports `pass`. A build without `:xnack+` →
`error: asan_device_not_instrumented`. A stock runtime directory → `not_checked:
asan_runtime_not_found`. The stock runtime preloaded → `error: asan_device_report_lost`.
Inherited `ASAN_OPTIONS=exitcode=0:halt_on_error=0 HSA_XNACK=0` → still `fail` with the finding.
`test_asan_gpu.py` runs all five cases (not skipped) when a runtime resolves.
