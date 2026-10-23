# Device AddressSanitizer on gfx950: what it catches, what it needs

**Status: measured on MI350 (gfx950), 2026-10-19.** This is the evidence behind the
`asan` backend's rules in
[`rocjitsu_sanitizers/README.md`](../../src/aorta/instrumentation/rocjitsu_sanitizers/README.md#addresssanitizer-asan).
Every rule there exists because the naive version was measured to pass a run that
checked nothing.

## Configurations

One probe, [`repro/asan_probe.hip`](repro/asan_probe.hip), one defect per mode, built three
ways: `-fsanitize=address -shared-libsan -g -O1` with `--offload-arch=gfx950:xnack+`, the same
with plain `gfx950`, and with no sanitizer. Run on MI350 nodes (gfx950, kernel 6.9, amdgpu
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

## Cost

`examples/profiling/rocprof/hip-gemm/gemm.hip` (tiled SGEMM, LDS-heavy, 20 iterations, three
reps each, all verifying their output), job 83211:

| build | n=1024 kernel | n=2048 kernel | process wall |
|---|---|---|---|
| plain `gfx950` | 0.143 ms | 0.996 ms | ~0.4 s |
| plain, `HSA_XNACK=1` | 0.142 ms | 0.995 ms | ~0.45 s |
| `gfx950:xnack+`, no ASan | 0.142 ms | 0.998 ms | ~0.45 s |
| `gfx950:xnack+` ASan, ASan runtime | 24.9–27.4 ms | 66.9–75.2 ms | 1.4–2.9 s |

XNACK costs nothing measurable here. Device ASan costs 70–185× on this kernel, more than
host ASan's usual ~2×, plausibly because LDS is lowered to instrumented global memory. One
kernel is not a model, but the order of magnitude says device ASan is for targeted repros and
small workloads, not whole training runs.

## Five silent failures, and what the backend does about each

1. **No `:xnack+`, anywhere**: `clang++: warning: ignoring '-fsanitize=address' option for
   offload arch 'gfx950'`, and the build succeeds. The same happens with `-fno-gpu-sanitize` on an
   `xnack+` target: the device code object is `gfx950:xnack+` with 0 ASan symbols while the host
   half carries 20 `__asan_report_*` strings.
   - **One unchecked translation unit is enough.** A binary whose instrumented `gfx950:xnack+` TU
     launches a kernel from a TU built for plain `gfx950` ran that kernel's overflow to exit 0
     under the full ASan setup (job 85339).
   - So the backend reads every device ELF out of every clang offload bundle and requires each
     one to be `xnack+` with the ASan marker *inside it*. It also refuses compressed bundles and
     a selected kernel found in none of them (`asan_device_not_instrumented`,
     `asan_kernel_not_in_checked_code`).
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

## Buffer-addressed accesses are not checked

Device ASan instruments `global_*`/`flat_*` accesses, not buffer instructions. Compiling one HIP
kernel two ways under `-fsanitize=address` for `gfx950:xnack+` (ROCm 7.0.2.2, clang 20) shows
it in the instructions:

- **`out[i] = i`:** a `flat_load_sbyte` of the shadow byte, a compare and a branch to the report
  path, before the `global_store_dword`.
- **The same store via `__builtin_amdgcn_raw_buffer_store_b32`:** a bare `buffer_store_dword`
  with no check.

Composable Kernel builds its global-memory path on buffer intrinsics (its
`amd_buffer_addressing` layer; not measured here), and Triton's AMD backend lowers `tl.store` to
them by default (measured below), so an instrumented code object can still have unchecked
accesses.

## Proving the kernel ran

A clean ASan run leaves no trace, so a repro that never launched its kernel exits 0 like one
that ran clean. HIP's own log records dispatches. At `AMD_LOG_LEVEL=3` with
`AMD_LOG_MASK=128` (`LOG_KERN`), each launch writes a `ShaderName : <kernel signature>` line, and
`AMD_LOG_LEVEL_FILE=<path>` sends it to `<path>_<pid>` per process. The ASan clean fixture's log
is 3.6 KB with one `ShaderName : store_index(int*, int)` line, and zero lines with XNACK off
(jobs 85370, 85390).

Two traps:
- **`AMD_LOG_MASK` is read as decimal.** Every `0x`-prefixed value selected nothing but the
  always-on lines.
- **The runtime writes its own messages to that file too**, including `Hostcall: no handler
  found for service ID 4`, so the backend reads the file for every runtime signal (job 85399
  caught it reading only stderr).

A binary that holds `store_index` and never launches it now gives `error:
asan_kernel_not_dispatched` (job 85402).

## ROCm's ASan library builds on MI350

The `*-asan` library packages were unpacked, every offload bundle decompressed with
`clang-offload-bundler`, and the code objects counted:

| package | bundles | targets | gfx942 objects with ASan |
|---|---|---|---|
| `miopen-hip-asan` 7.0.2 | 294 | gfx90a, gfx942, **no gfx950** | **4 of 294**, every object Composable Kernel code |
| `miopen-hip-asan` 7.2.4 | 335 | gfx90a, gfx942, **no gfx950** | not counted |
| `rocblas-asan` 7.0.2 | 64 | gfx90a, gfx942, **no gfx950** | 64 of 64 |

Across the 12,290 Composable Kernel functions in MIOpen 7.0.2's gfx942 objects,
**89.5% of global, flat and buffer memory instructions are buffer instructions** (670,372
against 78,517), the kind ASan does not check. MIOpen's 668 other functions use none. So on
MI350 these libraries carry no device code at all, and on MI300X ASan would check almost none
of MIOpen's CK kernels. TheRock's ASan builds (nightly, gfx950) were not measured.

## Python, PyTorch and Triton

`aorta run --sanitizer asan` (the last section) applies every setting below except the
named-barrier workaround.

An off-by-one `tl.store` in a Triton kernel launched from PyTorch (torch 2.10.0+rocm7.0,
triton-rocm 3.6.0, `rocm/dev-ubuntu-22.04:7.0.2` + the 7.0.2 `*-asan` packages; jobs 83210 and
83218, two nodes) **is** caught, and the report names the kernel and line (`store_index_off_by_one
at triton_oob.py:173:27`, global index 1000). It took more than the obvious settings
(`TRITON_ENABLE_ASAN=1`, `HSA_XNACK=1`, the ASan runtime, the caching allocator off). Each of the
items below has a control that removes it and fails:

| needed | without it |
|---|---|
| `AMDGCN_USE_BUFFER_OPS=0` | **silently missed, exit 0**: the store becomes a `buffer_store_dword` with no check, in a kernel holding 949 `__asan_` references from the linked device runtime |
| `PYTORCH_NO_HIP_MEMORY_CACHING=1` | missed: the 4000-byte tensor sits inside a 2 MiB cached segment, which is all ASan knows |
| `LD_PRELOAD=<host asan rt>:libamdhip64.so:libhsa-runtime64.so`, the HIP and HSA runtimes by **bare name** | torch's libraries carry `DT_RPATH=$ORIGIN` to their bundled stock runtime, which `LD_LIBRARY_PATH` cannot override. A full-path preload loads both copies; Triton launches through torch's (it probes `dlopen("libamdhip64.so", RTLD_NOLOAD)`) and the detection aborts unreported. With only the host runtime preloaded, ASan's allocator interceptor aborts under the stock HSA runtime |
| `torch/lib` on `LD_LIBRARY_PATH`, after `lib/asan` | `torch.cuda` init fails: the ASan runtime interposes `dlopen`, so torch's `$ORIGIN`-relative `dlopen("libcaffe2_nvrtc.so")` loses its RPATH |
| defining `amdgpu.max_num_named_barrier` in the kernel assembly | triton-rocm 3.6.0 cannot assemble any ASan kernel on gfx950 (`expression could not be evaluated`). The ASan module constructor's `amdgcn.device.init` references the symbol and the backend never defines it. An upstream defect, worked around in-process for the measurement |

Cost on that stack: process wall time ~3× cold / ~1.8× warm, a bandwidth-bound 256 MiB store
1.4–1.7×, and ~3 ms per synchronized launch against ~22 µs. Device ASan's cost is
kernel-dependent: compare the 70–185× of the LDS-heavy GEMM above.

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
`test_asan_gpu.py` runs all five cases (not skipped) when a runtime resolves. The mixed-TU
binary above → `error: asan_device_not_instrumented`, refused before it runs (job 85339; the
same matrix otherwise gave identical verdicts to job 83182).

## aorta's own workloads under ASan: the run mode

`aorta run --sanitizer asan` (see the package README) runs a workload under ASan and judges
each dispatched kernel. Measured on gfx950, ROCm 7.0.2 and its ASan runtime:

| case | verdict | how the kernels were attributed |
|---|---|---|
| `hrx` static probe, `gfx950` | `pass` | `add100(float*, float const*, unsigned long)`, from the probe binary the loader loaded |
| `hrx` module probe | `pass` | `add100`, from the `hipModuleLoad`ed code object `hrx` declared |
| `hrx` graph probe | `pass` | the graph-launched kernel is logged and attributed like any other |
| `hrx_perf` triad | `pass` | `fill` and `triad`, both checked |
| `hrx` running a planted `i <= n` overflow | `fail` | device `heap-buffer-overflow` at `hrx_probe_oob.cpp:12:24`, read from the per-process `log_path` file |
| the same, stock runtime | `error` | `Hostcall: no handler found for service ID 4` in the HIP log |
| an `extra_env` that sets `LD_PRELOAD` | refused | before any trial ran |
| `gpu_smoke` through the CLI (container) | `not_checked`, exit 3 | 0 of 3 kernels: `vectorized_elementwise_kernel` (fill, add) and `reduce_kernel`, defined in compressed device code in `libtorch_hip`, `librocblas`, `libMIOpen`, `librccl`, `librocsolver`, `librocsparse` |

Native jobs 85755 and 85760; container job 85756. The `gpu_smoke` control without
`--sanitizer` passes in the same container, and a planted overflow run under the mode's
environment there fails, so its `not_checked` is about coverage.

Two things the first measurement got wrong, both now covered by tests. `python -m aorta.cli`
cannot start the child (`aorta.cli` is a package without `__main__`), which a test that only
compared the command line let through. And job 85755 reported every clean `hrx` run
`not_checked`: `hrx` deletes its temporary build directory in `cleanup()`, so by the time the
parent read the evidence the declared probes were gone. A declaration now records what each
file defines when it is declared, and job 85760 is the measurement after that change.

## Node health: GPU page migration that corrupts memory

Device ASan reads its shadow memory, which is ordinary anonymous host memory, through XNACK
page faults, and the driver can migrate those pages into VRAM and back. On some MI350 nodes
that migration corrupts data. A pure-HIP check with no ASan writes known patterns through
fresh `MAP_NORESERVE` mappings without huge pages, reads them on the GPU, and reads them back
on the CPU:

- On affected GPUs every word comes back wrong once the GPU has faulted the pages into VRAM.
  The same pages advised to stay in host memory (`hipMemAdvise`, preferred location CPU)
  round-trip intact, so the fault is in migration.
- On other affected GPUs the patterns survive, but pages nothing wrote read back nonzero.
  Most of ASan's shadow is such pages.

The fault is per GPU and comes and goes. One node had it on half of its GPUs. Another passed
the check, then hung minutes later in the driver's SVM restore path under ASan load. Kernel,
amdgpu version, boot parameters, GPU firmware and sysctls were identical on affected and sound
nodes.

On an affected node ASan is wrong in both directions. Measured with `test_asan_gpu.py` (job
92263), the clean recipe **failed** on a bogus report, and the planted-overflow recipe
**passed**: the overflow went unseen. The bogus reports are `unknown-crash`, on the host
(interceptors, the HSA runtime, Python's allocator) or on the device, and their shadow-byte
dumps are full of unrelated page contents.

The canary, `asan_canary.hip`, is that pure-HIP check built with device ASan. The run mode runs
it before the workload, and the recipe backend before the repro, under the same ASan runtime.
With the ROCm 7.0.2 ASan runtime:

| node, by the pure-HIP check | canary on its own, 3 runs each | time |
|---|---|---|
| affected (two nodes, jobs 92222 and 92223) | `fail` 3 of 3: `unknown-crash` on the CPU's first read of a page the GPU had touched | 1–2 s |
| sound (job 92224) | `pass` 3 of 3: 4 of 4 kernels checked on 8 GPUs | 8–10 s |

Through the run mode at the commit that added the canary, on one sound node and two affected
ones, each job starting with the pure-HIP check as a control (jobs 92282, 92283, 92284):

| | sound node | affected nodes |
|---|---|---|
| `gpu_smoke` through the CLI, ASan PyTorch | `pass`, 3 of 3 kernels checked, node verified in 9.3 s | `error`, exit 1, workload not run, in 2.1 s |
| a planted overflow, ASan PyTorch | `fail`: `heap-buffer-overflow` | `error`: node health |
| `hrx` static probe, native | `pass`, node verified in 9.7 s | `error`, workload not run |
| `test_asan_gpu.py` | 7 passed | the canary's test fails, and so do both recipe tests: the overflow recipe passes |

Through the recipe backend once it runs the canary too, on one sound node and one affected node,
each job starting with the pure-HIP check as a control (jobs 92455 and 92456):

| | sound node | affected node |
|---|---|---|
| `asan-heap-overflow.yaml` | `fail`: device `heap-buffer-overflow`, node verified, 11 s | `error`: node health, repro not run, 2 s |
| `asan-clean.yaml` | `pass`, node verified, 11 s | `error`: node health, repro not run, 2 s |
| `test_asan_gpu.py` | 7 passed | 3 failed (both recipe tests now on `error`, not a wrong verdict) and 4 passed |

Built without ASan, the canary is the pure-HIP check on its own:
`hipcc --offload-arch=gfx950:xnack+ asan_canary.hip -o canary && HSA_XNACK=1 ./canary` exits 0
on the sound node and 3 on an affected one, in about 4.5 s (jobs 92293 and 92294).

## A prebuilt ASan PyTorch image

Building PyTorch from source with device ASan takes about 25 minutes, so `docker/Dockerfile.asan-pytorch`
packages it. **It is a testing convenience with no guarantees** -- not a supported PyTorch build, not
tracked against PyTorch or ROCm releases, and refreshed only every few months. If you need guarantees,
build your own from the file.

The build context is the repo root, so the image installs aorta from the checkout and matches the commit
it was built from. The four ROCm 7.0.2 ASan runtime packages (`comgr-asan`, `hip-runtime-amd-asan`,
`hsa-rocr-asan`, `rocm-core-asan`) are not in the base image; the build fetches them from `repo.radeon.com`
at pinned versions and verifies them against `docker/asan-pytorch/SHA256SUMS`, failing on a mismatch.

```sh
# from the repo root
docker build -f docker/Dockerfile.asan-pytorch -t aorta-asan-pytorch:2.10.0-rocm7.0.2-gfx950 .

docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
    --security-opt seccomp=unconfined aorta-asan-pytorch:2.10.0-rocm7.0.2-gfx950 \
    aorta run --workload gpu_smoke --sanitizer asan
```

To run as a non-root UID, also pass `--user $(id -u):$(id -g) -e HOME=/tmp -w /tmp` (or bind-mount a
writable directory and point `-w` at it): `aorta run` writes `./results` under the working directory,
and the image's default working directory is the root-owned `/`, so `--user` alone fails with
`PermissionError: '/results'`. A plain `import torch` fails outside the run
mode (`undefined symbol: __asan_...`); it works only under `aorta run --sanitizer asan`, which preloads
the ASan runtime. Device ASan depends on sound GPU<->host page migration, so run it only on a healthy
node -- the run mode's canary checks this first (see "Node health" above), and `gpu_smoke` must come back
`pass` with 3 of 3 kernels checked before any other result on that node can be trusted.
