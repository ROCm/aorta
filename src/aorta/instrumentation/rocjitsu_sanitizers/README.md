# RocJITsu kernel sanitizers

Public, typed primitives for selecting important GPU kernels and applying
RocJITsu guardrails without importing customer-derived artifacts.

**Scope.** This module delivers the Phase-1 engine and the `mode: sanitizer`
recipe UX (deterministic selection, exact-entry Waitcheck, fail-closed ConSan
default-detector parsing, versioned reports). The end goal of
[#316](https://github.com/ROCm/aorta/issues/316) -- fully automatic
workload-driven top-kernel execution (profile an uninstrumented model, resolve
the top-K kernels to exact identities, then run scoped ConSan) -- is **not** yet
delivered; the "Not available yet" and "Planned scoped ConSan flow" sections
below track that remaining work.

## Phase 0 capability

Available now as a Python library:

- normalize Magpie kernel summaries and generic dispatch CSVs;
- deterministically select `top_time` or `top_dispatch_count`;
- deduplicate stable kernel identities;
- run standalone Waitcheck on one exact code-object entry at a time;
- run the valid exact-entry Waitcheck CLI and retain its raw log;
- report a Waitcheck scan that hit the backend's diagnostic cap as truncated,
  rather than quoting its capped count as the whole hazard count;
- parse upstream `rj-waitcheck-diagnostic-v1` JSONL for corpus workflows;
- parse output from the combined Waitcheck + ConSan hook: the current `default`
  detector, plus legacy Sampled and Record/Replay logs retained for saved
  bundles captured before RocJITsu simplified its modes;
- preserve per-code-object ConSan coverage and fail closed on timeout, backend
  failure, missing verdicts, or incomplete coverage;
- write and strictly reload experimental `aorta.sanitizer_report/0.1` JSON;
- execute a `mode: sanitizer` recipe end-to-end (selection -> backends ->
  report) via `aorta sweep run --recipe ...` / `execute_sanitizer_run`.

Not available yet:

- automatic Magpie/TraceLens execution;
- automatic RocJITsu provisioning;
- top-K ConSan execution.

The last item is intentionally fail-closed. Current RocJITsu ConSan has no
documented kernel allowlist, so wrapping a model would instrument every
supported code object it loads. A top-K ConSan request therefore returns
`not_checked` with `worklist_scope_unsupported`; it never silently falls back to
whole-application instrumentation.

## Static Waitcheck example

```python
from pathlib import Path
import hashlib

from aorta.instrumentation.rocjitsu_sanitizers import (
    KernelIdentity,
    KernelObservation,
    SelectionRequirement,
    run_sanitizers,
    select_kernels,
)

artifact = Path("/tmp/my_kernel.hsaco")
observations = [
    KernelObservation(
        identity=KernelIdentity(
            name="my_kernel",
            target="gfx950",
            code_object=str(artifact),
            code_object_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
            entry_offset=0x120,
        ),
        total_time_ms=12.5,
        dispatch_count=40,
        sources=("user",),
    )
]
worklist = select_kernels(
    observations,
    requirement=SelectionRequirement.TOP_TIME,
    top_n=1,
)
report = run_sanitizers(
    worklist,
    target="gfx950",
    sanitizers=("waitcheck", "consan"),
    output_dir=Path("./sanitizer-out"),
    waitcheck_binary=Path("/path/to/rj_waitcheck"),
)
```

Waitcheck requires an exact final-code identity: code-object path plus kernel
entry offset, with code-object index when the container has multiple device
images. Fuzzy filename matching is deliberately not supported.

## Provisioning the RocJITsu binaries

`mode: sanitizer` runs need two RocJITsu binaries: the standalone Waitcheck CLI
(`rj_waitcheck`) and the combined Waitcheck + ConSan DBI hook
(`librocjitsu_dbi_hooks.so`). The resolvers discover them from either of two
environment variables (an explicit `waitcheck_binary` / `hook_lib` argument
still wins, and `rj_waitcheck` on `PATH` is a final fallback):

- **`ROCJITSU_PREBUILT`** (recommended) -- points at an unpacked *prebuilt
  bundle* published by ROCm/rocm-systems. The tree is flattened:
  `bin/rj_waitcheck` and `lib/librocjitsu_dbi_hooks.so`. Fetch it with the
  upstream `emulation/rocjitsu/scripts/download_sanitizer_artifacts.py`, which
  resolves the latest successful `rocjitsu-sanitizer-artifacts` run on
  `shared/rocjitsu/sanitizers`, verifies the recorded SHA-256 digests, and
  unpacks the bundle:

  ```bash
  python download_sanitizer_artifacts.py --dest ./rocjitsu-sanitizers
  export ROCJITSU_PREBUILT="$PWD/rocjitsu-sanitizers"
  ```

  GitHub does not serve Actions artifacts anonymously, so the downloader needs a
  token with `actions:read` (from `--token`, `$GITHUB_TOKEN`, `$GH_TOKEN`, or
  `gh auth token`) even though the repository is public: the unauthenticated
  artifact download endpoint returns HTTP 401, and these binaries are not
  published as a GitHub Release, so there is no anonymous download path. (In CI,
  note that the default `GITHUB_TOKEN` is scoped to its own repository and cannot
  read another repo's artifacts cross-repo.) The binaries are built
  in the ROCm manylinux image (glibc 2.28) with zlib/zstd/libgcc/libstdc++
  statically linked, so glibc is their only runtime dependency -- no ROCm SDK is
  required to *run* `rj_waitcheck`. These are ephemeral (default 30-day) run
  artifacts, not durable Releases; record the commit from the bundle's
  `MANIFEST.json` for provenance.

- **`ROCJITSU_BUILD`** -- points at a raw CMake *build tree* from a local
  rocm-systems source build. Here the resolvers expect the build-tree layout
  (`tools/rj_waitcheck` and
  `lib/rocjitsu/src/rocjitsu/hooks/librocjitsu_dbi_hooks.so`).

When both are set, the prebuilt bundle wins.

## Driving ConSan over a JIT-compiled Triton/Gluon kernel

ConSan only runs against a caller-supplied `source.consan_command`; it never
wraps a whole application to reach a few selected kernels. For a code object
built from HIP source that command is a small loader compiled per object with
`-DOBJECT` (`consan_load.hip`). A Triton/Gluon kernel cannot use that pattern:
the `.hsaco` does not exist until the kernel runs, it is keyed by a content hash
under `TRITON_CACHE_DIR`, and it is specific to the image, the GPU target, and
the compiled shapes -- there is nothing to pin in a fixture directory and no
source to compile ([#399](https://github.com/ROCm/aorta/issues/399)).

`scripts/sanitizers/triton_consan_loader.py` is the generic equivalent. It binds
`libamdhip64` through ctypes, so the object path is resolved at *run* time and no
compiler is involved. Point it at a Triton cache entry -- a `.hsaco` plus the
adjacent `.json` Triton writes beside it -- and it loads that object under
whatever hook the parent set in `HSA_TOOLS_LIB`:

```bash
# List what the JIT actually compiled, then emit one shim per code object.
python3 scripts/sanitizers/triton_consan_loader.py list --cache-entry "$TRITON_CACHE_DIR"
python3 scripts/sanitizers/triton_consan_loader.py emit-command \
  --cache-entry "$TRITON_CACHE_DIR" \
  --kernel-name _mfma_lds_mediumm_kernel \
  --copy-object recipes/sanitizers/fixtures/isa/_mfma_lds_mediumm_kernel.hsaco \
  --output recipes/sanitizers/fixtures/bin/consan_triton_kernel
```

Both paths are recipe-relative: a recipe resolves `consan_command` and
`code_object` against its own directory, so writing them under
`recipes/sanitizers/fixtures/` is what makes
`recipes/sanitizers/consan-triton-kernel.yaml` runnable as written.
`--copy-object` lifts the object and its sidecars out of the Triton cache, which
is scratch space that gets evicted and recompiled, and points the shim at the
copy -- so the recipe's `code_object` and the object the shim loads are the same
file, and it still exists on the next run.

`run_consan` executes `consan_command` as a bare argv with no arguments, so
`emit-command` writes a small executable shim with the resolved arguments baked
in -- the run-time analogue of the `-DOBJECT` build step. The shim also carries a
SHA-256 for every input whose bytes decide what runs (the code object, the
metadata, and the launch spec), and `run` re-checks each before touching the
device. That is what makes the `command_sha256` `run_consan` records meaningful:
a Triton cache entry can be evicted and repopulated, so pinning the paths alone
would let a rebuilt cache feed different code to the same recorded command and
selected identity. Re-run `emit-command` after any recompile.
`recipes/sanitizers/consan-triton-kernel.yaml` is a worked example.

Two modes mirror the two committed HIP loaders:

- **`load` (default)** -- `hipModuleLoad` the object and resolve the kernel
  symbol. ConSan instruments a code object when it is loaded, so this is enough
  to reach the kernel, and it needs no knowledge of the launch ABI. Works for
  any Triton kernel.
- **`dispatch`** -- additionally launch the kernel once. This needs the argument
  signature, which Triton does **not** write to the metadata JSON in every
  version (it is absent in 3.7.1), so pass `--launch-spec` with an explicit
  `signature` when the metadata has none. Block size comes from
  `num_warps * warp_size` and dynamic LDS from `shared`.

A `--launch-spec` signature accepts either a JSON object (what Triton emits,
where key order is the argument order) or a JSON array of `[name, type]` pairs.
Prefer the array form when hand-authoring one: key order carries no meaning in
JSON, so an editor or `json.dumps(sort_keys=True)` can silently reorder an
object, and a permutation keeps the packed buffer the same size — so the kernarg
cross-check below cannot catch it and the kernel reads a scalar where a pointer
belongs.

```json
{"signature": [["x_ptr", "*fp32"], ["y_ptr", "*fp32"], ["n_elements", "i32"]]}
```

Scalar arguments default to zero and pointer arguments get a fresh zeroed
buffer, because a synthesized size or stride would index buffers whose real
extents are unknowable and turn a sanitizer run into a memory fault. Use
`--arg NAME=VALUE` / `--buffer-bytes` to make the kernel touch memory. HIP does
not validate the launch buffer against the kernel descriptor, so the loader
cross-checks the packed buffer against `.kernarg_segment_size` from the
`.amdgcn` listing Triton writes beside the object and refuses to dispatch a
buffer that would over-run the segment.

Two limits are worth stating plainly. One logical Triton kernel usually compiles
to several objects (shape-selected variants), and ConSan takes exactly one code
object per run, so harvest one shim and one recipe per object; an ambiguous
selection fails closed and lists the candidates. And the object has to contain
sites ConSan can admit: a kernel whose only memory operations are ordinary
global loads/stores yields `access=0/0` and `applicable=false`, so strict
`moi_require_records` fails closed at exit 86 no matter what the driver does —
which is exactly what `daily-consan-tiny.yaml` documents, and dispatching does
not change it.

The dynamic-coverage half is **no longer blocked upstream**:
[ROCm/rocm-systems#9972](https://github.com/ROCm/rocm-systems/issues/9972)
(record/replay reporting itself as `an inventory-only stub`, so a dispatch
captured no records) is fixed in `15275dad`. On bundle `97c1640b`
`daily-consan-lds-dispatch.yaml` observes a clean pass with `access=5/5`,
`barrier=2/2` and `dynamic_complete=true`, so a Triton object with real LDS or
atomic sites can now reach a genuine verdict.

## AddressSanitizer (ASan)

`sanitizers: [asan]` runs an ASan-instrumented repro and reports every invalid
memory access ASan saw, host or GPU, as a `memory_error` finding: the error class
(`heap-buffer-overflow`, `heap-use-after-free`, ...), the kernel, the source
line, the access, and the line that allocated the buffer. Unlike Waitcheck and
ConSan, ASan is applied when the program is **compiled**, so aorta cannot
instrument a kernel after the fact: the recipe names a build that already has it.

```yaml
source:
  kind: kernel
  kernel: {name: store_index}
  asan_command: fixtures/bin/asan_heap_overflow
  # asan_objects: [bin/real_binary]   # required when asan_command is a script
sanitizers: [asan]
```

Five things have to be true, and each one fails silently when it is not, which
is why aorta checks all five instead of trusting the build:

1. **All of the GPU code is built for an `xnack+` target.** `hipcc -fsanitize=address
   -shared-libsan -g --offload-arch=gfx950:xnack+`. With a plain `gfx950` target
   the compiler drops device ASan with only a warning and the build succeeds;
   `-fno-gpu-sanitize` does the same. Before running anything aorta reads every
   device code object out of the checked binaries (the `asan_command` when it is
   an ELF, every `asan_objects` entry, and a `.hsaco` the kernel identity names)
   and fails closed (`asan_device_not_instrumented`) unless **each one** is
   `xnack+` and instrumented. One is not enough: the runtime loads every
   translation unit's object, and one built without `xnack+` still runs, so an
   unchecked kernel beside a checked one would pass. A compressed bundle, which
   cannot be inspected, fails too, and so does a selected kernel found in none of
   the checked objects (`asan_kernel_not_in_checked_code`). When `asan_command`
   is a script, name the real binaries in `asan_objects`.
2. **The ASan-instrumented ROCm runtime is loaded.** The GPU half reports by a
   hostcall that only `lib/asan/libamdhip64` handles. With the stock runtime a
   real detection aborts with `Hostcall: no handler found for service ID 4` and
   no report (`asan_device_report_lost`). Point `AORTA_ASAN_ROCM_LIB` at the
   instrumented `lib/asan` directory, or `ROCM_ASAN_PATH` at an install root
   (TheRock's name for it); a directory whose `libamdhip64` is not instrumented
   is refused. Without one the check is `not_checked` (`asan_runtime_not_found`).
3. **XNACK is on, and the kernel loaded.** aorta pins `HSA_XNACK=1`; an xnack+
   kernel will not load without it (`hipErrorInvalidDeviceFunction`). A run that
   prints a load failure is an error (`asan_kernel_not_loaded`) whatever its exit
   code, because a repro that swallows the error and exits 0 checked nothing.
4. **The selected kernel was dispatched.** ASan leaves no trace of a kernel that
   ran clean, so aorta pins HIP's own dispatch logging (`AMD_LOG_LEVEL=3`,
   `AMD_LOG_MASK=128` -- decimal; the runtime reads a `0x` value as selecting
   nothing -- and `AMD_LOG_LEVEL_FILE` in the output directory, one file per
   process) and a clean exit passes only if the selected kernel appears in it
   (`asan_kernel_not_dispatched`; no log at all is `asan_dispatch_unverified`).
   The runtime's own messages go to that file too, and are read with the streams.
5. **The node's GPU page migration is sound.** Before the repro, aorta runs the
   run mode's node-health canary (see **Node health** below) under the same ASan
   runtime, into `<output>/asan/node-health-*/node-health/`. A canary that does not
   come back clean makes the check an `error` (`asan_node_unhealthy`) and the repro
   is not run; when no canary can run, a repro that would pass is `not_checked`
   (`asan_node_health_not_verified`) and a memory error still fails. The check's
   `backend` records `node_health`, its reason, and the canary's evidence directory,
   which holds the canary's own `asan_report.json`.

aorta also pins `ASAN_OPTIONS=detect_leaks=0:halt_on_error=1:exitcode=1:symbolize=1:color=never`
(replacing any inherited value, so a parent cannot set `exitcode=0`), prepends
the instrumented libraries to `LD_LIBRARY_PATH`, and preloads the host ASan
runtime so an uninstrumented launcher such as Python can load them. Leak checking
is off because LeakSanitizer reports the stock runtime's own allocations on a
correct program and rewrites the exit code when it does.

Where the runtime comes from:

- ROCm 7.x: the `*-asan` packages (`hip-runtime-amd-asan`, `hsa-rocr-asan`,
  `comgr-asan`, `rocm-core-asan`, and `rocblas-asan`, `hipblaslt-asan`,
  `miopen-hip-asan`, `rccl-asan` ... for library coverage) from
  `repo.radeon.com/rocm/apt/<ver>`. They are published for Ubuntu only. On an
  el9 host they work unpacked into a scratch directory with `dpkg -x`-style
  extraction, given a `libstdc++` with `GLIBCXX_3.4.30` on `LD_LIBRARY_PATH`.
- ROCm 10 (TheRock): nightly ASan tarballs and packages for gfx94X and gfx950
  (`nightly.repo.amd.com/rocm/core/tarball-asan/`); set `ROCM_ASAN_PATH` to the
  extracted tree. None is published for the stable 10.0 the CI image pins, and
  the CI image ships no `lib/asan`, so these recipes are not in the nightly.

What a PASS does **not** cover, measured on gfx950: accesses that skip past the
redzone (a store 1 MiB past a buffer was not caught), `hipMallocManaged`
allocations (an overflow was not caught), and any code not compiled with ASan,
including ROCm library kernels unless the `*-asan` library packages are loaded.
Nor does it cover **buffer-addressed accesses inside an instrumented kernel**:
ASan instruments `global_*`/`flat_*` accesses, and a `buffer_store` gets no
shadow check, so an overflow through one is silently missed (measured in HIP, and
in Triton, whose AMD backend emits them by default; Composable Kernel's
global-memory path is built on them). Hand-written
assembly kernels (Tensile) cannot be instrumented at all. The preflight therefore
proves a code object was *built with* device ASan, not that every access in it is
checked. The dispatch check shows the kernel ran, not that its result was right,
so a repro should still check its own output, as `asan_heap_overflow.hip` does.
Run one process per check: ASan reports from several processes interleaved on one
stream can be misattributed to each other.

ROCm's own ASan library builds cover little of what a production PyTorch workload runs
on MI350, measured on the `*-asan` packages: MIOpen 7.0.2 and 7.2.4 and rocBLAS 7.0.2
carry **no gfx950 code object at all** (gfx90a and gfx942 only), and on gfx942
290 of MIOpen 7.0.2's 294 code objects -- every one of them Composable Kernel code
-- carry no ASan instrumentation.
LDS overflows, `hipMalloc` and `hipHostMalloc` overflows, out-of-bounds reads and
device use-after-free were all caught. `recipes/sanitizers/asan-heap-overflow.yaml`
(expect FAIL) and `asan-clean.yaml` (expect PASS) are the worked pair; the build
commands are at the top of `fixtures/repro/asan_heap_overflow.hip`.

### Running a workload under ASan: `aorta run --sanitizer asan`

The recipe above checks one repro. `aorta run --workload <name> --sanitizer asan`
runs any workload under ASan instead, then says which of the kernels it dispatched
ASan actually checked. A sanitizer has to be in a process from its first
instruction, so the command runs itself twice: the parent stays uninstrumented,
makes `<results-dir>/<workload>/asan/<run>/`, and runs the same `aorta run` with the
ASan environment. That is the four requirements above plus: the HIP and HSA
runtimes preloaded by name (so PyTorch's bundled `libamdhip64` binds to the
instrumented build), PyTorch's caching allocator off, Triton building with ASan and
without buffer instructions, and three records that every process the workload
starts writes into that directory -- ASan's reports (`log_path`), HIP's dispatch
log, and the files the loader loaded (`LD_DEBUG=files`). The parent then writes
`asan_report.json` beside them.

A dispatched kernel counts as checked only when aorta finds it defined in an
ASan-instrumented, xnack+ code object, and in no uninstrumented one, among the files
that process loaded, the code objects the workload declared, and Triton's kernel
cache. Kernels are matched on the full signature HIP logs (`store_index(int*,
int)`), read from each code object's kernel-descriptor symbols. A kernel aorta
cannot find counts as unchecked, so the rule can only under-state coverage.

| exit | verdict | meaning |
|---|---|---|
| 0 | `pass` | no memory error, every dispatched kernel was checked, and the node passed its health check |
| 1 | `fail` / `error` | a memory error; a report that could have been lost; a failed trial; or a node that failed its health check |
| 3 | `not_checked` | no memory error, but some kernels ran from code ASan does not check, or no health check could run |

**Node health.** Device ASan reads its shadow memory, which is ordinary host memory,
through the GPU's XNACK page faults, and the driver may migrate those pages to VRAM and
back. On some MI350 nodes that migration corrupts data, and every check there misfires:
an in-bounds program reports host-side `unknown-crash` errors, and a real error can go
unseen. Nothing in a workload's own report tells that apart from a real result. So before
the workload, the parent builds a small canary, [`asan_canary.hip`](asan_canary.hip),
with `hipcc -O1 -g -fsanitize=address -shared-libsan --offload-arch=<gpu>:xnack+` for
each GPU target the driver lists, into `<run>/node-health/`, and runs it under the same
environment. On every visible GPU it sends known data through fresh anonymous mappings
without huge pages, which is how ASan maps its shadow, with every access in bounds. It is
judged exactly as a workload is: it has to `pass` with its own data checks intact. The
build takes about a second, and the run about ten on an 8-GPU node.

- A canary that reports a memory error, finds its data corrupted, crashes, hangs, or runs
  a kernel ASan cannot be shown to check fails the node. The run is an `error` (exit 1)
  and the workload is **not run**: on that node its result would be an error either way,
  and the workload can take minutes. `asan_report.json` keeps the canary's report under
  `node_health`.
- When no canary can run (no `hipcc`, no GPU in the driver's topology, or a canary that
  does not build), `node_health` is `not_verified`, and a run that would have passed is
  `not_checked` (exit 3). A memory error still fails.
- The canary checks the node when the run starts. The fault was measured to come and go
  on some GPUs, so a long run can still meet it later. A report of host-side
  `unknown-crash` errors inside runtime code (interceptors, the HSA runtime's threads,
  Python's allocator) is worth repeating on another node.

What that means per workload, measured on MI350:

- `hrx` and `hrx_perf` build their own HIP code, so under the mode they build it with
  device ASan (`--offload-arch=<arch>:xnack+`, into `<build_dir>/<arch>-asan`) and
  declare it, and a clean run **passes**. The HRX runtime replaces the HIP runtime
  ASan reports through, so an `HRX_GPU_DRIVER` cell is refused. `aorta run` has no
  option for workload config, and `gpu_arch` defaults to `gfx942`, so on gfx950 set
  it from a recipe or the Python API.
- The PyTorch workloads (`gpu_smoke`, `training`, `inference`, `llm_determinism`,
  `race`) run and report **not checked**: PyTorch's kernels, and those of the ROCm
  libraries it calls, are built without ASan and shipped compressed. Triton kernels
  compiled during the run are built with ASan, but triton-rocm 3.6.0 cannot assemble
  any ASan-instrumented kernel, so on that version a workload that compiles one fails.
- `tokenspeed_serve` does its GPU work in a container, which the mode does not reach.

A workload that loads code objects by file (`hipModuleLoad`) must declare them with
`asan_run.declare_code_objects`, or their kernels count as unchecked: the loader's
record does not list them, and a workload may delete its build before the run ends,
so a declaration records what each file defines when it is declared. A declaration is
evidence only for the program that made it. A loaded file aorta cannot read, or that
the loader recorded by a relative path, could hold the copy of a kernel that ran, so it
leaves that process's kernels unchecked, as compressed device code does. An evidence
file aorta cannot read (an ASan report, HIP log or loader log that is unreadable or a
link) makes the run an error, since it could hold a report; a memory error read from the
other files still fails it. A cell whose
environment, mitigations or `extra_env` would change a setting the mode owns
(`LD_PRELOAD`, `ASAN_OPTIONS`, `HSA_XNACK`, ...) or which GPUs the workload sees
(`HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`, ...) is refused rather than run with the
check silently off, or on GPUs the canary did not check; choose GPUs by setting their
visibility before `aorta run`. A PASS under the mode has the same limits as a recipe
PASS.

## Verdict and health rules

- Waitcheck structured hazard → `warn`.
- ConSan attributed race → `fail`.
- ASan memory error → `fail`, whatever the exit code; a device detection the
  runtime could not report, or a clean exit without the selected kernel's
  dispatch → `error`.
- Backend timeout, launch failure, unexpected exit, missing verdict, or
  incomplete coverage → `error`.
- Missing backend or unsupported worklist scoping → `not_checked`.

Coverage evidence is cross-checked two ways: the aggregate `coverage` counts
must be internally consistent, and the itemized `coverage_site` lines must
reconcile against them. Malformed or contradictory evidence is a parse error.
A site kind the hook *counted* but never itemized is not — it is a coverage
gap, reported as `consan_coverage_incomplete` naming the kind and the count
(`reader 3 barrier sites not itemized: 0 of 12`). Both fail closed, but the gap
keeps the rest of the run's evidence — the per-object counts, why sites failed
to lower, and any race finding — instead of discarding it behind a parse
complaint. The run that motivated this is written up in
[`docs/sanitizers/consan-405-unitemized-barrier-sites.md`](../../../../docs/sanitizers/consan-405-unitemized-barrier-sites.md).
- `pass` only means a requested backend ran healthily and produced no finding;
  a bounded ConSan snapshot is not proof that a program is race-free.

A Waitcheck finding count is not automatically the whole hazard count.
`rj_waitcheck` stops collecting at `--max-diagnostics` (32 per code object by
default, and each code object is its own subprocess) and marks a capped scan by
prefixing the summary count with `>=` (`diagnostics=>=32`). aorta carries that
marker through as `diagnostics_truncated: true` on the kernel result and on the
check, so `sanitizer_report.json` distinguishes a complete count from a floor —
the gpt-oss `_topk_topp_kernel` object scanned at the default cap reports 32 of
its 45 hazards ([#480](https://github.com/ROCm/aorta/issues/480)). Set
`sanitizer_plan.policy.waitcheck_max_diagnostics` to raise the cap for a lane
that needs the complete count; a higher cap costs analysis time, so the flag,
not a hard-coded larger default, is what makes the partial result visible.

A ConSan conflict locates its two accesses by reader and text offset
(`first_instruction` / `second_instruction`, kept in the finding's metadata) and
never names a kernel. The parser joins those offsets to the itemized
`coverage_site` records, which name the container at each offset, and sets
`kernel_name` only when both accesses land in the same kernel. `entry_offset`
stays the kernel's entry offset, as on `KernelIdentity`. A check's
`kernel_results[].findings` repeat its `findings` per kernel, so count one list,
never both ([#451](https://github.com/ROCm/aorta/issues/451)).

The report keeps finding severity and execution completeness separate. For
example, a Waitcheck warning plus scoped ConSan `not_checked` produces
`overall_verdict: warn` and `execution_status: partial`.

## Planned scoped ConSan flow

1. Profile the uninstrumented model/module through Magpie or TraceLens.
2. Resolve top-K observations to stable code-object/kernel identities.
3. Rerun the normal application under the combined hook.
4. Pass the identities through a new RocJITsu allowlist so ConSan instruments
   only the resolved worklist.

This keeps real framework state, arguments, streams, collectives, and fused/JIT
kernels intact without requiring generic kernel replay.
