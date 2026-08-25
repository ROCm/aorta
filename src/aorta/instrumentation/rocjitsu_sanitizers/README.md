# RocJITsu kernel sanitizers

Public, typed primitives for selecting important GPU kernels and applying
RocJITsu guardrails without importing customer-derived artifacts.

**Scope.** This module delivers the Phase-1 engine and the `mode: sanitizer`
recipe UX (deterministic selection, exact-entry Waitcheck, fail-closed ConSan
Record/Replay, versioned reports). The end goal of
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
- parse upstream `rj-waitcheck-diagnostic-v1` JSONL for corpus workflows;
- parse Record/Replay output from the combined Waitcheck + ConSan hook;
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
  --output fixtures/bin/consan_triton_kernel
```

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
selection fails closed and lists the candidates. And on the currently shipping
RocJITsu build, record/replay reports itself as `an inventory-only stub`, so a
dispatch captures no dynamic records and strict policy fails closed with exit 86
— the same outcome the committed `daily-consan-tiny.yaml` and
`daily-consan-lds-dispatch.yaml` lanes document
([ROCm/rocm-systems#9972](https://github.com/ROCm/rocm-systems/issues/9972)).
The loader removes the aorta-side gap; the dynamic-coverage half stays blocked
upstream.

## Verdict and health rules

- Waitcheck structured hazard → `warn`.
- ConSan attributed race → `fail`.
- Backend timeout, launch failure, unexpected exit, missing verdict, or
  incomplete coverage → `error`.
- Missing backend or unsupported worklist scoping → `not_checked`.
- `pass` only means a requested backend ran healthily and produced no finding;
  Record/Replay's bounded snapshot is not proof that a program is race-free.

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
