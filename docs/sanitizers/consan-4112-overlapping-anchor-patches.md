# ConSan transform rejection `status=4112` — overlapping anchor patches

**Status: fixed upstream, not yet reachable from a published bundle.**

Found while verifying the upstream fixes claimed for ROCm/rocm-systems#9964,
#9970 and #9972, and filed as
[ROCm/rocm-systems#10378](https://github.com/ROCm/rocm-systems/issues/10378).
Timeline:

| | |
|---|---|
| #9964, #9970 | fixed in `db0c47df`; verified here |
| 4112 (#10378) | filed from this document; **fixed in `dc7c8e04`**, closed 2026-08-24, verified here |
| #9972 | diagnostics only in `db0c47df`, re-open requested; **fixed in `15275dad`**, closed 2026-08-24, verified here |

Every measurement below was taken on `db0c47df`, which is the state this document
was written against and still the behaviour a consumer sees, because **no
published bundle carries either fix**: `rocjitsu-sanitizer-artifacts` has been red
since 2026-08-23, so the newest green bundle predates the #9972 fix. Branch head
additionally GPU-faults any LDS-touching dispatch with the hook merely loaded.
Both blockers are tracked in
[ROCm/rocm-systems#10622](https://github.com/ROCm/rocm-systems/issues/10622); the
fixes themselves were verified from source builds. So nothing on the dashboard
changes until a green bundle ships.

Affects: `daily-consan-gemm.yaml` (dashboard Tab 2, observed-only, non-gating).
Repro: [`repro/consan_4112_repro.sh`](repro/consan_4112_repro.sh).

## What happens

This section describes the defect as observed on `db0c47df` — still what a
consumer gets, per the status note above. Under the combined rocjitsu ConSan hook
in `record-replay` / `strict` mode on gfx950, loading a large hipBLASLt f32
Tensile code object (16,265,200 bytes, 490 kernels) gets all the way through MOI
inventory and report planning, and is then rejected by the transform's final
validation:

```
ConSan MOI emitted 3950 barrier record probe(s)
ConSan final validation found partially overlapping patch ranges:
  existing=4447720-4447756 kind=45 role=anchor patch=908 source=4447720->182537992
  next=4447732-4447768     kind=45 role=anchor patch=910 source=4447732->182538056
ConSan load rejection reader=36851488 reason=transform-error status=4112 policy=strict action=terminate exit_code=92
```

Two `kind=45` `role=anchor` patches claim overlapping byte ranges: patch 908 owns
`4447720-4447756` and patch 910 owns `4447732-4447768`, overlapping by 24 bytes.
Strict policy correctly terminates rather than emitting a patched image built
from conflicting rewrites, so this is a fail-closed outcome, not a false pass.

`status=4112` is distinct from the `status=4104` ABI-capacity rejection of #9970,
which is fixed. 4112 was tracked upstream as ROCm/rocm-systems#10378 and is fixed
there in `dc7c8e04`.

## Why it only became visible now

This object could not previously reach the patch stage:

| Stage | Before the #9964 / #9970 fixes | After (rocjitsu `db0c47df`) |
|---|---|---|
| MOI inventory | never terminated (killed at 1800 s) | ends in 658,971 ms |
| Auto report plan | n/a — never reached | `required_bytes=513459640` ≤ `cap_bytes=536870912`, allocated |
| Patch + validation | n/a — never reached | **rejected, `status=4112`** |

So 4112 is not a regression. It is the next defect in line, previously masked by
a non-terminating earlier phase.

## Which test unblocks when 4112 is fixed — and which does not

Short answer: **fixing 4112 will not turn any dashboard row green.** It moves
`consan-gemm` from one fail-closed reason to a different one.

`consan_gemm_load` is deliberately load-only. Production StreamK GEMM kernels
need hipBLASLt to launch them, so the fixture does `hipModuleLoad` and nothing
else (see the header comment in `recipes/sanitizers/fixtures/kernels/consan_load.hip`).
Under `strict`, `moi_require_records=true` then fails the run because no dispatch
was ever observed.

That prediction is measured, not assumed. Loading `lds.hsaco` — an object that
*does* have admitted MOI sites (`access_patched=5`, `barrier_patched=2`) and that
transforms cleanly today — through the same load-only driver gives:

```
ConSan MOI auto report buffer … allocation_outcome=allocated
ConSan coverage … access_discovered=5 access_supported=5 access_selected=5 access_patched=5
RJ_CONSAN_MOI_REQUIRE_RECORDS requested, but 1 auto MOI report buffer(s) contained
zero visible records and no kernel dispatch packet was observed
exit 86
```

No load rejection: the transform succeeded and the failure moved to the
require-records check. So:

The first column is what the nightly still reports, since it is the newest bundle
a consumer can get. The second is measured from source builds of the fixes, not
predicted. The third remains a projection — it needs work nobody has done yet.

| Case | On `db0c47df` (what the dashboard shows) | With the `dc7c8e04` / `15275dad` fixes (measured, source build) | Additionally **with a dispatching driver** (not built) |
|---|---|---|---|
| `consan-gemm` (Tab 2) | `error`, exit 92 `consan_strict_load_rejection` | `error`, exit 86 `combined_hook_exit_86` — transform now succeeds | could reach a real `pass`/`fail` verdict |
| `consan-lds-dispatch` (Tab 2) | `error`, exit 86 | records captured, `dynamic_complete=true`, exit 0 at `STRIDE=16` | `pass`/`fail` |
| `consan-tiny` (Tab 2) | `error`, exit 86 | unchanged (no sites, by design) | unchanged |
| `consan-clean` / `consan-racy` (Tab 1) | `pass` / `fail` | unchanged | unchanged |

The concrete thing that started working is the **ConSan transform of a large
production code object**. On `dc7c8e04` the same object patches cleanly —
`outcome=modified-valid errors=0 patches=75996`, no load rejection, and MOI
inventory down to 436 s from 685 s — which is exactly the assertion the repro
script makes, and it now reports `fixed` rather than `reproduced`.

`consan-gemm` still cannot produce a race verdict, and that part of the original
prediction stands: the driver is load-only, so the run ends on strict
require-records at exit 86. Turning it into a genuine pass/fail additionally needs
a driver that dispatches a GEMM kernel from the instrumented module (hipBLASLt, or
a hand-written launcher for one extracted Tensile kernel). That is unbuilt, and it
is the only remaining blocker now that #9972 is fixed.

## Reproducing

The reproducer is a **pair of files that must travel together**:
`repro/consan_4112_repro.sh` and `repro/consan_4112_load.hip`. The script compiles
the loader from the `.hip` at runtime and looks for it as a sibling, so shipping
only the shell script fails immediately with `loader source not found next to this
script`. Both are attached to the upstream issue for that reason.

Together they are self-contained — no aorta imports, no fixtures from this repo.
The script extracts the object from the local public hipBLASLt install, compiles
the ~20-line loader, runs it under the hook, and asserts on the outcome.

```bash
docs/sanitizers/repro/consan_4112_repro.sh --hook /path/to/librocjitsu_dbi_hooks.so
```

It exits `0` when it reproduces the 4112 rejection, `1` when the defect is fixed,
`2` when the environment is unusable, and `3` when no verdict could be
established. `1` requires *both* that the module loaded and that the hook ended
the run with its own exit 86 — the load-only driver never dispatches, so strict
require-records is expected to fail afterwards, and that is the post-fix state
described above rather than a second defect. Demanding the hook-owned exit code
as well as the loader's success marker is deliberate: the marker alone would also
appear if no hook had loaded at all, which would otherwise read as "fixed".

The hooked run is bounded by `--timeout` (default 2400 s, against a measured
~1290 s end-to-end); hitting the ceiling reports `3`, not a hang, which matters
because a pre-#9964 hook never terminates MOI inventory for this object. If the
local hipBLASLt does not carry the Tensile bundle — it has shipped both flat
under `library/` and under `library/gfx950/`, and slim installs may omit it —
pass `--object` with an already-unbundled gfx950 code object.

## Environment where this was observed

- gfx950 (cdna4), ROCm 7.0.2.2, 8 GPUs
- rocjitsu `db0c47df6bc127c4df6f0283d00b42e69deef2bf`, prebuilt bundle from
  ROCm/rocm-systems Actions run 31716952381
- `RJ_CONSAN_MODE=record-replay`, `RJ_CONSAN_POLICY=strict`,
  `moi_profile=standard-v1`
- Object: `TensileLibrary_SS_SS_HA_Bias_SAV_UA_Type_SS_Contraction_l_Ailk_Bjlk_Cijk_Dijk_gfx950.co`,
  unbundled for `hipv4-amdgcn-amd-amdhsa--gfx950` → 16,265,200 bytes, 490 kernels
