# ConSan transform rejection `status=4112` — overlapping anchor patches

Status: open upstream defect in rocjitsu, found while verifying the fixes for
ROCm/rocm-systems#9964, #9970 and #9972.
Affects: `daily-consan-gemm.yaml` (dashboard Tab 2, observed-only, non-gating).
Repro: [`repro/consan_4112_repro.sh`](repro/consan_4112_repro.sh).

## What happens

Under the combined rocjitsu ConSan hook in `record-replay` / `strict` mode on
gfx950, loading a large hipBLASLt f32 Tensile code object (16,265,200 bytes,
490 kernels) now gets all the way through MOI inventory and report planning, and
is then rejected by the transform's final validation:

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
which is fixed. No upstream issue covers 4112 yet.

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

| Case | Today | After 4112 is fixed | After 4112 **and** #9972 are fixed, **with dispatch** |
|---|---|---|---|
| `consan-gemm` (Tab 2) | `error`, exit 92 `consan_strict_load_rejection` | `error`, exit 86 `combined_hook_exit_86` | could reach a real `pass`/`fail` verdict |
| `consan-lds-dispatch` (Tab 2) | `error`, exit 86 | unchanged (different defect, #9972) | `pass`/`fail` |
| `consan-tiny` (Tab 2) | `error`, exit 86 | unchanged (no sites, by design) | unchanged |
| `consan-clean` / `consan-racy` (Tab 1) | `pass` / `fail` | unchanged | unchanged |

The concrete thing that starts working is the **ConSan transform of a large
production code object**: the patched image is produced and the module loads. That
is what the repro script asserts on, and it is the assertion that will flip from
fail to pass.

Turning `consan-gemm` into a genuine pass/fail verdict additionally needs a driver
that dispatches a GEMM kernel from the instrumented module (hipBLASLt, or a
hand-written launcher for one extracted Tensile kernel), plus a resolution to
ROCm/rocm-systems#9972, under which dispatched caller-supplied objects still
capture zero records.

## Reproducing

`repro/consan_4112_repro.sh` is self-contained — no aorta imports, no fixtures
from this repo — so it can be attached to an upstream issue as-is. It extracts
the object from the local public hipBLASLt install, compiles a ~20-line loader,
runs it under the hook, and asserts on the outcome.

```bash
docs/sanitizers/repro/consan_4112_repro.sh --hook /path/to/librocjitsu_dbi_hooks.so
```

It exits `0` when it reproduces the 4112 rejection, `1` when the object loads
cleanly (the defect is fixed), `2` when the environment is unusable, and `3` when
the run failed some other way. Note that on `1` the process itself still exits
86, because the load-only driver never dispatches and strict require-records
fails afterwards — that is the expected post-fix state described above, not a
second defect.

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
