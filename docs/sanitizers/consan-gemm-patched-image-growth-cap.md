# `consan-gemm` still reports `status=4112` — but for a different reason

**Status: root-caused, not a rocjitsu defect.** The Tab 2 `consan-gemm` row is
rejected by ConSan's *patched-image growth* policy, which is a configurable
capacity ceiling that AORTA has never set. It is not the anchor-overlap defect
of [ROCm/rocm-systems#10378](https://github.com/ROCm/rocm-systems/issues/10378),
even though both surface as `status=4112` / exit 92 /
`consan_strict_load_rejection`.

Read this together with
[`consan-4112-overlapping-anchor-patches.md`](consan-4112-overlapping-anchor-patches.md),
which documents the *other* 4112. That one is genuinely fixed; its predicted
follow-on state never materialised because this ceiling intervenes first.

Affects: `daily-consan-gemm.yaml` (dashboard Tab 2, observed-only, non-gating).

## What the nightly reports

From run
[32967422099](https://github.com/ROCm/aorta/actions/runs/32967422099)
(2026-08-26, aorta `bed2771d`, rocjitsu bundle `97c1640b`, ROCm 7.2.4):

```
ConSan patch end   visited=true modified=false outcome=invalid errors=1 warnings=6 patches=0 patch_ms=151746.121
ConSan MOI first-light probe rejected patched-image file growth:
  required total 1492987904 bytes, limit 419430400 bytes (policy absolute-bytes=419430400)
ConSan load rejection reason=transform-error status=4112 policy=strict action=terminate exit_code=92
```

The transform needs ~1.39 GiB of patched image; the ceiling in force is 400 MiB.
Being 3.56x over, the object is never instrumented — `patches=0`.

**No overlapping-patch complaint appears anywhere in that log.** #10378 is fixed
in this bundle. `status=4112` is a generic `transform-error` bucket, so the
dashboard row is byte-identical to the pre-fix one while the cause underneath it
changed completely. That is the single most misleading thing about this failure,
and it is why the earlier document's prediction reads as though the upstream fix
did not land.

**It is not a timeout.** `timeout_seconds: 6000` is untouched: waitcheck 447 s +
MOI inventory 131 s + patch 152 s is roughly 730 s before the rejection.

## Why the ceiling is exceeded: the fixture grew 11.8x

Every prior sizing and upstream-verification exercise for this case used a
**16,265,200-byte (15.5 MB)** object with 490 kernels, measured on ROCm 7.0.2.2.
The object CI now extracts is **191,935,808 bytes (183 MiB)**:

| | Verification object (ROCm 7.0.2.2) | CI object (ROCm 7.2.4) | Ratio |
|---|--:|--:|--:|
| Unbundled size | 16,265,200 | 191,935,808 | 11.8x |
| Access sites discovered | 68,894 | 637,823 | 9.3x |
| Barrier sites | 3,950 probes | 1,386,944 | — |
| Report buffer required | 513,459,640 | 503,295,496 | ~1.0x |

`prepare_gemm_isa.py` extracts *whichever* heavy f32 SS Tensile bundle the local
ROCm ships for the NT (`Ailk_Bjlk`) layout, so the fixture's size is a per-release
property of hipBLASLt rather than something this repo pins. That is already
recorded in the script's own docstring as of
[#402](https://github.com/ROCm/aorta/pull/402) (~183 MiB on 7.2.4, ~156-168 MiB
on 7.14) — the size drift was known. What was not known is that it pushes the
transform past a capacity ceiling nobody had needed to think about, because at
15.5 MB the 400 MiB ceiling was ~24x larger than anything the object could ask
for.

`daily-consan-gemm.yaml`'s timeout rationale still reasons entirely from the
15.5 MB object, so its "~62-69 min" budget describes a run this case no longer
performs.

## The ceiling is a documented knob, not a defect

Verified by direct experiment on a gfx950 host (ROCm 7.0.2.2, rocjitsu bundle
`7d2c61e7`, `lds.hsaco` at 5,816 bytes) — the *same* rejection can be produced
and removed at will:

| Setting | Resulting policy | Outcome |
|---|---|---|
| *(default)* | `absolute-bytes=402653184` | `outcome=modified-valid patches=13` |
| `RJ_CONSAN_MAX_PATCHED_IMAGE_GROWTH_BYTES=4096` | `absolute-bytes=4096` | **`status=4112` exit 92**, `required total 12288 bytes, limit 4096 bytes` |
| `RJ_CONSAN_MAX_PATCHED_IMAGE_GROWTH_PERCENT=100` | `input-percent=100` | **`status=4112` exit 92**, `limit 5816 bytes` |
| `RJ_CONSAN_MAX_PATCHED_IMAGE_GROWTH_PERCENT=10` | `input-percent=10` | **`status=4112` exit 92**, `limit 581 bytes` |
| `RJ_CONSAN_MAX_PATCHED_IMAGE_GROWTH_BYTES=67108864` | `absolute-bytes=67108864` | `outcome=modified-valid patches=13` |

The forced-low runs reproduce the nightly's signature exactly: `patch end
outcome=invalid errors=1 patches=0`, then the same `first-light probe rejected
patched-image file growth` line, then the same
`reason=transform-error status=4112 policy=strict action=terminate exit_code=92`.

So the rejection is a deliberate, tunable capacity policy with two spellings:

* `RJ_CONSAN_MAX_PATCHED_IMAGE_GROWTH_BYTES=N` → absolute ceiling of `N` bytes;
* `RJ_CONSAN_MAX_PATCHED_IMAGE_GROWTH_PERCENT=P` → ceiling of `P`% of the
  *original input image*, which scales with the object.

AORTA sets neither. `run_consan` pins only `RJ_CONSAN_MODE`, `RJ_CONSAN_POLICY`
and `RJ_CONSAN_LOG`; everything else is inherited from the ambient environment.
Note the default ceiling itself moved between bundles — 402,653,184 (384 MiB) on
`7d2c61e7` versus 419,430,400 (400 MiB) on `97c1640b` — so it is not a constant
to rely on either.

To clear the CI object the ceiling has to admit 1,492,987,904 bytes against a
191,935,808-byte input, i.e. **`PERCENT` ≥ 778** or **`BYTES` ≥ ~1.39 GiB**. The
percent form is the better fit here precisely because the fixture's size is a
hipBLASLt release property.

### Raising the ceiling alone will not turn this row green

Two things still stand between this case and a real verdict, and both should be
settled before anyone sets the knob and expects a pass:

1. **Runtime.** Clearing the ceiling means the object actually gets instrumented
   — 637,823 access sites, 9.3x the site count that took ~62-69 min of wall
   clock on the 15.5 MB object. The 6000 s ceiling is very unlikely to hold, so
   the likely next row is `combined_hook_timeout` rather than a pass. Sizing
   that budget needs a measurement, not an extrapolation.
2. **No dispatch.** `consan_gemm_load` is load-only by construction
   (`consan_load.hip` calls `hipModuleLoad` and stops, because production StreamK
   GEMM kernels need hipBLASLt to launch them). Under `strict`,
   `moi_require_records=true` therefore fails the run on require-records at exit
   86 even after a clean transform. This is the part of the earlier document's
   prediction that remains correct and remains unbuilt.

## What is worth raising upstream

Not a defect report. Two diagnostics points:

1. **`status=4112` conflates unrelated transform failures.** Anchor-range
   overlap (#10378) and a growth-ceiling rejection are different problems with
   different owners and different fixes, and they are indistinguishable from the
   status code, the exit code, and therefore from any dashboard built on them.
   Only the human-readable line above the rejection separates them. A distinct
   status per rejection class — or the reason token echoed into the
   `load rejection` line — would have made this diagnosis immediate.
2. **The expansion factor is worth a sanity check.** Instrumenting a 183 MiB
   object is reported as needing ~1.39 GiB of patched image, a 7.78x expansion.
   That may be entirely expected for full MOI record/replay coverage; it is
   recorded here as an observation rather than a claim.

## Reproducing the ceiling behaviour

No GEMM object or ROCm 7.2.4 install needed — any object with at least one
admitted MOI site shows it. Using the repo's own LDS fixture:

```bash
export PATH="$(python -c "from aorta.instrumentation.rocm_paths import resolve_rocm_roots; print(resolve_rocm_roots().llvm_bin_dir)"):${PATH}"
hipcc --genco --offload-arch=gfx950 recipes/sanitizers/fixtures/kernels/lds_reduce.hip -o /tmp/lds.o
clang-offload-bundler --type=o --unbundle --input=/tmp/lds.o \
    --targets=hipv4-amdgcn-amd-amdhsa--gfx950 --output=/tmp/lds.hsaco
hipcc --offload-arch=gfx950 -DLDS_HSACO='"/tmp/lds.hsaco"' \
    recipes/sanitizers/fixtures/kernels/lds_dispatch.hip -o /tmp/lds_dispatch

env HSA_TOOLS_LIB="${ROCJITSU_PREBUILT}/lib/librocjitsu_dbi_hooks.so" \
    HSA_TOOLS_DISABLE_REGISTER=1 \
    RJ_CONSAN_MODE=record-replay RJ_CONSAN_POLICY=strict RJ_CONSAN_LOG=1 \
    RJ_CONSAN_MAX_PATCHED_IMAGE_GROWTH_BYTES=4096 \
    /tmp/lds_dispatch
```

`ROCJITSU_PREBUILT` must point at an unpacked rocjitsu bundle. Expect exit 92 plus
**both** hook-owned lines — `installed ConSan hook` (proof the hook engaged at all;
without it a bare exit code says nothing) and `first-light probe rejected
patched-image file growth`. Drop the `RJ_CONSAN_MAX_PATCHED_IMAGE_GROWTH_BYTES`
override and the same command transforms cleanly.

## Knock-on: the 4112 reproducer could not tell these apart

[`repro/consan_4112_repro.sh`](repro/consan_4112_repro.sh) decided "reproduced"
from `reason=transform-error status=4112` plus exit 92 — a signature this capacity
rejection produces exactly. On any ROCm shipping a large Tensile bundle it would
therefore have reported the *fixed* overlapping-anchor defect as still present, to
an upstream maintainer, which is the worst direction for that script to be wrong
in. It now requires the `final validation found partially overlapping patch
ranges` diagnostic for a reproduction and reports the growth ceiling as its own
inconclusive outcome with the knob to retry under.

The general lesson, which is why it is worth recording: a status code is not a
defect identity. 4112 is a bucket, and anything that keys a verdict off the bucket
rather than the hook's stated reason will misattribute one defect to another.
