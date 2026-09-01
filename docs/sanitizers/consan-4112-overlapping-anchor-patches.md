# ConSan transform rejection `status=4112` — overlapping anchor patches

**Status: fixed upstream, published, and confirmed consumed.** Fixed in
`dc7c8e04`; a bundle carrying it (and the #9972 fix) went green on 2026-08-24,
and the nightly has since run on `97c1640b` with no overlapping-patch complaint
anywhere in its log. This document is kept as the record of the defect and of
what it predicted.

> ⚠️ **`consan-gemm` on Tab 2 still shows `status=4112` / exit 92 — for an
> unrelated reason.** `status=4112` is a generic `transform-error` bucket, and the
> row it produces is indistinguishable from the pre-fix one. The current
> rejection is ConSan's *patched-image growth* ceiling (400 MiB against a
> ~1.39 GiB requirement), triggered because the extracted fixture grew from
> 15.5 MB to 183 MiB with the move to ROCm 7.2.4. That is a configurable
> capacity policy, not a defect, and it is written up separately in
> [`consan-gemm-patched-image-growth-cap.md`](consan-gemm-patched-image-growth-cap.md).
>
> Consequence for this document: the prediction below — that the fix moves
> `consan-gemm` to exit 86 on strict require-records — **has not been observed in
> CI and cannot be**, because the growth ceiling now intervenes before the
> transform completes. The prediction was verified against the 15.5 MB object on
> a source build, and it remains correct *for that object*. It is not what the
> nightly reports.

Found while verifying the upstream fixes claimed for ROCm/rocm-systems#9964,
#9970 and #9972, and filed as
[ROCm/rocm-systems#10378](https://github.com/ROCm/rocm-systems/issues/10378).
Timeline:

| | |
|---|---|
| #9964, #9970 | fixed in `db0c47df`; verified here |
| 4112 (#10378) | filed from this document; **fixed in `dc7c8e04`**, closed 2026-08-24, verified here |
| #9972 | diagnostics only in `db0c47df`, re-open requested; **fixed in `15275dad`**, closed 2026-08-24, verified here |

The defect measurements below — the ones describing the rejection itself — were
taken on `db0c47df`, the bundle this document was written against and the last
one the nightly consumed. Post-fix figures are labelled with the bundle they come
from. **A bundle carrying both fixes is now published**: after
[ROCm/rocm-systems#10622](https://github.com/ROCm/rocm-systems/issues/10622)
(artifact build red since 2026-08-23, plus a branch-head regression that faulted
any LDS-touching dispatch) was fixed in `4227d40fb5` and closed,
`rocjitsu-sanitizer-artifacts` run 32745941408 went green on 2026-08-24. Since the
downloader selects the newest successful run, the next nightly picks that up
rather than `db0c47df`.

So the `db0c47df` numbers here are historical from the next nightly onward.

> **What actually happened, recorded 2026-08-27.** Everything from "A bundle
> carrying both fixes is now published" down to the end of this section was
> written before the nightly consumed one, and it is kept as the prediction it
> was. Two things in it are wrong as a description of CI. The nightly did not
> land on `4227d40fb5` — the downloader selects the newest successful run, and by
> the time one ran that was `97c1640b`. And `consan-gemm` did **not** stop
> reporting `status=4112`: the anchor-overlap defect is gone from the log, but
> the patched-image growth ceiling now rejects the object first, on the much
> larger fixture ROCm 7.2.4 ships. See
> [`consan-gemm-patched-image-growth-cap.md`](consan-gemm-patched-image-growth-cap.md)
> and the CI column in the table below.

Re-running this document's own reproducer against `4227d40fb5` confirms the fix
end to end **on the 15.5 MB source-build object**, and the prediction held
exactly for it — that object stops reporting `status=4112` and ends on strict
require-records instead:

```
ConSan MOI inventory end   elapsed_ms=246060.502
ConSan patch end  outcome=modified-valid errors=0 warnings=11 patches=75978 patch_ms=137307.900
ConSan coverage   analysis_complete=true access_discovered=68894
[consan_4112_load] loaded and instrumented … (no dispatch)
RJ_CONSAN_MOI_REQUIRE_RECORDS requested, but … no kernel dispatch packet was observed
exit 86 after 4152s
```

Two runs of that path completed in **3696 s and 4152 s** (~62–69 min), emitting an
identical 508,727 log lines, so the spread is host contention rather than workload
variance.

⚠️ **That is roughly 2.9× the 1416 s this case needed on
`db0c47df`, and well past any ceiling sized against it.** The defect used to abort
the run early; now the transform succeeds and the object is genuinely instrumented
(75,978 patches over 68,894 discovered access sites), so the work that follows is
work the old path never reached. Any timeout for this case has to be sized against
the ~69 min figure, not the ~24 min one — see the ceiling discussion in
`daily-consan-gemm.yaml`.

That remains a lower bound rather than a budget: everything in this section was
measured on the 15.5 MB object, and the one CI extracts has 9.3x the access
sites. Nobody has instrumented it end-to-end, because the growth ceiling stops it
first.

Affects: `daily-consan-gemm.yaml` (dashboard Tab 2, observed-only, non-gating).
Repro: [`repro/consan_4112_repro.sh`](repro/consan_4112_repro.sh).

## What happens

This section describes the defect as observed on `db0c47df`, per the status note
above; it is fixed in `dc7c8e04`. Under the combined rocjitsu ConSan hook
in `record-replay` / `strict` mode on gfx950, loading a large hipBLASLt f32
Tensile code object (16,265,200 bytes, 245 kernels) gets all the way through MOI
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

The first column is what the dashboard showed while `db0c47df` was the newest
bundle. The second is measured from source builds of the fixes, not predicted. The
third is what the nightly **actually** reported once it picked up a bundle with
both fixes (`97c1640b`, run 32967422099, 2026-08-26). The fourth remains a
projection — it needs work nobody has done yet.

| Case | On `db0c47df` (through 2026-08-24) | With the `dc7c8e04` / `15275dad` fixes (measured, source build, 15.5 MB object) | Observed in CI on `97c1640b` (183 MiB object) | Additionally **with a dispatching driver** (not built) |
|---|---|---|---|---|
| `consan-gemm` (Tab 2) | `error`, exit 92 `consan_strict_load_rejection` | `error`, exit 86 `combined_hook_exit_86` — transform now succeeds | ❌ **still `error`, exit 92** — different cause: patched-image growth ceiling, see [the growth-cap doc](consan-gemm-patched-image-growth-cap.md) | could reach a real `pass`/`fail` verdict |
| `consan-lds-dispatch` (Tab 2) | `error`, exit 86 | records captured, `dynamic_complete=true`, exit 0 at `STRIDE=16` | ✅ **`pass`**, `access=5/5 barrier=2/2`, `visible_evidence=3216` | already there |
| `consan-tiny` (Tab 2) | `error`, exit 86 | unchanged (no sites, by design) | `error`, exit 86 — as designed | unchanged; a dispatching driver does **not** help (measured) |
| `consan-clean` / `consan-racy` (Tab 1) | `pass` / `fail` | unchanged | `pass` / `fail` | unchanged |

The second column's `consan-gemm` prediction is the one entry that CI never
reached. It was measured against the 15.5 MB ROCm 7.0.2.2 object and holds for it;
the object CI extracts is 11.8x larger and is rejected earlier, on capacity.

The concrete thing that started working is the **ConSan transform of a large
production code object**. On `dc7c8e04` the same object patches cleanly —
`outcome=modified-valid errors=0 patches=75996`, no load rejection, and MOI
inventory down to 436 s from 685 s — which is exactly the assertion the repro
script makes, and it now reports `fixed` rather than `reproduced`.

`consan-gemm` still cannot produce a race verdict, and that part of the original
prediction stands: the driver is load-only, so the run ends on strict
require-records at exit 86. Turning it into a genuine pass/fail additionally needs
a driver that dispatches a GEMM kernel from the instrumented module (hipBLASLt, or
a hand-written launcher for one extracted Tensile kernel). That is unbuilt.

It is **not** the only remaining blocker, and the exit 86 above is what the
15.5 MB object does, not what CI sees. On the 183 MiB object CI extracts, the
growth ceiling rejects the transform first, so the run never reaches
require-records at all — that has to be settled before the dispatching driver
matters. See the [growth-cap doc](consan-gemm-patched-image-growth-cap.md).

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
established.

> **Updated 2026-08-27.** The script used to key "reproduced" off
> `status=4112` + exit 92 alone. Because 4112 is a shared `transform-error`
> bucket, the patched-image growth ceiling produces that same signature — so on a
> modern ROCm base (large Tensile bundle, ~183 MiB object) the script would have
> reported this *fixed* defect as still present. It now requires the
> `final validation found partially overlapping patch ranges` diagnostic for a
> reproduction, and reports a growth-ceiling rejection as inconclusive with the
> knob to retry under. See
> [`consan-gemm-patched-image-growth-cap.md`](consan-gemm-patched-image-growth-cap.md).
> The shared status code is filed upstream as
> [ROCm/rocm-systems#10950](https://github.com/ROCm/rocm-systems/issues/10950).

`1` requires *both* that the module loaded and that the hook ended
the run with its own exit 86 — the load-only driver never dispatches, so strict
require-records is expected to fail afterwards, and that is the post-fix state
described above rather than a second defect. Demanding the hook-owned exit code
as well as the loader's success marker is deliberate: the marker alone would also
appear if no hook had loaded at all, which would otherwise read as "fixed".

The hooked run is bounded by `--timeout`, defaulting to 6000 s. That is sized
against the *slower* of the two outcomes, which is the fixed one: a hook that
still has the defect rejects the object after ~1420 s, but a fixed hook
instruments it and runs for ~4150 s, so a ceiling sized against the defect would
kill the very case it is meant to confirm. Hitting the ceiling reports `3`, not a
hang, which matters because a pre-#9964 hook never terminates MOI inventory for
this object at all.

Both figures are for the 15.5 MB object, and 6000 s is not a budget for anything
larger. The object a modern hipBLASLt ships has 9.3x the access sites and has
never been instrumented end-to-end, because the growth ceiling stops it earlier —
so raising that ceiling without also raising `--timeout` mostly trades one
inconclusive result for another. The script says so at the point it suggests the
retry. If the
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
  unbundled for `hipv4-amdgcn-amd-amdhsa--gfx950` → 16,265,200 bytes, 245 kernels.
  The byte count is the shipped bundle *expanded*: the `.co` is a zlib-compressed
  offload bundle (`CCOB`), so the file hipBLASLt installs is substantially
  smaller than this. The ratio was never measured for this object; the
  comparable ROCm 10 CU256 SS layouts inflate 38-45x on unbundling, but that is
  a different object on a different stack and is not carried over here. The
  kernel count was recorded as 490 until it was found to be a 2x
  double count — `llvm-readelf --symbols` prints `.dynsym` and `.symtab` both, so
  each `.kd` was counted twice. 245 is that correction, not a re-measurement: no
  ROCm 7.0.2.2 image remains on the gate host. Nothing in the analysis above
  depends on the count.
