# ConSan on TokenSpeed's Gluon attention kernels — nothing lowers, and an unitemized barrier count hid it

Filed as [ROCm/aorta#405](https://github.com/ROCm/aorta/issues/405), found while
wiring TokenSpeed's JIT kernels through the Triton ConSan loader
([#403](https://github.com/ROCm/aorta/pull/403), closing
[#399](https://github.com/ROCm/aorta/issues/399)). The loader works; this is
about what ConSan does once it reaches the object.

Two defects, and the second hid the first.

| | Owner | Status |
|---|---|---|
| **1.** No site can be lowered on the attention kernels | RocJITsu (rocm-systems) | open |
| **2.** Barrier sites are counted but not itemized, so aorta could not report defect 1 | aorta | **fixed** — this document's other half |

## Measured on

- gfx950 (cdna4), ROCm 7.0.2.2
- rocjitsu `4227d40fb5b4ea76273589c56dac069af08b7aab`, artifact run 32745941408
  — the same bundle
  [`consan-4112-overlapping-anchor-patches.md`](consan-4112-overlapping-anchor-patches.md)
  records as carrying both the `dc7c8e04` and `15275dad` fixes, so none of this
  is a stale-artifact effect
- `scripts/sanitizers/triton_consan_loader.py` in `load` mode, lenient policy
- objects harvested from TokenSpeed's Triton cache
  (`lightseekorg/tokenspeed-amd:nightly-20260714`)

## Defect 1 — no site lowers on the attention kernels

Gluon **gemm** kernels from the same JIT and the same harvest path instrument
cleanly:

```
access_discovered=77  access_supported=77  access_selected=77  access_patched=77
barrier_discovered=12 barrier_supported=12 barrier_patched=12
analysis verdict static_complete=true  access=77/77 barrier=12/12
```

Gluon **attention** kernels patch nothing:

```
access_discovered=232 access_supported=232 access_selected=232 access_patched=0
                      access_placement_or_lowering_failed=232
barrier_discovered=12 barrier_patched=0 barrier_placement_or_lowering_failed=12
coverage_site ... outcome=placement_or_lowering_failed lowering_reason=instrumentation_patch_missing
```

Every site is discovered, reported *supported*, then fails to lower with
`instrumentation_patch_missing`. Uniformly: 20 of 20 harvested attention
objects, 2502 sites, zero patched, no exceptions. Affected kernels are
`_fwd_kernel`, `_mha_prefill` and `_mha_prefill_sliding`, each in several shape
specializations. Since gemm objects from the same pipeline patch fine this is
kernel-dependent rather than a provisioning problem; the attention kernels are
the more aggressive Gluon code, and `ds_read_b64_tr_b16`, `ds_read2_b64` and
heavy LDS staging dominate their site inventory.

Waitcheck is unaffected and passes on all 20 attention objects, so the kernels
are reachable — ConSan just cannot instrument them. **This half needs a RocJITsu
fix and is not actionable in aorta.**

## Defect 2 — the reporting gap that hid it

On the failing path the hook emits `coverage_site` records for the access sites
but none for the barrier sites, while still reporting `barrier_discovered=12`:

| | `coverage_site` records emitted | aggregate `coverage` record says |
|---|---|---|
| gemm, patched | 77 access + 12 barrier | `access_discovered=77 barrier_discovered=12` |
| attention, unpatched | 232 access + **0 barrier** | `access_discovered=232 barrier_discovered=12` |

`consan_coverage.py` cross-checks itemized sites against the discovered counts
and used to fail closed on any disagreement:

```
consan_output_parse_error: reader <N> barrier site count mismatch
```

The cross-check is right to exist — coverage that cannot be seen should not be
trusted. But a run whose real story is *"0 of 232 sites could be instrumented"*
surfaced as an opaque parser complaint that reads like an aorta bug, the
actionable numbers never reached `sanitizer_report.json`, and the parse error
discarded the rest of the parsed output along with them. All 20 attention runs
reported only this.

### What changed

A site kind the hook *counted* but never itemized is now treated as a coverage
**gap** rather than malformed output. Nothing is trusted that was not seen: the
decision is still rejected, the check is still `error`, and the un-itemized kind
is named with its count. What survives is the evidence:

```
consan_coverage_incomplete: verdict analysis_complete=false;
  verdict static_complete=false;
  access patched/supported mismatch: 0/232;
  barrier patched/supported mismatch: 0/12;
  access placement_or_lowering_failed: 232 instrumentation_patch_missing;
  reader 1 barrier sites not itemized: 0 of 12
```

Three things that were previously lost now reach the report: the per-object
coverage entries (so the dashboard shows `0/232`), the lowering reason every
failed site gave, and any race finding from the same run — a race found while
coverage was incomplete is still a race, and the parse error used to throw it
away.

Partial itemization — some sites of a kind present, but not as many as were
discovered — remains a parse error. That is lossy output with no honest
interpretation, unlike a kind the hook simply never itemizes.

## Reproducing

```bash
export ROCJITSU_PREBUILT=/path/to/rocjitsu/prebuilt
python3 src/aorta/workloads/tokenspeed/harvest_code_objects.py \
    --image lightseekorg/tokenspeed-amd:nightly-20260714 \
    --pytest-suite tokenspeed-kernel/test/ops/test_attention.py --pytest-k mha_prefill \
    --dest /tmp/ts-work/consan-attn \
    --waitcheck "$ROCJITSU_PREBUILT/bin/rj_waitcheck" --consan --consan-limit 1

aorta sweep run --recipe /tmp/ts-work/consan-attn/consan/consan-*.yaml \
    --output-dir /tmp/ts-work/consan-attn-out
```

Swap `--pytest-suite ...` for `--kernel gluon_mm_a16w16_gfx950 --dtype bf16
--dtype-role a` to get the passing gemm comparison.

## What this still blocks

ConSan coverage for TokenSpeed remains gemm-only until defect 1 is fixed
upstream. The difference is that the attention runs now say so in
`sanitizer_report.json` — with the site counts and the lowering reason — instead
of failing as an unreadable parse error.
