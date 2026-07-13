# NaN Logger v2 (June 29) — Usage Guide (staged runs)

**Target NaN:** `emb_proj.projections.10.layers.0` **INPUT** goes bad for one step
(`3.13e36` + 1 NaN + 4 huge), which poisons the weight and cascades to the
step-999 Shampoo eigendecomp crash. Leading hypothesis: a freed buffer is handed
to `projections.10`'s input while a kernel on another CUDA stream is still writing
it (cross-stream allocator free→reuse aliasing).

This guide walks the new flags as a **staged escalation**: start with the run that
is *least* likely to disturb the race (and gives basic info), and only escalate to
heavier instrumentation once the NaN is confirmed to still reproduce. Each round is
a self-contained command you can copy/paste.

---

## Why staged? (the perturbation ladder)

The logger is built so its measurement work is on-GPU and drained once per step —
no per-op host sync. But some flags still change timing or memory lifetime, and the
target bug is timing-sensitive. Order of increasing risk of *changing/suppressing*
the NaN:

| Flag / setting | What it costs | Can it hide the bug? |
| --- | --- | --- |
| `NANLOG_ADDR` (default on) | host-side pointer read | No — zero GPU/timing effect |
| `NANLOG_LOCATE`, `NANLOG_BAD_VALUES` | extra on-GPU reductions, same single drain | Very unlikely |
| extra `NANLOG_CHANNELS` (`input`, `weight`…) | more on-GPU reductions, same drain | Unlikely |
| wider `NANLOG_WATCH_TYPES` | more layers reduced per step | Unlikely |
| `NANLOG_SAMPLE_EVERY=1`, `NANLOG_PRE_CONTEXT` | more disk writes + host memory | Unlikely (off GPU path) |
| `NANLOG_ALLOC_SNAPSHOT=1` | ~10% step slowdown (per-alloc stack capture) | **Maybe** — shifts overall timing / race window |
| `NANLOG_DUMP_TENSOR=1` | holds one extra tensor ref per step | **Maybe** — delays free→reuse, can *suppress* the aliasing write |

Rule of thumb: **Round 1 first** to confirm the NaN and name the donor with minimal
disturbance; **Round 2** for the definitive allocator proof (~10% timing shift);
**Round 3 last** for the full tensor dump, since it is the most likely to move the
race window / suppress the bug.

### Each round introduces one new June29 feature (cumulatively)

The three June29-new capabilities are staged one per round, in increasing
perturbation order, and **flags are cumulative** — each round keeps everything from
the previous round and turns on one more:

| Round | New June29 feature turned on | Carried forward |
| --- | --- | --- |
| 1 | `NANLOG_BAD_VALUES` + expanded watch scope (wide `WATCH_TYPES`) | `NANLOG_ADDR`, `NANLOG_LOCATE` (June26) |
| 2 | `NANLOG_ALLOC_SNAPSHOT` | all of Round 1 |
| 3 | `NANLOG_DUMP_TENSOR` | all of Round 2 |

Cumulative matters because `ALLOC_SNAPSHOT` and `DUMP_TENSOR` are only interpretable
*with* the Round-1 fields: the allocator trace is proof only when its addresses can
be tied to `projections.10`'s input via `NANLOG_ADDR` + the wide scope, and a dumped
`.pt` is meaningless without the JSONL naming its layer/step/position.

---

## One-time setup (all rounds)

Drop `instrument_nan_logger.py` anywhere on the training host, then run your
existing reproducer **through** it (the wrapper arms hooks, then runs your script
via `runpy`; no edit to the training script needed):

```bash
python3 instrument_nan_logger.py /path/to/your_repro.py [args...]
```

If you cannot change the launch command (e.g. a fixed `torchrun` line), instead add
3 lines at the **very top** of the training script, before the model is built:

```python
import sys
sys.path.insert(0, "/path/to/nan_logger_v2")   # dir holding instrument_nan_logger.py
import instrument_nan_logger  # noqa: F401  (import arms the hooks)
```

**Pre-flight self-check (every run).** In the startup log you must see:

```
nanlog: attached layer hooks to N module(s) (... ); channels=[...]; capture_addr=True; ...
```

- `N` must be **> 0**. With the wide `NANLOG_WATCH_TYPES` below it is **~156**; with
  `NANLOG_WATCH_TYPES=Linear` it is **~99**.
- If `N == 0`, the model is not going through `DistributedModelParallel` and hooks
  did not bind — fix that before spending a run.
- A **clean** run ends with `first_bad: null` in `summary_rank*.json`. That is the
  logger working correctly and finding nothing, not a failure.

---

## Round 1 — `NANLOG_BAD_VALUES` + wide watch scope (confirm + victim + donor)

**Goal:** prove the NaN still reproduces with the logger attached, name the
first-bad layer/step, confirm the aliasing signature on the victim
(`bad_rows == 1`, exact bad position/value), **and** name the aliasing donor from
addresses. This is the lightest new-feature set (all on-GPU, one drain, no host
sync), so if the race is fragile it still fires.

```bash
NANLOG_DIR=/output/run1_bad_values_widescope \
NANLOG_CHANNELS=act,input,igrad \
NANLOG_WATCH_TYPES=Linear,LinearProjection,AttentionBlock,InteractionLayer,MLP,EmbeddingGate \
NANLOG_ADDR=1 \
NANLOG_LOCATE=1 \
NANLOG_BAD_VALUES=1 \
NANLOG_ALLOC_SNAPSHOT=0 \
NANLOG_DUMP_TENSOR=0 \
NANLOG_SAMPLE_EVERY=1 \
  python3 instrument_nan_logger.py /path/to/your_repro.py
```

Notes:
- The `input` channel is **required** — the NaN is in the forward *input*, not the
  output. With only the default `act,igrad` you would miss it.
- **Expanded watch scope** (~156 modules) is the June29 feature here: the donor
  could be a `LinearProjection` (e.g. `attn.dot_proj`), `AttentionBlock`,
  `InteractionLayer`, `MLP`, or `EmbeddingGate` matmul that the bare `Linear`
  (~99 modules) filter misses.
- `NANLOG_BAD_VALUES` (June29) gives `first_bad_flat_idx/row/col/value`;
  `NANLOG_LOCATE` (June26) gives `bad_rows`; `NANLOG_ADDR` (June26, default on)
  gives the storage ranges. All are cheap on-GPU reductions / host-side reads (no
  host sync).
- `SAMPLE_EVERY=1` writes clean records too, so a producer's storage range at
  step S-1 is present to match against the victim. *(Lower-disk alternative for a
  long run: `NANLOG_SAMPLE_EVERY=50 NANLOG_PRE_CONTEXT=20` — the run-up buffer
  still captures the S-1 producers around the event.)*

**What we learn:**
- `summary_rank*.json → first_bad` (step/layer/dir/kind).
- On the victim `projections.10.layers.0|input` record: `bad_rows == 1` (single
  ~8 KiB tile = a stray write, *not* a spread numeric blowup) + flat
  `finite_abs_max` history + exact `first_bad_*`.
- **Donor:** match the victim's `[storage_ptr, storage_ptr + storage_nbytes)` range
  at step S against every other watched record at step S and S-1; an
  equal/overlapping range names the donor (strongest = equal `storage_ptr`).
  Byte offset into the 8 MiB block:
  `byte_offset = storage_offset_bytes + first_bad_flat_idx * 4` (fp32). We run this
  with `alias_fingerprint.py`.

**If the NaN does NOT reproduce here**, stop and tell us — this set is too light to
be the cause, so the repro config itself changed. Don't escalate blindly.

---

## Round 2 — add `NANLOG_ALLOC_SNAPSHOT` (definitive cross-stream proof)

**Goal:** capture the allocator event trace that shows `(free addr=X on stream A) →
(alloc addr=X on stream B)` at the victim's address — the direct proof of
cross-stream reuse. Everything from Round 1 is kept; this only adds the snapshot.
Run **after** Round 1 confirms the NaN, since this is the first round that perturbs
timing meaningfully (~10%).

```bash
NANLOG_DIR=/output/run2_alloc_snapshot \
NANLOG_CHANNELS=act,input,igrad \
NANLOG_WATCH_TYPES=Linear,LinearProjection,AttentionBlock,InteractionLayer,MLP,EmbeddingGate \
NANLOG_ADDR=1 \
NANLOG_LOCATE=1 \
NANLOG_BAD_VALUES=1 \
NANLOG_ALLOC_SNAPSHOT=1 \
NANLOG_DUMP_TENSOR=0 \
NANLOG_SAMPLE_EVERY=1 \
  python3 instrument_nan_logger.py /path/to/your_repro.py
```

Notes / cautions:
- `NANLOG_ALLOC_SNAPSHOT=1` (June29) adds ~10% per-step overhead (Python stack
  capture per allocation; GPU kernel timing itself is unaffected). It records the
  full alloc/free history and dumps a one-shot pickle
  (`alloc_snapshot_step*_rank*.pickle`) on the first NaN, then stops recording.
- If the NaN disappears in this round but was present in Round 1, that itself is
  evidence the bug is timing/allocation-sensitive (consistent with the cross-stream
  race) — report it; do not treat it as "not reproducible."

**What we learn:** the `device_traces` in the pickle give per-event GPU address,
size, **stream ID**, timestamp, and Python call stack. Cross-referenced with the
Round-1 donor address, this upgrades "named suspect" to a proven cross-stream
free→reuse at the victim address.

---

## Round 3 — add `NANLOG_DUMP_TENSOR` (full corrupted tensor; bonus, run last)

**Goal:** save the entire corrupted tensor to disk for exhaustive post-hoc
inspection (all bad element positions/values/context, no schema limit). Everything
from Round 2 is kept; this only adds the tensor dump.

```bash
NANLOG_DIR=/output/run3_dump_tensor \
NANLOG_CHANNELS=act,input,igrad \
NANLOG_WATCH_TYPES=Linear,LinearProjection,AttentionBlock,InteractionLayer,MLP,EmbeddingGate \
NANLOG_ADDR=1 \
NANLOG_LOCATE=1 \
NANLOG_BAD_VALUES=1 \
NANLOG_ALLOC_SNAPSHOT=1 \
NANLOG_DUMP_TENSOR=1 \
NANLOG_SAMPLE_EVERY=1 \
  python3 instrument_nan_logger.py /path/to/your_repro.py
```

Notes / cautions:
- **This is the most likely round to suppress the bug.** `NANLOG_DUMP_TENSOR=1`
  holds an extra reference to each watched tensor until the next step's drain,
  which delays allocator free→reuse — the exact mechanism under investigation. It
  can produce a **false negative** (NaN vanishes because the logger prevented the
  reuse). That is why it is **last**: by now Rounds 1–2 already have the donor and
  the cross-stream proof, so this round is pure bonus — if it suppresses the NaN,
  nothing is lost.
- On the first bad step it writes `bad_tensor_step*_<layer>_<role>_rank*.pt`
  (~1 ms GPU→host copy + ~10 ms disk write, one-shot).

**What we learn:** the full `[1024, 2048]` corrupted input tensor on disk — every
corrupted element, not just the first — for confirming the single-tile stray-write
pattern beyond the `bad_rows`/`first_bad_*` summary.

---

## Optional add-on — persistence channels (why the NaN sticks)

To also record how a one-step input NaN gets *baked into the weight* (NaN wgrad →
optimizer writes NaNs into the weight → all-NaN forever), add the param/grad
channels to any round above:

```bash
NANLOG_CHANNELS=act,input,igrad,weight,wgrad \
NANLOG_MAX_PARAM_NUMEL=100000000 \
```

- `weight` is read at step start (previous step's value, `value_is_from_prev_step:
  true`); `wgrad` is read at the optimizer `step_pre_hook` (current step). Both are
  sync-free.
- Requires PyTorch ≥ 2.1 and a standard optimizer (grads must survive to
  `optimizer.step()`). If the optimizer update is fused into backward
  (`apply_optimizer_in_backward`) or grads are freed early, `wgrad`/`bgrad` log a
  one-time warning and produce nothing — use `weight`/`bias` instead. Confirm via
  the summary: `optimizer_hook_registered: true` and `grad_records_stashed > 0`.

---

## Control run — does `record_stream` remove the NaN?

Run the **same config as Round 1**, but with your normal `record_stream` behavior
enabled (i.e. do **not** disable it). Expected: `first_bad: null`. If the NaN
vanishes, that strongly supports the cross-stream allocator-aliasing hypothesis; if
it persists, a different mechanism is also in play.

```bash
NANLOG_DIR=/output/run4_control_recordstream \
NANLOG_CHANNELS=act,input,igrad \
NANLOG_WATCH_TYPES=Linear,LinearProjection,AttentionBlock,InteractionLayer,MLP,EmbeddingGate \
NANLOG_ADDR=1 NANLOG_LOCATE=1 NANLOG_BAD_VALUES=1 \
NANLOG_SAMPLE_EVERY=1 NANLOG_PRE_CONTEXT=20 \
  python3 instrument_nan_logger.py /path/to/your_repro.py
```

---

## New-flag quick reference

| Flag | Default | New in | Round | What it adds | Perturbation |
| --- | --- | --- | --- | --- | --- |
| `NANLOG_ADDR` | `1` (on) | June26 | 1+ | `data_ptr`, `storage_ptr`, `storage_offset_bytes`, `storage_nbytes` per record → bridge to the donor buffer | none (host-side) |
| `NANLOG_LOCATE` | `0` | June26 | 1+ | `bad_rows` (dim-0 rows with any bad elem); `==1` ⇒ tile-sized late write | negligible (on-GPU, same drain) |
| `NANLOG_CHANNELS=…input…` | `act,igrad` | — | 1+ | capture the forward **input** (required for this NaN) + optional `weight,bias,wgrad,bgrad` | low |
| `NANLOG_SAMPLE_EVERY=1` | `50` | — | 1+ | write clean records too (needed to see producers at step S-1) | low (disk) |
| `NANLOG_BAD_VALUES` | `0` | **June29** | 1+ | `first_bad_flat_idx/row/col/value` of the first bad element | negligible (on-GPU, same drain) |
| `NANLOG_WATCH_TYPES` (wide) | `Linear` | **June29** | 1+ | also watch `LinearProjection`/`AttentionBlock`/`InteractionLayer`/`MLP`/`EmbeddingGate` so the donor isn't missed | low |
| `NANLOG_ALLOC_SNAPSHOT` | `0` | **June29** | 2+ | allocator alloc/free trace w/ stream IDs; pickle dumped on first NaN | ~10% step slowdown |
| `NANLOG_DUMP_TENSOR` | `0` | **June29** | 3 | save the full bad tensor to `.pt` on first detection | **may suppress the aliasing bug** — run last |

---

## What to send back (per round)

For each round, send the **entire** `NANLOG_DIR`:
- `layers_rank*.jsonl` — per-(layer, step) records
- `summary_rank*.json` — `first_bad` fingerprint + totals + the flags that were on
- `alloc_snapshot_step*_rank*.pickle` — Rounds 2–3 only, if a NaN was detected
- `bad_tensor_step*_*.pt` — Round 3 only, if a NaN was detected
- the console/stderr log of the run (has the `attached … N module(s)` and
  `FIRST BAD:` lines)

The single most valuable artifact is a **Round 1** `NANLOG_DIR` from a run that
actually reproduced the NaN (`first_bad` populated) — that alone names the donor
via addresses. Round 2's pickle then upgrades it from "named suspect" to
"cross-stream proof"; Round 3's `.pt` is exhaustive confirmation.
