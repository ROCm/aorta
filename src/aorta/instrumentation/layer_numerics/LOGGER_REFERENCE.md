# NaN Logger v2 — per-layer origin tracker + aliasing-writer finder

## ⮕ What to run and send back

Run your existing reproducer through this logger. Replace `/path/to/repro.py`
with your reproducer script. This command watches **all `nn.Linear` layers** (99
modules — projections, interaction towers, dense arch, task heads) on the
**3 key channels** with **address capture + bad-value locate + allocator
snapshot**, so a run that reproduces the NaN captures everything needed to
(1) name the first-bad layer, (2) pinpoint the corruption position, (3) identify
the aliasing donor, and (4) prove or rule out cross-stream allocator reuse.

```bash
NANLOG_DIR=/output/nanlog \
NANLOG_CHANNELS=act,input,igrad \
NANLOG_WATCH_TYPES=Linear,LinearProjection,AttentionBlock,InteractionLayer,MLP,EmbeddingGate \
NANLOG_ADDR=1 NANLOG_LOCATE=1 NANLOG_BAD_VALUES=1 \
NANLOG_ALLOC_SNAPSHOT=1 \
NANLOG_PRE_CONTEXT=20 \
  python3 instrument_nan_logger.py /path/to/repro.py
```

**Send back:** the output directory `/output/nanlog` — it contains
`layers_rank*.jsonl`, `summary_rank*.json`, and (if NaN was detected)
`alloc_snapshot_step*_rank*.pickle` — plus the console/stderr log from
the run.

**Quick self-check before sending:** in the startup log you should see
`attached layer hooks to N module(s)` with **N > 0** and
`capture_addr=True; locate=True; bad_values=True; alloc_snapshot=True`.
With the recommended `NANLOG_WATCH_TYPES` this reproducer reports **~156
module(s)**. If N is 0, the hooks did not bind — see "What it hooks" below.

> **Why the wide type list?** `attn.dot_proj` is a `LinearProjection` (uses
> `self.w @ x + self.b`, not `nn.Linear`), so the bare `Linear` default misses
> it. The recommended type list covers all module classes that run matmuls:
> `nn.Linear` (99 modules), `LinearProjection` (projections like `attn.dot_proj`),
> `AttentionBlock` (in-forward matmul), `InteractionLayer`, `MLP`, `EmbeddingGate`
> — **~156 modules** total. The NaN first appears in the **input** of
> `projections.10`, but the corruption **donor** (the producer whose buffer was
> reused) could be any of these — particularly `attn.dot_proj`, which is the
> leading suspect. Full coverage lets you cross-reference `storage_ptr` across
> all layers to find the actual writer.
>
> **Why `act,input,igrad`?** The NaN appears in the forward **input**, not the
> output. The `input` channel captures the input tensor's address so
> `NANLOG_ADDR` can trace which memory block it occupies. `act` (output) and
> `igrad` (backward grad) complete the picture. To also monitor weights and
> gradients, add `weight,bias,wgrad,bgrad` to `NANLOG_CHANNELS`.
>
> **To watch only projections** (smaller JSONL, lower perturbation):
> `NANLOG_WATCH_NAMES=emb_proj.projections` — this auto-disables the `Linear`
> type default and watches only the 62 projection layers. But note this will
> miss `attn.dot_proj` (a `LinearProjection`) as a donor candidate.

---

A single-file PyTorch sidecar that watches the **model layers** (forward
activations + backward gradients) so a NaN/Inf can be walked **upstream** to the
layer/step where it is born. Each step it records, per tracked layer:

```text
nan_count / inf_count / huge_count   # |x| > threshold counts
finite_abs_max / finite_max / min    # magnitude EVERY step, even when nan_count==0
tf32_path                            # is this layer on the fp32 + allow_tf32 (TF32) path
matmul_calls_so_far                  # non-syncing counter; lines up with a GEMM tool's "call #N"
layer_name / direction / rank/gpu/host/pid
```

The most valuable signal is the **finite → huge → NaN magnitude trajectory**. A
NaN-only check fires only once the tensor is already 100% NaN; this logger
records `finite_abs_max` every step, so a value blowing up is visible *before* it
becomes NaN.

No monitoring happens inside any matrix multiplication. Per-layer hooks compute
their reductions on-device and stash GPU scalars **without** `.item()`; once per
step every stashed scalar is stacked and copied to the host in a **single**
`.cpu().tolist()` transfer (the one implicit device sync per step — all watched
tensors for the step flushed together, automatic and with no fixed count; ~300
per step in the validated run: 156 layers × fwd/bwd). GPU kernel launch order is
therefore unchanged and timing-sensitive behavior is preserved.

---

## What changed since the previous version

If you have run an earlier version of this logger:

- **Address capture (`NANLOG_ADDR`, default ON).** Every record carries the
  watched tensor's GPU address (`data_ptr`) and backing-storage extent
  (`storage_ptr`, `storage_offset_bytes`, `storage_nbytes`). Pure host-side
  metadata — **no GPU sync, kernel order unchanged**. This is the key to tracing
  memory-**aliasing** corruption to the producer buffer (see "Finding the aliasing
  writer from addresses"). **Important:** `NANLOG_ADDR` records the address for
  every enabled channel. Since the NaN appears in the forward **input**, the
  `input` channel must be enabled (`NANLOG_CHANNELS=act,input,igrad`) to capture
  the input tensor's address. With the default `act,igrad` only, the input address
  is not recorded.
- **Row-locate (`NANLOG_LOCATE`, default OFF).** Adds `bad_rows` (how many dim-0
  rows hold a NaN/Inf/huge element). `bad_rows==1` on an 8 MiB `[1024,2048]`
  input = one tile-sized late write (aliasing); large `bad_rows` = numeric blowup.
- **Bad-value locate (`NANLOG_BAD_VALUES`, default OFF).** *(new)* For each bad
  tensor, records the first corrupted element's exact position and value:
  `first_bad_flat_idx`, `first_bad_row`, `first_bad_col`, `first_bad_value`.
  Combined with `NANLOG_ADDR`, the precise byte offset into the 8 MiB storage
  block can be computed: `storage_offset_bytes + first_bad_flat_idx × 4` (fp32).
  Implemented as GPU-side `argmax` + scalar indexing — no extra host sync.
- **Allocator snapshot (`NANLOG_ALLOC_SNAPSHOT`, default OFF).** *(new)* Enables
  PyTorch's caching allocator event recorder at model init. On the first NaN
  detection, dumps a full snapshot (pickle) containing every alloc/free event with
  **GPU address, size, stream ID, timestamp, and Python call stack**. This is the
  definitive proof for cross-stream allocator reuse: the trace shows `(free
  addr=X on stream A)` then `(alloc addr=X on stream B)`, pinpointing the reuse.
  ~10% per-step overhead from stack capture; GPU kernel timing unaffected. Dump
  is one-shot; recording stops automatically after.
- **Expanded watch scope.** The default `NANLOG_WATCH_TYPES=Linear` now covers
  99 modules across the model (not just projections): `emb_proj.projections`
  (62), `interaction.towers` (26), `task_heads` (4), `dense_arch` (3),
  `combiner.gate` (3), `interaction.fuse` (1).

The output format is otherwise unchanged and the new fields are additive (old
parsers ignore unknown keys). All new features default to OFF; the recommended
command at the top enables them.

---

## Quick start

1. Drop these files anywhere reachable on the training host
   (e.g. `/opt/nan_logger_v2`):

    ```bash
    unzip nan_logger_v2.zip -d /opt/nan_logger_v2
    ```

2. Choose an output directory — set `NANLOG_DIR` to any writable path (the logger
   creates it if it doesn't exist; defaults to `./nan_logger_v2_out`). The command
   below sets it inline, so there is nothing to export separately.

3. Run your reproducer **through** the logger — it arms the hooks, then runs your
   script via `runpy` so the hooks attach to the real eager module tree. This is
   the same command shown at the top of this file:

    ```bash
    NANLOG_DIR=/output/nanlog \
    NANLOG_CHANNELS=act,input,igrad \
    NANLOG_WATCH_TYPES=Linear,LinearProjection,AttentionBlock,InteractionLayer,MLP,EmbeddingGate \
    NANLOG_ADDR=1 NANLOG_LOCATE=1 NANLOG_BAD_VALUES=1 \
    NANLOG_ALLOC_SNAPSHOT=1 \
    NANLOG_PRE_CONTEXT=20 \
      python3 instrument_nan_logger.py /path/to/repro.py [args...]
    ```

   This wrapper invocation is the recommended, tested path — it arms the hooks and
   then runs your script for you. The wide `NANLOG_WATCH_TYPES` covers all ~156
   matmul-bearing modules including `attn.dot_proj` (`LinearProjection`). To
   narrow scope, use `NANLOG_WATCH_NAMES=emb_proj.projections` (only the 62
   projection layers — but misses `LinearProjection` donor candidates).
   To also monitor weights and gradients, add `weight,bias,wgrad,bgrad` to
   `NANLOG_CHANNELS` and set `NANLOG_MAX_PARAM_NUMEL=100000000`.

   *Advanced / alternative:* if you'd rather launch your script the usual way
   (e.g. through an existing `torchrun` line you can't change), add **three lines
   at the very top** of the script — **before the model is built** — and set
   `NANLOG_*` env vars in your shell as usual:

    ```python
    import sys
    sys.path.insert(0, "/opt/nan_logger_v2")   # path to instrument_nan_logger.py
    import instrument_nan_logger  # noqa: F401  (importing this arms the hooks)
    ```

That's it. On startup you'll see one line confirming the hooks are armed, and
one line when they attach to the built model (see "Expected log lines"). Output
appears under `NANLOG_DIR`.

**Sanity check the startup lines:** you must see
`attached layer hooks to N module(s)` with **N > 0**, and on the same line
`capture_addr=True; locate=True; bad_values=True; alloc_snapshot=True`. With the
recommended `NANLOG_WATCH_TYPES`, N should be **~156** on this model. If N is 0
(or that line never appears), the model isn't going through
`DistributedModelParallel` and the hooks didn't bind — see "What it hooks" below.

**A clean run** (no NaN reproduced) ends with `first_bad` = `null` in the
summary. That's the logger working correctly and finding nothing — not a
failure. `first_bad` is populated only when a layer actually goes NaN/Inf/huge.

To disable the instrumentation, drop the wrapper invocation (or comment out the
`import instrument_nan_logger` line) — nothing else needs to be undone.

---

## What it hooks (and changing it)

By default the logger hooks every `torch.nn.Linear` in the model. It attaches by
auto-wrapping
`torchrec.distributed.model_parallel.DistributedModelParallel.__init__`, so the
hooks bind to the **real sharded module tree** the moment it is built — no edit
to the (frozen) reproducer. The model is eager (only `@torch._dynamo.disable` on
the embedding lookup), so `nn.Module` hooks bind cleanly.

To watch other module types, set a comma list of class names:

```bash
export NANLOG_WATCH_TYPES=Linear,LinearProjection,EmbeddingGate
```

**Recommended for this reproducer — cast a wider net.** The default `Linear`
filter covers the dense `nn.Linear` layers (where the upstream TF32 GEMM that
originates the NaN lives), but it misses matmuls written as raw `@` / functional
ops inside other modules — e.g. `LinearProjection` (`self.w @ x + self.b`) and
`AttentionBlock` (an in-`forward` matmul). A hook sees a module's **output**
tensor, so adding these classes catches a NaN *emerging from* that block even
though the internal matmul itself isn't a `Linear`. Cost is negligible (a few
extra on-device reductions per step, still one batched sync), so grab them all:

```bash
export NANLOG_WATCH_TYPES=Linear,LinearProjection,AttentionBlock,InteractionLayer,MLP,EmbeddingGate
```

### Targeting a specific layer or block (`NANLOG_WATCH_NAMES`)

To investigate one layer instead of the whole model, match on the module **path**
with `NANLOG_WATCH_NAMES` (comma-separated substrings):

```bash
# only this exact layer — no NANLOG_WATCH_TYPES needed:
export NANLOG_WATCH_NAMES=emb_proj.projections.10.layers.0

# the whole projection-10 block (every sub-layer under it):
export NANLOG_WATCH_NAMES=emb_proj.projections.10
```

A module is watched if its class name matches `NANLOG_WATCH_TYPES` **or** its path
contains any `NANLOG_WATCH_NAMES` substring (union). Setting `NANLOG_WATCH_NAMES`
alone automatically turns off the `Linear` type default, so you get exactly the
named layers — you do **not** need to clear `NANLOG_WATCH_TYPES`. To watch named
layers **and** a class at once, set both (e.g. `NANLOG_WATCH_NAMES=...` plus
`NANLOG_WATCH_TYPES=MoELayer`). The startup log prints the matched module names
when 10 or fewer are watched, so you can confirm you targeted the right one.

Two scope notes for this model:

- **Embedding tables are intentionally not in the list.** They are a
  `torchrec.EmbeddingBagCollection`; after sharding the lookup goes through a
  distributed gather/all-to-all (not a clean forward tensor), backward grads are
  sparse (drop the `igrad` channel from `NANLOG_CHANNELS` if they misbehave), and
  the analysis already places them **downstream** — they receive already-NaN
  grads, they don't originate the NaN. Nothing needs to be pulled from another
  library; the layers that matter are native and covered above. The `weight`/
  `bias` param channels also skip embedding modules by type (logged in the
  summary's `skipped_params`).
- **Compiled regions lose per-layer granularity.** This reproducer runs eager
  (only `@torch._dynamo.disable` on the embedding lookup), so hooks bind cleanly.
  If you `torch.compile` a block, hooks on modules *inside* the compiled region
  generally do not fire — you'd only see the outermost compiled module's output.

*What* each watched layer records (activations, gradients, parameter values) is
controlled separately by `NANLOG_CHANNELS` — see "Capture channels" below. The
command at the top of this file already enables all of them.

---

## Expected log lines

On startup (you should see these, once each):

```text
nanlog: auto-hook armed (will attach when DistributedModelParallel is built)
nanlog: allocator snapshot: recording enabled (stacks=python, max_entries=500000). ...
nanlog: attached layer hooks to 156 module(s) (types=['Linear', 'LinearProjection', 'AttentionBlock', 'InteractionLayer', 'MLP', 'EmbeddingGate'], names=[]); channels=['act', 'igrad', 'input']; capture_addr=True; locate=True; bad_values=True; alloc_snapshot=True; params_scanned=0; params_skipped=0; per-step drain installed on DMP root
```

(The allocator snapshot line appears only when `NANLOG_ALLOC_SNAPSHOT=1`.)

When you enable a gradient channel (`wgrad`/`bgrad`), you will also see, once at
startup and once per optimizer:

```text
nanlog: grad channels ['bgrad', 'wgrad'] enabled; will read gradients at the optimizer step boundary
nanlog: grad channels ['bgrad', 'wgrad']: hooked optimizer <OptimizerClass> to read gradients
```

You will also see two benign PyTorch warnings once at startup; both are expected
and do not affect logging:

```text
UserWarning: Full backward hook is firing when gradients are computed with respect to module outputs since no inputs require gradients.
UserWarning: For backward hooks to be called, module output should be a Tensor ... but received <class 'torch.fx.proxy.Proxy'>
```

The second comes from the FX-traced sparse/embedding path (those modules are out
of scope by design); dense-path backward hooks still fire normally.

On the first layer that goes bad (once per run):

```text
19:58:12 [PID=1033357 rank=0] nanlog: FIRST BAD: step=84 layer=_dmp_wrapped_module.module.train_module.model.dense_arch.gating.layers.0 dir=bwd kind=nan call#~8050 tf32=True
```

`FIRST BAD` is the cross-run fingerprint — `step` / `layer` / `dir` (fwd|bwd) /
`kind` (nan|inf|huge) / approximate `call#` / TF32 flag. Diff it across runs:

- **stable** step+layer across runs → looks deterministic → argues *numerics*.
- **jittery** (different layer/step each run) on the TF32 path → consistent with
  timing nondeterminism. **Suggestive only.**

At exit, one line points at the summary:

```text
13:36:10 [PID=1 rank=0] nanlog: summary -> /output/nanlog/summary_rank0.json (first_bad={...})
```

---

## Configuration (env vars)

| variable                | default                 | meaning                                                          |
| ----------------------- | ----------------------- | ---------------------------------------------------------------- |
| `NANLOG_DIR`            | `./nan_logger_v2_out`   | where JSONL + summary are written                                |
| `NANLOG_HUGE_THRESHOLD` | `1e10`                  | finite-but-huge cutoff treated as corruption (`huge_count`)      |
| `NANLOG_SAMPLE_EVERY`   | `50`                    | write a CLEAN layer record 1-in-N steps (bad layers ALWAYS written) |
| `NANLOG_MAX_RECORDS`    | `200000`                | hard cap on JSONL records                                        |
| `NANLOG_WATCH_TYPES`    | `Linear`                | module **class** names to hook (comma-separated); unioned with `NANLOG_WATCH_NAMES` |
| `NANLOG_WATCH_NAMES`    | _(empty)_               | substrings matched against module **paths**; watch a specific layer/block (see below) |
| `NANLOG_CHANNELS`       | `act,igrad`             | capture channels (see table below); replaces `NANLOG_BWD`        |
| `NANLOG_MAX_PARAM_NUMEL`| `50000000`              | param channels skip params bigger than this (embeddings skipped by type) |
| `NANLOG_PRE_CONTEXT`    | `0`                     | keep last K steps in memory, dump on first bad (0 = off)         |
| `NANLOG_ADDR`           | `1`                     | record `data_ptr` + storage extent per tensor (sync-free); the key to naming an aliasing writer |
| `NANLOG_LOCATE`         | `0`                     | also record `bad_rows` (dim-0 rows holding any NaN/Inf/huge); `bad_rows==1` ⇒ tile-sized late write |
| `NANLOG_BAD_VALUES`     | `0`                     | record `first_bad_flat_idx/row/col/value` for each bad tensor (sync-free GPU scalar reductions) |
| `NANLOG_ALLOC_SNAPSHOT` | `0`                     | record allocator alloc/free events; dump snapshot (pickle) on first NaN (~10% step overhead) |
| `NANLOG_STOP_ON_FIRST`  | `0`                     | stop writing clean records after the first bad layer is seen     |
| `NANLOG_VERBOSE`        | `0`                     | log one line every step (worst bad layer / abs_max)              |

### Capture channels (`NANLOG_CHANNELS`)

Comma-separated. Each channel is an independent observation adding its own on-GPU
reductions; all feed the **same single per-step drain**, so no channel adds a host
sync. New (non-default) channels are off unless listed, so the default keeps the
original activation-only timing profile.

| channel  | reads                          | fires in           | default |
| -------- | ------------------------------ | ------------------ | ------- |
| `act`    | forward output activation      | fwd hook           | **on**  |
| `input`  | forward input activation       | fwd hook           | off     |
| `igrad`  | grad w.r.t. inputs (grad_input)| bwd hook           | **on**  |
| `weight` | param values, `ndim >= 2`      | root pre-hook      | off     |
| `bias`   | param values, `ndim == 1`      | root pre-hook      | off     |
| `wgrad`  | `param.grad`, `ndim >= 2`      | optimizer step-hook| off     |
| `bgrad`  | `param.grad`, `ndim == 1`      | optimizer step-hook| off     |

The `weight`/`bias` channels are read at the **start** of the step (root
pre-hook), so their values are the **previous** step's (the only sync-free read
point for persistent params) — each such record carries
`value_is_from_prev_step: true`.

The `wgrad`/`bgrad` channels read `param.grad` at the **optimizer
`step_pre_hook`** (after backward, before the update), so they are the **current**
step's grad (`value_is_from_prev_step: false`) and still sync-free / outside the
autograd graph. They require a `torch.optim.Optimizer` (auto-discovered — any
subclass, library or custom; no script edit). If grads are freed during backward
(optimizer-in-backward / fused) or there is no optimizer at all, a **one-time
WARNING** is logged and these channels produce nothing — use `weight`/`bias`
instead.

The weight/bias split is by **shape** (`ndim`), so custom param names (`.w`,
`.scale`, `.b`) are handled correctly; the exact name is in `param_name`.
Embedding tables are skipped (by type) and catalogued in the summary's
`skipped_params` — embedding scanning is a separate deferred follow-up.

`RANK` / `LOCAL_RANK` (set by `torchrun`) are read to name the per-rank output
files; no need to set them yourself.

### Gradient channels (`wgrad` / `bgrad`)

These read `param.grad` and have two requirements:

1. **PyTorch >= 2.1** (they use `Optimizer.register_step_pre_hook`).
2. **A standard optimizer** whose gradients persist until `optimizer.step()`.
   They do **not** work when the optimizer update is fused into the backward pass
   (e.g. `apply_optimizer_in_backward`), because each gradient is freed during
   backward before it can be read.

If either requirement is unmet, the logger prints a one-time warning and these
channels produce no records — the rest of the run is unaffected. To confirm they
worked, check the summary: `optimizer_hook_registered` should be `true` and
`grad_records_stashed` should be greater than 0. If grads are not readable on your
model, use the `weight`/`bias` channels instead.

---

## Output layout

```text
<NANLOG_DIR>/
  layers_rank<N>.jsonl                          # one record per (layer, step) written
  summary_rank<N>.json                          # first_bad fingerprint + totals (at exit)
  alloc_snapshot_step<S>_rank<N>.pickle         # allocator snapshot (only if NaN detected)
```

- `layers_rank<N>.jsonl` — every **bad** layer is written; **clean** layers are
  sampled 1-in-`NANLOG_SAMPLE_EVERY` (keeps the file small while still sampling
  the magnitude trajectory). Each record is self-contained — no line arithmetic
  needed to recover step/layer/shape/path.
- `summary_rank<N>.json` — `first_bad` (the determinism fingerprint),
  `steps_seen`, `records_written`, `matmul_calls_total`, `tf32_allowed`,
  `watched_count` / `watched_names`, `channels`, `capture_addr`, `locate`,
  `bad_values`, `alloc_snapshot`, `alloc_snapshot_dumped`,
  `params_scanned` / `skipped_params`, and (when grad channels are on)
  `optimizer_hook_registered` / `grad_params_planned` / `grad_records_stashed`.
- `alloc_snapshot_step<S>_rank<N>.pickle` — *(only when `NANLOG_ALLOC_SNAPSHOT=1`
  and NaN is detected)* full caching allocator snapshot containing `segments`
  (current memory layout) and `device_traces` (timestamped alloc/free events with
  GPU address, size, **stream ID**, and Python call stack). Load with
  `pickle.load()` or PyTorch's built-in memory visualizer.

One JSONL record:

```jsonc
{
  "type": "layer_step",
  "step": 23, "rank": 5, "gpu": 5, "host": "node-03", "pid": 558400,
  "layer_name": "model.blocks.7.mlp.fc2",
  "direction": "fwd",                // fwd | bwd | param | grad
  "role": "input",                   // act|input|igrad|weight|bias|wgrad|bgrad
  "param_name": "",                  // owned-param name for param/grad roles; "" otherwise
  "value_is_from_prev_step": false,  // true for weight/bias (read at step start); false for grads
  "shape": [1024, 2048], "dtype": "torch.float32",
  "tf32_path": true,                 // fp32 operands + allow_tf32 -> TF32 path
  "matmul_calls_so_far": 8050,       // lines up with a GEMM tool's "call #N"
  "data_ptr": "0x7efeedb71200",      // tensor address (NANLOG_ADDR=1, default)
  "storage_ptr": "0x7efeedb71200",   // start of the backing block
  "storage_offset_bytes": 0,         // tensor offset into that block
  "storage_nbytes": 8388608,         // size of the backing block (8 MiB here)
  "bad_rows": 1,                     // rows with any bad elem (NANLOG_LOCATE=1)
  "nan_count": 1, "inf_count": 0, "huge_count": 4,
  "finite_count": 2097147,
  "finite_abs_max": 3.13e36,         // logged EVEN when nan_count==0  <-- the value-add
  "finite_max": 3.13e36, "finite_min": -2.87e36,
  "numel": 2097152, "bad": true,
  "first_bad_flat_idx": 344164,      // flat index of first bad element (NANLOG_BAD_VALUES=1)
  "first_bad_row": 42,               // dim-0 index
  "first_bad_col": 100,              // offset within that dim-0 slice
  "first_bad_value": "NaN"           // "NaN" / "Inf" / "-Inf" / numeric (JSON-safe)
}
```

`data_ptr` / `storage_*` appear when `NANLOG_ADDR=1` (the default); `bad_rows`
appears when `NANLOG_LOCATE=1`; `first_bad_*` fields appear when
`NANLOG_BAD_VALUES=1` **and** the record is bad.

The default run emits only `role: "act"` and `role: "igrad"` records. The
`weight` / `bias` / `wgrad` / `bgrad` roles appear only when you enable those
channels via `NANLOG_CHANNELS`; those records carry the owning `param_name`.
`weight`/`bias` carry `value_is_from_prev_step: true` (read at step start);
`wgrad`/`bgrad` carry `value_is_from_prev_step: false` (read at the optimizer
step boundary, current step).

When a tensor is **entirely** non-finite (`finite_count == 0`, e.g. an all-NaN
gradient — the typical first-bad case), `finite_abs_max` / `finite_max` /
`finite_min` are written as JSON `null` (there is no finite value to report).
The output is always valid, strict-parseable JSON.

### How to read the output — decision table

A layer record is **bad** when any of `nan_count` / `inf_count` / `huge_count`
> 0; otherwise **clean**. Read the trajectory of the first-bad layer:

| `finite_abs_max` trend before first NaN | `huge_count` | Conclusion |
| --------------------------------------- | ------------ | ---------- |
| flat / normal, then NaN appears abruptly | 0 | **NaN arrived from elsewhere** — this layer is downstream of the origin; look at earlier `matmul_calls_so_far` / other layers same step. |
| climbing over steps, crosses threshold, then NaN | >0 before NaN | **Finite blowup originating here** — precision/overflow growth on this layer (check `tf32_path`); the NaN-only checks would have missed this. |
| bad on the **TF32 path** layers only, jittery across runs | (any) | TF32 path implicated; combined with a jittery fingerprint → consistent with timing nondeterminism (suggestive, never proof). |

The columns map directly to JSONL fields: `finite_abs_max`, `huge_count`,
`nan_count`, `tf32_path`, `matmul_calls_so_far`.

---

## Finding the aliasing writer from addresses

The decision table above identifies the **victim** (the first layer/step that
goes bad). The address fields go one step further and identify the **producer**
whose buffer was reused/overwritten — the donor/writer of a cross-stream aliasing
late-write. The idea: a freed block is recycled as the victim's input while a
producer kernel is still writing it, so the victim's storage range will **equal
or overlap** a producer's output (or param) storage range.

Two fast signatures, computed entirely from the JSONL (no GPU, offline):

1. **Aliasing signature on the victim itself.** With `NANLOG_LOCATE=1`, the
   first-bad `emb_proj.projections.10.layers.0` `input` record should show
   `bad_rows == 1` (a single ~8 KiB row of an 8 MiB `[1024,2048]` block — one GEMM
   tile's worth of a stray write), **not** a large/spread `bad_rows`. A spread
   pattern argues a genuine numeric blowup instead; `bad_rows==1` + flat
   `finite_abs_max` history (no run-up) argues a foreign write.

2. **Address match to a producer.** Take the victim record's
   `[storage_ptr, storage_ptr + storage_nbytes)` range at the bad step `S`. Scan
   the **same step and step S-1** for any **other** watched record (`act`, or a
   `weight`/`wgrad` of a producer like `attn.dot_proj`) whose storage range equals
   or overlaps it. A hit names the donor/writer: that producer's output buffer was
   handed to proj.10's input. Equal `storage_ptr` is the strongest hit; an overlap
   (different offset, same block) is also conclusive.

```python
import json, collections
recs = [json.loads(l) for l in open("nanlog/layers_rank0.jsonl")]
# 1) the first-bad victim input
bad = next(r for r in recs if r["bad"] and "projections.10" in r["layer_name"]
           and r["role"] == "input")
lo = int(bad["storage_ptr"], 16); hi = lo + bad["storage_nbytes"]; S = bad["step"]
print("victim", bad["layer_name"], "step", S, "bad_rows", bad.get("bad_rows"),
      "block", hex(lo), bad["storage_nbytes"])
# 2) producers sharing that block at step S or S-1
for r in recs:
    if r is bad or r["step"] not in (S, S - 1) or "storage_ptr" not in r:
        continue
    a = int(r["storage_ptr"], 16); b = a + r.get("storage_nbytes", 0)
    if a < hi and lo < b:               # ranges overlap
        print("ALIAS WRITER:", r["step"], r["layer_name"], r["role"],
              r["shape"], hex(a), r.get("storage_nbytes"))
```

Notes and limits:

- The caching allocator reuses freed blocks aggressively, so an **equal address in
  the SAME step** between an unrelated freed tensor and proj.10's input is normal
  and expected — that alone is not proof. The strong evidence is the **conjunction**:
  proj.10 input first-bad **+** `bad_rows==1` **+** flat magnitude history **+** the
  matching block belonging to a plausible upstream **producer that ran just before**
  the prefetch (e.g. `attn.dot_proj`). Use the candidate as the suspect, not the
  verdict.
- This logger observes **tensor-level** addresses, not kernel completion order. To
  pin the exact writing kernel, cross-reference the suspect producer + step with a
  `HIPBLASLT_LOG` / rocprof trace (kernel name, stream, end time vs. the prefetch
  copy's start). The address match tells you *which buffer*; the trace tells you
  *which kernel and when*.
- Addresses are virtual and per-process; only compare within one rank's JSONL.

---

## Files in this drop

| file                       | purpose                                                          |
| -------------------------- | --------------------------------------------------------------- |
| `instrument_nan_logger.py` | the logger itself (importing/running it arms the hooks)          |
| `README.md`                | run instructions (what to run, what to send back)                |
| `LOGGER_REFERENCE.md`      | this file — full technical reference                             |

No pip install, no extra dependencies beyond PyTorch (already required by the
training run). All new features (`NANLOG_BAD_VALUES`, `NANLOG_ALLOC_SNAPSHOT`)
default to OFF and do not affect existing repro runs unless explicitly enabled.

---

## Appendix: Address Comparability

The `storage_ptr` values recorded by the logger are **absolute GPU virtual
addresses** within the process, obtained via `tensor.untyped_storage().data_ptr()`.

- **Within one rank's JSONL:** All records share the same virtual address space.
  Comparing `[storage_ptr, storage_ptr + storage_nbytes)` between any two records
  (regardless of layer, step, channel, or role) is a direct apples-to-apples
  comparison. Overlapping ranges prove that two tensors share the same underlying
  physical GPU memory block.

- **Across ranks / processes:** Each process has its own virtual address space
  (confirmed with multi-process experiments: zero shared base pointers). Do NOT
  compare `storage_ptr` values from different ranks.

- **Across steps (same rank):** Addresses are stable within a process lifetime.
  The same `storage_ptr` at step S and step S+1 means the allocator handed out
  the same block twice (expected — caching allocators reuse). Combined with
  `bad==true` at step S on the victim and a donor at step S or S-1, this is the
  aliasing evidence.

- **No GPU sync:** `data_ptr()` and `untyped_storage().data_ptr()` are pure
  host-side metadata queries. They do NOT trigger device synchronization or
  change kernel launch order — safe for timing-sensitive reproduction runs.
