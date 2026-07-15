# Layer Numerics (NaN / magnitude logger)

Trace a NaN/Inf back to the **layer and step where it first appears**, and
capture the finite-magnitude run-up leading to it — a value growing
large → huge → NaN over several steps, which a NaN-only check cannot see.

The logger arms hooks on your model the moment it is built and records, per
watched layer and step:

- `nan_count` / `inf_count` / `huge_count` (huge = `|x| > threshold`)
- `finite_abs_max` / `finite_max` / `finite_min`, logged **every** step even
  when there are no NaNs yet (this is what captures the run-up)
- layer identity, direction (fwd/bwd), and a running matmul-call counter

All per-layer reductions run on the GPU and are drained to the host in **one
batched transfer per step**, so no per-GEMM sync is introduced and
timing-sensitive behavior is preserved.

## When to use it

- A training or inference run produces a NaN/Inf and you need to know **which
  layer and step** it started at, not just that it happened.
- You suspect a value is blowing up over time and want the magnitude trajectory
  leading into the NaN.
- You want to check whether a tensor leaves an expected numeric range
  (out-of-range detection) at a specific pipeline stage.

It locates the first bad layer/stage/step. It does **not**, by itself, prove a
kernel-level timing race — that would need per-GEMM ordering, which would change
the timing the logger is designed to preserve.

## Prerequisites

- PyTorch built with ROCm/HIP.
- `torchrec` **only** if you use the automatic hook on a
  `DistributedModelParallel` root or pipeline-stage tracking
  (`NANLOG_PIPELINE=1`). Plain `nn.Module` models are supported through the
  standalone script; see below.
- No extra install — `instrument_nan_logger.py` has no dependencies beyond
  PyTorch, and all configuration is through `NANLOG_*` environment variables.

## Standalone usage (supported path)

The logger runs as a front-end around your training/repro script. It arms its
hooks, then executes your script via `runpy`, so the hooks attach to the real
model the moment it is built — **no edit to your script is required**.

From an installed `aorta` package:

```bash
NANLOG_DIR=/output/layer_numerics \
NANLOG_WATCH_NAMES=encoder.blocks \
NANLOG_PRE_CONTEXT=10 \
  python -m aorta.instrumentation.layer_numerics /path/to/your_script.py [args...]
```

From a bare checkout of just the script (e.g. handed to someone running the
repro on their own host, with no `aorta` install):

```bash
NANLOG_DIR=/output/layer_numerics \
  python instrument_nan_logger.py /path/to/your_script.py [args...]
```

Both forms are byte-equivalent — the module entry point just `runpy`-execs the
script.

### Startup sanity check

Before trusting a run, confirm the hooks attached. On startup the logger prints
a line to stderr like:

```
nanlog: attached layer hooks to N module(s) (types=[...], names=[...]); channels=[...]; ...
```

`N > 0` means hooks bound to your model. If `N == 0`, either the model was built
before the logger armed, or it is not reachable through the hook path (see
[Troubleshooting](#troubleshooting)).

## Options

All configuration is via `NANLOG_*` environment variables. The authoritative,
exhaustive reference is the module docstring at the top of
[`instrument_nan_logger.py`](../src/aorta/instrumentation/layer_numerics/instrument_nan_logger.py).
All heavy features default **OFF** and feed the same single per-step host
transfer (no added sync). Most-used knobs:

| Var | Purpose | Default |
|---|---|---|
| `NANLOG_DIR` | Output dir for the JSONL + summary | `nan_logger_out` |
| `NANLOG_CHANNELS` | Capture channels: `act,input,igrad,weight,bias,wgrad,bgrad` | `act,igrad` |
| `NANLOG_WATCH_NAMES` | Substrings matched against module paths (target a block/layer) | (empty) |
| `NANLOG_WATCH_TYPES` | Module class names to watch (e.g. `Linear`) | `Linear` (unless `WATCH_NAMES` set) |
| `NANLOG_PRE_CONTEXT` | Buffer the last K clean steps; dump on first-bad for the full run-up | `0` (off) |
| `NANLOG_SAMPLE_EVERY` | Write a clean record 1 step in N (bad steps always written) | `50` |
| `NANLOG_HUGE_THRESHOLD` | `\|x\| > T` counts as "huge" | `1e10` |
| `NANLOG_BAD_VALUES` | Record the first bad element's flat idx / row / col / value | `0` (off) |
| `NANLOG_ADDR` | Record `data_ptr` + backing-storage extent per tensor | `0` (off) |
| `NANLOG_DUMP_TENSOR` | Save the full bad tensor to a `.pt` file on first detection | `0` (off) |
| `NANLOG_ALLOC_SNAPSHOT` | Dump a caching-allocator event trace on first bad (~10% step overhead) | `0` (off) |
| **Pipeline / bounds** | | |
| `NANLOG_PIPELINE` | Track tensors at the TorchRec stage boundaries | `0` (off) |
| `NANLOG_TRACK_ATTR` | Batch attribute(s) to follow as tracked tensors | `embedding_features` |
| `NANLOG_BOUNDS` | Per-tensor in-range check `substr:lo:hi;...` (out-of-range → `kind="oob"`) | (empty) |
| `NANLOG_SPARSE` | Cheap host-side KJT metadata at the sparse stage | `0` (off) |
| `NANLOG_TRACK_EVERY_LAYER` | Re-scan tracked tensors at each layer (high overhead) | `0` (off) |

Channel notes:

- `weight`/`bias` are read at the **start** of the step, so they hold the
  **previous** step's value (`value_is_from_prev_step=true`).
- `wgrad`/`bgrad` are read at the optimizer step boundary (current step) and
  require a `torch.optim.Optimizer`. If the update is fused into backward, a
  one-time warning is logged — use `weight`/`bias` instead.

## Two tracking modes

**Layer channels** (default). Hooks the watched `nn.Module`s and records their
activations / grads / params, configured by `NANLOG_CHANNELS` +
`NANLOG_WATCH_*`.

**Pipeline-stage tracking** (`NANLOG_PIPELINE=1`, TorchRec). Follows named
tensors *by object* off the TorchRec batch (default `embedding_features`) and
scans them at each pipeline stage — `copy`, `sparse_start`, `sparse_wait`, and
`forward` — **before** the layers read them. Each record is tagged with its
`phase` and a `batch_id` (assigned at `copy_batch_to_gpu`, re-read at every later
stage) so one batch's records line up across steps. This catches corruption that
arrives *upstream* of the layers, which the layer hooks alone cannot see. Needs
no `WATCH_NAMES`.

## Stage 1 / Stage 2 workflow

A two-stage workflow keeps the timing-sensitive first pass cheap, then escalates
only after the first pass localizes the problem.

### Stage 1 — cheap, timing-safe scan

Find **which stage** a tracked tensor first goes bad (e.g. leaves an expected
range). Uses only the sync-free bounds check and pre-context buffer, so it is
safe to run on a timing-sensitive repro:

```bash
NANLOG_DIR=/output/layer_numerics \
NANLOG_PIPELINE=1 \
NANLOG_TRACK_ATTR=embedding_features \
NANLOG_BOUNDS=embedding_features:0:60 \
NANLOG_BAD_VALUES=1 \
NANLOG_PRE_CONTEXT=10 \
NANLOG_SAMPLE_EVERY=50 \
  python -m aorta.instrumentation.layer_numerics /path/to/your_script.py
```

Find the first out-of-range hit:

```bash
jq -c 'select(.oob_count>0) | {step,batch_id,phase,layer_name,first_bad_value}' \
  /output/layer_numerics/layers_rank0.jsonl | head
```

The `phase` field on the first hit tells you where to focus Stage 2.

### Stage 2 — targeted follow-up

Run Stage 2 as a **separate** sweep once Stage 1 names a stage. These options
add heavier GPU work or a host sync that can **hide** a timing-sensitive bug, so
never enable them alongside the Stage 1 scan.

If the first hit was at `phase=forward`, re-scan per layer to bracket the layer
(coarse stride first):

```bash
NANLOG_DIR=/output/layer_numerics_stage2 \
NANLOG_PIPELINE=1 \
NANLOG_TRACK_ATTR=embedding_features \
NANLOG_BOUNDS=embedding_features:0:60 \
NANLOG_BAD_VALUES=1 \
NANLOG_TRACK_EVERY_LAYER=1 \
NANLOG_TRACK_LAYER_STRIDE=5 \
NANLOG_WATCH_NAMES=<suspect.block> \
NANLOG_PRE_CONTEXT=10 \
  python -m aorta.instrumentation.layer_numerics /path/to/your_script.py
```

Or as a `collect:` block in a recipe:

```yaml
collect:
  layer_numerics:
    NANLOG_PIPELINE: "1"
    NANLOG_TRACK_ATTR: "embedding_features"
    NANLOG_BOUNDS: "embedding_features:0:60"
    NANLOG_BAD_VALUES: "1"
    NANLOG_TRACK_EVERY_LAYER: "1"
    NANLOG_TRACK_LAYER_STRIDE: "5"
    NANLOG_WATCH_NAMES: "<suspect.block>"
    NANLOG_PRE_CONTEXT: "10"
```

If the first hit was at a sparse stage, watch the sparse modules and capture
cheap KJT metadata:

```bash
NANLOG_DIR=/output/layer_numerics_stage2 \
NANLOG_PIPELINE=1 \
NANLOG_TRACK_ATTR=embedding_features \
NANLOG_BOUNDS=embedding_features:0:60 \
NANLOG_BAD_VALUES=1 \
NANLOG_SPARSE=1 \
NANLOG_CHANNELS=input,act \
NANLOG_WATCH_NAMES=<sparse.module.names> \
NANLOG_PRE_CONTEXT=10 \
  python -m aorta.instrumentation.layer_numerics /path/to/your_script.py
```

> `NANLOG_SPARSE_HEAVY=1` adds an index-range check that **syncs**; run it on a
> single shard as a data-sanity control only, never on the timing-sensitive
> repro.

## Output

Written under `NANLOG_DIR`:

- `summary_rank<N>.json` — the headline: the `first_bad` fingerprint
  (step / layer / direction / kind / `matmul_calls_so_far`), and when bounds are
  set `first_oob` / `oob_records` / `peak_finite_min` / `peak_finite_max`, plus
  totals.
- `layers_rank<N>.jsonl` — the full per-(layer, step, channel) trajectory. Each
  record carries `phase` and `batch_id` when pipeline tracking is on.
- `bad_tensor_step*_rank<N>.pt` — the full bad tensor (only with
  `NANLOG_DUMP_TENSOR=1`, on first detection).
- `alloc_snapshot_step*_rank<N>.pickle` — caching-allocator event trace (only
  with `NANLOG_ALLOC_SNAPSHOT=1`, on first detection).

## Analysis recipes

Find the first bad layer/step from the summary:

```bash
jq '.first_bad' /output/layer_numerics/summary_rank0.json
```

With `NANLOG_ADDR=1`, storage pointers are absolute per-process GPU virtual
addresses and are directly comparable within one rank's JSONL. To find another
record whose storage range overlaps a bad tensor's range (e.g. a buffer that was
reused), search by address overlap:

```python
import json

recs = [json.loads(l) for l in open("layers_rank0.jsonl")]

# Pick the first bad "input" record for a layer of interest.
# Replace the substring with the layer you are investigating.
bad = next(r for r in recs
           if r.get("bad") and "your.layer.substr" in r["layer_name"]
           and r.get("role") == "input")

lo = int(bad["storage_ptr"], 16)
hi = lo + bad["storage_nbytes"]
step = bad["step"]

for r in recs:
    if r is bad or r["step"] not in (step, step - 1) or "storage_ptr" not in r:
        continue
    a = int(r["storage_ptr"], 16)
    b = a + r.get("storage_nbytes", 0)
    if a < hi and lo < b:  # ranges overlap
        print(f"overlap: step={r['step']} {r['layer_name']}|{r['role']} "
              f"shape={r['shape']} ptr={r['storage_ptr']}")
```

`storage_ptr` values come from `tensor.untyped_storage().data_ptr()` — pure
host-side metadata, no GPU sync.

## Troubleshooting

**`N = 0` (no modules hooked).** The model was built before the logger armed, or
it is not reachable through the hook path. Run your script through the logger
front-end (so hooks arm first), and confirm the model is a plain `nn.Module` you
can target with `NANLOG_WATCH_NAMES` / `NANLOG_WATCH_TYPES`, or a torchrec
`DistributedModelParallel` root (auto-hooked).

**JSONL is empty.** The default channels are `act,igrad`. If you need the
forward input, set `NANLOG_CHANNELS` to include `input`. Also confirm `N > 0` in
the startup line.

**No NaN reproduced.** The bug may be intermittent — run more trials or more
steps. If a heavy Stage 2 config never reproduces but the NaN happens in
production, its overhead may be perturbing the timing; fall back to the cheap
Stage 1 config.

**Allocator snapshot missing despite a NaN.** Confirm `NANLOG_ALLOC_SNAPSHOT=1`
and check the startup log. On some PyTorch builds
`torch.cuda.memory._record_memory_history` is unavailable — the logger logs a
warning and continues.

## Sweep / collector integration

`layer_numerics` is a recognized collector name for `aorta sweep`. You can
attach it to every cell of a sweep on the CLI:

```bash
aorta sweep run --recipe <recipe>.yaml --collect layer_numerics
```

or per-recipe / per-cell via the `collect:` mapping (see
[`recipes/README.md`](../recipes/README.md#schema-rules-full-detail)):

```yaml
collect:
  layer_numerics:
    NANLOG_PIPELINE: "1"
    NANLOG_TRACK_ATTR: "embedding_features"
    NANLOG_BOUNDS: "embedding_features:0:60"
    NANLOG_PRE_CONTEXT: "10"
    NANLOG_SAMPLE_EVERY: "50"
```

> **Important — opt-in required.** The sweep engine validates the collector name
> and threads it (plus any `NANLOG_*` options) into the workload config under
> `_aorta_collect` / `_aorta_collect_options`. It does **not** launch the logger
> itself. A workload must opt in — read those keys and run its entry script
> through the logger front-end — for output to be produced. The built-in
> workloads do not currently do this, so a `collect: [layer_numerics]` recipe on
> a built-in workload will validate and run **without** producing logger output.
> Passing `--dry-run` or a successful run does **not** imply capture occurred;
> confirm the `NANLOG_DIR` artifacts exist. For a guaranteed capture today, use
> the [standalone path](#standalone-usage-supported-path).

## See also

- [`src/aorta/instrumentation/layer_numerics/README.md`](../src/aorta/instrumentation/layer_numerics/README.md)
  — developer notes and provenance.
- The module docstring in
  [`instrument_nan_logger.py`](../src/aorta/instrumentation/layer_numerics/instrument_nan_logger.py)
  — the authoritative `NANLOG_*` reference.
- [Recipes](../recipes/README.md) — the `collect:` schema.
