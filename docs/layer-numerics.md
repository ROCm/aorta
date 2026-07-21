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

This is a **localization** step, not a detector: it assumes you already have a
way to reproduce the NaN/Inf (a failing training run, a `aorta sweep`/`probe`
repro, a customer's crash log) and answers *where it starts*, not *whether it
happens*. Reach for it after a repro is in hand and before you commit to a
mitigation.

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
- `torchrec`, with your model wrapped in `DistributedModelParallel` (DMP) —
  the automatic, no-script-edit hook attach only fires when a DMP root is
  constructed (pipeline-stage tracking, `NANLOG_PIPELINE=1`, also needs
  `torchrec`, for the same reason). **A plain `nn.Module` script is not
  auto-hooked today** — see [Troubleshooting](#troubleshooting) if you need to
  attach to one anyway.
- No extra install beyond that — `instrument_nan_logger.py` has no other
  dependencies, and all configuration is through `NANLOG_*` environment variables.

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

## Configuration (`NANLOG_SPEC`, recommended)

The clearest way to configure the logger is a single structured JSON value,
`NANLOG_SPEC`. It unifies the many flat flags onto a small set of axes and two
observation kinds:

- **`watch`** — observe a module's **own** tensors.
- **`follow`** — trace **one named tensor** across positions.

| Axis | Question | Values |
|---|---|---|
| `scope` | which modules | `names: [substr,…]` and/or `types: [ClassName,…]` |
| `tensors` | which of a module's tensors | `input, output, weight, bias, igrad, grad` |
| `stages` | (follow) check at the pipeline stage boundaries | `true` (`copy`/`sparse_start`/`sparse_wait`/`forward`) |
| `stride` | (follow) also check every Nth module in `scope` | `N` (default `1`; requires `scope`). Set `stages`, `scope`, or both; neither ⇒ stages only |
| `pipeline` | (follow) install the copy/sparse stage wrappers | `true` (default) / `false` — `false` follows at forward entry + `scope` blocks only, **without** the stage wrappers (timing-safe; see below). Requires a `scope`; incompatible with `stages: true` |
| `stride` | (watch) hook only every Nth matched module | `N` (integer `>= 1`, default `1`); thins a broad watch whose per-module reduction volume is too high |
| `diagnostics` | (top-level) how-much-detail toggles applied to **every** captured record | `addr, locate, bad_values, dump_tensor, alloc_snapshot` (see below) |

```bash
NANLOG_SPEC='{
  "watch":  [{"scope": {"types": ["MLP","AttentionBlock"]}, "tensors": ["input","output"]}],
  "follow": [{"tensor": "embedding_features", "stages": true, "bounds": [0,60]}],
  "sample_every": 50, "pre_context": 10
}' \
  python -m aorta.instrumentation.layer_numerics /path/to/your_script.py
```

Common patterns (each cell is a complete, copy-paste `NANLOG_SPEC` value):

| Goal | Spec |
|---|---|
| Standard tensors by block | `{"watch":[{"scope":{"types":["AttentionBlock","MLP"]},"tensors":["input","output","weight","bias","grad"]}]}` |
| Selected tensors from one scope | `{"watch":[{"scope":{"names":["emb_proj"]},"tensors":["input","output"]}]}` |
| Activation-input grad only (no param grads) | `{"watch":[{"scope":{"types":["Linear"]},"tensors":["igrad"]}]}` |
| Thin a broad watch to every 4th Linear | `{"watch":[{"scope":{"types":["Linear"]},"tensors":["output"],"stride":4}]}` |
| Extra per-record detail via diagnostics | `{"watch":[{"scope":{"types":["MLP"]},"tensors":["output"]}],"diagnostics":["addr","bad_values"]}` |
| Follow a tensor through the pipeline | `{"follow":[{"tensor":"embedding_features","stages":true}]}` |
| Follow a tensor every N layers | `{"follow":[{"tensor":"embedding_features","scope":{"names":["emb_proj"]},"stride":8}]}` |
| Follow a tensor at named layers only | `{"follow":[{"tensor":"embedding_features","scope":{"names":["emb_proj.projections.0"]}}]}` |
| Follow through blocks WITHOUT the stage wrappers (timing-safe) | `{"follow":[{"tensor":"embedding_features","pipeline":false,"scope":{"names":["emb_proj"]},"bounds":[0,60]}]}` |

When `NANLOG_SPEC` is set it **wins**; when it is unset, the flat `NANLOG_*`
vars below are read directly (still fully supported). A malformed spec logs a
warning and falls back to the flat vars — it never takes the run down.

### Supplying the spec from a file

Inlining a large JSON blob in an env var is awkward. The same spec JSON can come
from a file instead, in two ways:

- `--config <path>` (or `--config=<path>`) — a CLI flag to the logger. The flag
  and its value are stripped from argv before your script runs, so the target
  never sees them.
- `NANLOG_SPEC_FILE=<path>` — an env var pointing to the JSON file.

Put the spec in `spec.json`:

```json
{
  "follow": [{"tensor": "embedding_features", "stages": true, "bounds": [0, 60]}],
  "pre_context": 10,
  "sample_every": 50
}
```

then point the logger at it with the CLI flag:

```bash
NANLOG_DIR=/output/layer_numerics \
  python -m aorta.instrumentation.layer_numerics --config spec.json /path/to/your_script.py
```

or via the env var:

```bash
NANLOG_DIR=/output/layer_numerics \
NANLOG_SPEC_FILE=spec.json \
  python -m aorta.instrumentation.layer_numerics /path/to/your_script.py
```

Precedence when more than one is set: `--config` (CLI) beats `NANLOG_SPEC_FILE`
(env) beats inline `NANLOG_SPEC` (env). A config file that can't be read logs a
warning and falls back to the flat `NANLOG_*` vars — it never takes the run down.
The output dir stays a **separate** `NANLOG_DIR` — it is not read from the config
file. The summary JSON records which input won in a `spec_source` field
(`"NANLOG_SPEC"`, `"NANLOG_SPEC_FILE=<path>"`, or `"--config=<path>"`).

## The two things you can capture

Everything the logger does is one of two kinds — the `watch` and `follow` keys of
`NANLOG_SPEC` map directly onto them:

- **`watch` — a module's own tensors.** Hooks the modules you name (by class
  `types` or path `names`) and records their `input` / `output` / `weight` /
  `bias` / `igrad` / `grad`. This is the "which layer went bad" view. `grad` is
  the umbrella for **all** gradients (input-grad + both param grads); `igrad` is
  the input-grad piece alone, for when you want the activation-input gradient
  without the weight/bias grads. Add `"stride": N` (integer `>= 1`, default `1`)
  to hook only every Nth matched module in traversal order — a way to thin a
  broad watch whose per-module reduction volume is too high (e.g.
  `{"scope":{"types":["Linear"]},"tensors":["output"],"stride":4}` hooks every
  4th `Linear`). This watch `stride` is separate from a `follow` entry's own
  `stride`. If the same spec also uses a scoped `follow`, the logger **ignores**
  `watch[].stride` and warns, because the follow re-scan runs inside those module
  hooks and thinning them would drop follow evidence.
- **`follow` — one named tensor across positions.** Follows a batch tensor
  (default `embedding_features`) and re-checks it at the TorchRec pipeline stage
  boundaries (`stages: true` → `copy`, `sparse_start`, `sparse_wait`, `forward`)
  and/or at each module in a `scope` (`stride: N` for every Nth match, default
  `1`). Set `stages`, `scope`, or both — with neither, it checks stages only.
  Each record is tagged with its `phase` and a `batch_id` so one batch's records
  line up across steps. This catches corruption that arrives *upstream* of the
  layers, which watching alone cannot see.
  > Note: by default a scoped re-scan rides the pipeline stage wrappers, so it
  > **also** emits the stage-boundary records even when `stages` is omitted or
  > `false` — you will see `phase` records at the stages regardless. To drop the
  > stage wrappers entirely, set `pipeline: false` (next).

#### `pipeline: false` — follow through blocks without the stage wrappers

By default any `follow` installs the TorchRec stage-method wrappers
(`copy_batch_to_gpu` / `start_sparse_data_dist` / `wait_sparse_data_dist`) — that
is what produces the `copy`/`sparse_start`/`sparse_wait` records. Those wrappers
**serialize the overlapped copy/compute** the batch pipeline runs, which can
**suppress a timing-sensitive cross-stream race** (the buffer never gets corrupted
because the overlap the race needs is gone). If a `follow`/`stages` sweep comes
back with **0 NaN on a bug you can otherwise reproduce**, this is the likely cause.

Set `"pipeline": false` on the follow entry to keep the per-block re-scan (at each
module in `scope`) and the forward-entry checkpoint, but **not** install the stage
wrappers. The re-scan rides the ordinary forward hooks — the same timing-safe
mechanism `watch` uses — so the copy/compute overlap is preserved and the race
still fires:

```bash
NANLOG_DIR=/output/layer_numerics \
NANLOG_SPEC='{"follow":[{"tensor":"embedding_features","pipeline":false,"scope":{"names":["emb_proj"]},"bounds":[0,60]}],"pre_context":10}' \
  python -m aorta.instrumentation.layer_numerics /path/to/your_script.py
```

Constraints: `pipeline: false` **requires a `scope`** (there are no stages left to
capture at, so a block scope is what it checks) and is **incompatible with
`stages: true`** (stage capture *is* the wrappers) — either combination rolls the
spec back. What you give up vs. the default: the `copy` / `sparse_start` /
`sparse_wait` stage records (the upstream-of-forward checkpoints). What you keep:
the block-by-block trajectory of the followed tensor. The summary records
`follow_mode` (`stage_wrappers` / `forward_blocks` / `off`) so a run is auditable.

Add a `bounds: [lo, hi]` to a `follow` entry to flag out-of-range values
(`kind="oob"`) — a two-sided check the one-sided "huge" threshold misses.

### Diagnostics — how much detail per record

A top-level `"diagnostics": [...]` list adds "how much detail" toggles that
apply to **every** captured record, independent of `watch` / `follow`. Valid
names:

- `addr` — GPU `data_ptr` + backing-storage extent per tensor.
- `locate` — how many rows hold a bad value.
- `bad_values` — the first bad element's flat idx / row / col / value.
- `dump_tensor` — save the full bad tensor to a `.pt` file (one-shot, on first
  detection).
- `alloc_snapshot` — caching-allocator event trace on first bad (~10% overhead).

These correspond one-to-one to the flat `NANLOG_ADDR` / `NANLOG_LOCATE` /
`NANLOG_BAD_VALUES` / `NANLOG_DUMP_TENSOR` / `NANLOG_ALLOC_SNAPSHOT` vars. The
output dir stays env-only (`NANLOG_DIR`) — it is **not** a diagnostic.

```json
{
  "watch": [{"scope": {"types": ["MLP"]}, "tensors": ["output"]}],
  "diagnostics": ["addr", "bad_values"]
}
```

## Stage 1 / Stage 2 workflow

A two-stage workflow keeps the timing-sensitive first pass cheap, then escalates
only after the first pass localizes the problem.

### Stage 1 — cheap, timing-safe scan

Find **which stage** a tracked tensor first goes bad (e.g. leaves an expected
range). Follows the tensor at each pipeline stage with only the sync-free bounds
check and pre-context buffer, so it is safe to run on a timing-sensitive repro:

```bash
NANLOG_DIR=/output/layer_numerics \
NANLOG_SPEC='{"follow":[{"tensor":"embedding_features","stages":true,"bounds":[0,60]}],"pre_context":10,"sample_every":50}' \
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
(coarse stride first — `"stride":5` re-checks every 5th module in the suspect
block):

```bash
NANLOG_DIR=/output/layer_numerics_stage2 \
NANLOG_SPEC='{"follow":[{"tensor":"embedding_features","scope":{"names":["<suspect.block>"]},"stride":5,"bounds":[0,60]}],"pre_context":10}' \
  python -m aorta.instrumentation.layer_numerics /path/to/your_script.py
```

Or as a `collect:` block in a recipe:

```yaml
collect:
  layer_numerics:
    NANLOG_SPEC: '{"follow":[{"tensor":"embedding_features","scope":{"names":["<suspect.block>"]},"stride":5,"bounds":[0,60]}],"pre_context":10}'
```

If the first hit was at a sparse stage, watch the sparse modules' own tensors and
capture cheap KJT metadata (the `NANLOG_SPARSE` knob has no `NANLOG_SPEC` axis —
add it as a flat var alongside the spec):

```bash
NANLOG_DIR=/output/layer_numerics_stage2 \
NANLOG_SPEC='{"watch":[{"scope":{"names":["<sparse.module.names>"]},"tensors":["input","output"]}],"follow":[{"tensor":"embedding_features","stages":true,"bounds":[0,60]}],"pre_context":10}' \
NANLOG_SPARSE=1 \
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
  totals. For a `follow`, it also records **how** the tensor was tracked so a run
  is auditable:
  - `follow_mode` — `"stage_wrappers"` (stage boundaries + any blocks, the
    default), `"forward_blocks"` (a `pipeline:false` follow: forward entry + blocks,
    **no** stage wrappers), or `"off"` (no follow).
  - `pipeline` / `pipeline_installed` — whether the copy/sparse stage wrappers were
    requested / actually installed. Both `false` in a `pipeline:false` run.
  - `pipeline_checkpoints` — count of **stage-wrapper** checkpoints
    (`copy`/`sparse_start`/`sparse_wait`) that captured the tensor. Always `0` in a
    `pipeline:false` run (no wrappers ran) — so a nonzero value here is proof the
    stage instrumentation was active.
  - `forward_checkpoints` — count of **forward-entry** checkpoints (the read of the
    tensor at the top of the forward pass, which rides the root pre-hook, not the
    stage wrappers). This is the "yes, the pipeline-off follow captured something"
    signal for a `forward_blocks` run, kept separate from `pipeline_checkpoints` so
    a timing-safe run never looks like it used stage instrumentation.
- `layers_rank<N>.jsonl` — the full per-(layer, step, channel) trajectory. Each
  record carries `phase` and `batch_id` when a follow is active (`phase="forward"`
  for the forward-entry checkpoint; `checkpoint=<block name>` for a per-block
  re-scan).
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

**`N = 0` (no modules hooked).** Today the automatic attach only fires for a
torchrec `DistributedModelParallel` (DMP) root — a plain `nn.Module` script is
**not** auto-hooked no matter what `NANLOG_WATCH_NAMES` / `NANLOG_WATCH_TYPES`
you set. If your script does build a DMP root: confirm you ran it *through* the
logger front-end (so hooks arm before the model is built, not after), and check
the startup log line for `watched modules:` to see whether your filters actually
matched anything.

To hook a plain `nn.Module` (no torchrec), you need a small script edit: import
the logger module and call its private `_attach(model)` right after building the
model (see `instrument_nan_logger.py`). This is not a stable public API, but it
reuses the same hook/drain machinery as the DMP path.

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
    NANLOG_SPEC: '{"follow":[{"tensor":"embedding_features","stages":true,"bounds":[0,60]}],"pre_context":10,"sample_every":50}'
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

## Reference: flat `NANLOG_*` vars

`NANLOG_SPEC` (above) is the recommended interface. Under the hood it translates
into the flat `NANLOG_*` environment variables below, which you can also set
directly — useful for a bare-checkout handoff, or for the few knobs that have no
`NANLOG_SPEC` axis (`NANLOG_SPARSE`, `NANLOG_ADDR`, `NANLOG_DUMP_TENSOR`,
`NANLOG_ALLOC_SNAPSHOT`). If both are set, `NANLOG_SPEC` wins. The authoritative,
exhaustive reference is the module docstring at the top of
[`instrument_nan_logger.py`](../src/aorta/instrumentation/layer_numerics/instrument_nan_logger.py).
All heavy features default **OFF** and feed the same single per-step host transfer
(no added sync).

| Var | Purpose | Default | `NANLOG_SPEC` equivalent |
|---|---|---|---|
| `NANLOG_DIR` | Output dir for the JSONL + summary | `nan_logger_out` | — (env-only; `dir` is **not** a spec key) |
| `NANLOG_SPEC_FILE` | Path to a JSON file whose contents are used as the spec (`--config` overrides it) | (empty) | (whole spec) |
| `NANLOG_CHANNELS` | Capture channels: `act,input,igrad,weight,bias,wgrad,bgrad` | `act,igrad` | `watch[].tensors` |
| `NANLOG_WATCH_NAMES` | Substrings matched against module paths | (empty) | `scope.names` |
| `NANLOG_WATCH_TYPES` | Module class names to watch | `Linear` (unless `WATCH_NAMES` set) | `scope.types` |
| `NANLOG_PRE_CONTEXT` | Buffer the last K clean steps; dump on first-bad | `0` (off) | `pre_context` |
| `NANLOG_SAMPLE_EVERY` | Write a clean record 1 step in N (bad always written) | `50` | `sample_every` |
| `NANLOG_HUGE_THRESHOLD` | `\|x\| > T` counts as "huge" | `1e10` | — |
| `NANLOG_BAD_VALUES` | Record the first bad element's flat idx / row / col / value | `0` (off) | `diagnostics:[bad_values]` |
| `NANLOG_ADDR` | Record `data_ptr` + backing-storage extent per tensor | `0` (off) | `diagnostics:[addr]` |
| `NANLOG_LOCATE` | Record how many rows hold a bad value | `0` (off) | `diagnostics:[locate]` |
| `NANLOG_DUMP_TENSOR` | Save the full bad tensor to a `.pt` on first detection | `0` (off) | `diagnostics:[dump_tensor]` |
| `NANLOG_ALLOC_SNAPSHOT` | Dump a caching-allocator event trace on first bad (~10% overhead) | `0` (off) | `diagnostics:[alloc_snapshot]` |
| `NANLOG_PIPELINE` | Install the TorchRec stage-method wrappers (copy/sparse checkpoints) | `0` (off) | `follow[].stages: true`; suppressed by `follow[].pipeline: false` |
| `NANLOG_TRACK_ATTR` | Batch attribute(s) to follow as tracked tensors | `embedding_features` | `follow[].tensor` |
| `NANLOG_BOUNDS` | Per-tensor in-range check `substr:lo:hi;...` (→ `kind="oob"`) | (empty) | `follow[].bounds` |
| `NANLOG_SPARSE` | Cheap host-side KJT metadata at the sparse stage | `0` (off) | — |
| `NANLOG_TRACK_EVERY_LAYER` | Re-scan tracked tensors at each scoped block. Rides an **active follow** — with `NANLOG_PIPELINE=1` (stage+block). **Setting this flat var alone (no `NANLOG_PIPELINE`, no spec) is a warned no-op**, so a legacy run is never silently switched into forward-capture mode; the wrapper-free block follow is opt-in via a `pipeline:false` spec. | `0` (off) | `follow[].scope` (+ `stride`) |
| `NANLOG_PIPELINE_OFF_FOLLOW` | **Spec-internal, not a public flat knob.** Set only by a validated `follow[].pipeline:false`; enables the forward+block follow without the stage wrappers. Documented for artifact readers, not for direct use. | `0` (off) | `follow[].pipeline: false` |
| `NANLOG_TRACK_LAYER_STRIDE` | Re-scan every Kth layer when re-scan is on | `1` | `follow[].stride` |

Channel notes:

- `weight`/`bias` are read at the **start** of the step, so they hold the
  **previous** step's value (`value_is_from_prev_step=true`).
- `wgrad`/`bgrad` are read at the optimizer step boundary (current step) and
  require a `torch.optim.Optimizer`. If the update is fused into backward, a
  one-time warning is logged — use `weight`/`bias` instead.

## See also

- [`src/aorta/instrumentation/layer_numerics/README.md`](../src/aorta/instrumentation/layer_numerics/README.md)
  — developer notes and provenance.
- The module docstring in
  [`instrument_nan_logger.py`](../src/aorta/instrumentation/layer_numerics/instrument_nan_logger.py)
  — the authoritative `NANLOG_*` reference.
- [Recipes](../recipes/README.md) — the `collect:` schema.
