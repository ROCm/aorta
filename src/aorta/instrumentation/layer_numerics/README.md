# layer_numerics — per-layer NaN / magnitude logger

> User-facing how-to (standalone usage, options, Stage 1/2 workflow, analysis,
> troubleshooting): [`docs/layer-numerics.md`](../../../../docs/layer-numerics.md).
> This file is developer notes + provenance.

Workload-agnostic instrumentation sidecar that traces a NaN/Inf back to the
**layer and step where it first appears**, and captures the finite-magnitude
run-up leading to it (a value growing large → huge → NaN over steps, which a
NaN-only check cannot see).

It arms hooks on a torchrec `DistributedModelParallel` root and (optionally)
every `torch.optim.Optimizer` the moment they are built — **no edit to the
training/repro script is required**. All per-layer reductions run on the GPU and
completed records are drained in a batched transfer per device. Incomplete timing-safe
stage observations are deferred rather than blocking the next forward, so no per-GEMM
sync is introduced.

Provenance: used in anger to isolate a training-time NaN to a corrupted **input**
arriving from upstream of the flagged layer (the layer's own weights/grads stayed
clean) — distinguishing an injected out-of-range value from a numerical blowup.

## Two ways to run

### 1. Standalone (handoff — unchanged)

A self-contained invocation for handing to a partner running the repro on their
own host. Works from a bare checkout of just the script; no `aorta` install
needed:

```bash
HIP_VISIBLE_DEVICES=0 NUM_STEPS=1000 \
  NANLOG_DIR=/output/layer_numerics \
  NANLOG_SPEC='{"watch":[{"scope":{"names":["encoder.blocks"]},"tensors":["output"]}],"pre_context":10}' \
  python instrument_nan_logger.py /path/to/standalone_single_file.py
```

From an installed `aorta` package, the same thing without needing the file path:

```bash
python -m aorta.instrumentation.layer_numerics /path/to/standalone_single_file.py
```

Both forms are byte-equivalent (`__main__.py` just `runpy`-execs the script).

#### Buck `.par` integration

A Buck Python binary owns `__main__`, so the logger front-end cannot wrap it with
`runpy`. Import the sidecar before constructing the model/pipeline and install it
explicitly:

```python
import atexit
import instrument_nan_logger as nanlog

nanlog._install_autohook()
nanlog._install_optimizer_autohook()
nanlog._install_pipeline_hook()
atexit.register(nanlog._write_summary)
```

Set `NANLOG_SPEC_FILE` / `NANLOG_SPEC` and `NANLOG_DIR` **before** importing the
module; configuration is resolved at import time.

For a C/C+ handoff, the 30-second artifact check is:

```bash
jq '{spec_applied,pipeline_installed,follow_mode,stage_evidence_valid,
     stage_missing_phases,stage_skip_reasons,stage_phase_counts,
     post_step_active,post_step_valid}' "$NANLOG_DIR/summary_rank0.json"
```

Do not interpret the phase unless `follow_mode` is
`stage_wrappers_side_read`, `stage_evidence_valid` is true, required phases are
present with trusted observations, and skip reasons are empty. C+ additionally
requires active/valid post-step observations.

### 2. As the `layer_numerics` collector (sweeps)

Request it on a mitigation × environment sweep:

```bash
aorta sweep run --recipe <recipe>.yaml \
  --mitigations-file <sidecar>.json \
  --collect layer_numerics
```

**Opt-in required.** The sweep engine validates the collector name and threads
it (plus any `NANLOG_*` options) into the workload config under
`_aorta_collect` / `_aorta_collect_options`; it does **not** launch the logger
itself. A workload must read those keys and run its entry through the logger for
output to be produced. The built-in workloads do not do this, so `--collect
layer_numerics` on a built-in workload validates and runs without producing
logger output — use the standalone path above for a guaranteed capture. A
workload that opts in can point the logger at `<cell>/layer_numerics/` (via
[`build_env`](__init__.py), which fills `NANLOG_DIR` and a default `NANLOG_*`
bundle) so outputs are picked up by `aorta bundle`; recipe/CLI overrides win.

## Configuration

All configuration is `NANLOG_*` environment variables — `NANLOG_SPEC` (structured,
recommended) or the flat vars it maps to. The how-to, the full option list, and
worked Stage 1 / Stage 2 examples live in the user doc:
[`docs/layer-numerics.md`](../../../../docs/layer-numerics.md). The authoritative,
exhaustive reference is the module docstring at the top of
[`instrument_nan_logger.py`](instrument_nan_logger.py). Don't duplicate the option
table here — keep it in one place.

The spec JSON can be supplied three ways, in order of precedence: the `--config
<path>` CLI flag beats `NANLOG_SPEC_FILE=<path>` (env), which beats inline
`NANLOG_SPEC` (env). The summary JSON records which won in a `spec_source` field.
See [`docs/layer-numerics.md`](../../../../docs/layer-numerics.md) for the how-to.

`NANLOG_SPEC` is a thin front-end: at import it validates the JSON and translates
it into the flat vars the engine already reads (a malformed spec warns and falls
back). Keeping the engine flat-var-only is what preserves the standalone contract.

Recent schema additions (full how-to in [`docs/layer-numerics.md`](../../../../docs/layer-numerics.md)):
a `watch` entry's `tensors` list also accepts `igrad` (the activation-input grad
alone, vs. `grad` which is all gradients); a `watch` entry may set `"stride": N`
(`>= 1`, default `1`) to hook only every Nth matched module; a top-level
`"diagnostics": [...]` list (`addr`, `locate`, `bad_values`, `dump_tensor`,
`alloc_snapshot`) toggles per-record detail across every captured record; and a
`follow` entry may set `"pipeline": false` to follow the tensor through its `scope`
blocks (and at forward entry) **without** installing the copy/sparse stage wrappers
— the timing-safe mode for a cross-stream race the wrappers would otherwise suppress
(requires a `scope`, incompatible with `stages: true`; summary records `follow_mode`);
and a `follow` entry may set `"stage_reads": true` (with `pipeline: true`) to KEEP the
stage wrappers. Copy/sparse observations use side streams after explicit TorchRec
producer events; forward entry is compute-stream ordered before the model body, matching
Pass A. Unresolved upstream CUDA observations are skipped and audited — they never fall
back inline. Optional `"post_step": true` adds a default-off C+ observation of live
prefetch buffers after a compute tick.

Collector defaults (applied by [`build_env`](__init__.py)): all seven channels,
`NANLOG_PRE_CONTEXT=10`, `NANLOG_SAMPLE_EVERY=50`. `NANLOG_DIR` is filled per-cell
so `aorta bundle` picks up the output; recipe/CLI overrides win.

## Design invariants (do not break)

- **Never wait for a live producer observation at forward entry.** The root pre-hook
  queries side-read completion and defers incomplete records; shutdown forces the
  remaining queue. Completed records drain in one batched transfer per device, with no
  per-GEMM `.item()`.
- **Standalone-runnable.** No imports of the rest of `aorta`; the engine reads
  config from `NANLOG_*` env vars only (the `--config` flag and `NANLOG_SPEC_FILE`
  are thin front-ends that resolve to those vars). The script is handed to partners
  as a bare file.
- **Fail-soft.** A sidecar must never take the training job down: bad config warns
  and falls back, hooks catch their own errors. Timing evidence is stricter:
  unresolved producer/context/device/side-stream state skips the checkpoint and sets
  `stage_evidence_valid=false` rather than silently using an inline read.

## Notes / limitations

- `weight`/`bias` are read at the start of the step, so they hold the **previous**
  step's value (`value_is_from_prev_step=true`); `wgrad`/`bgrad` are read at the
  optimizer step boundary (current step) and need a `torch.optim.Optimizer`. If the
  update is fused into backward, a one-time warning is logged — use `weight`/`bias`.
- Producer lookup currently supports the standard TorchRec `_memcpy_stream` and
  `_data_dist_stream` contracts. Check `stage_evidence_valid`, per-phase counts, and
  skip reasons on every customer artifact.
- A stage observation is not an atomic snapshot. Use only `trusted` records; an
  `overlapped_next_stage` record was still running when the next same-batch stage began.
- Even a trusted phase is a wall-clock bracket, not proof that the writer logically
  belongs to that stage: three pipeline batches overlap. Use `pipeline_tick`,
  `compute_batch_id`, and optional post-step observations for that distinction.
- `tf32_path` is printed on the "FIRST BAD" stderr line, not stored in the summary.
