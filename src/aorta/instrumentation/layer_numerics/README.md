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
are drained to the host in **one batched transfer per step**, so no per-GEMM sync
is introduced and timing-sensitive behavior is preserved.

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
stage wrappers but run each copy/sparse/forward stage read on a side CUDA stream, so
the stage reads give timing-safe **stage** brackets without serializing the overlap
(summary records `stage_reads_active` — `false` means it fell back to the inline read).

Collector defaults (applied by [`build_env`](__init__.py)): all seven channels,
`NANLOG_PRE_CONTEXT=10`, `NANLOG_SAMPLE_EVERY=50`. `NANLOG_DIR` is filled per-cell
so `aorta bundle` picks up the output; recipe/CLI overrides win.

## Design invariants (do not break)

- **One host sync per step.** All per-layer reductions run on the GPU and drain in
  a single batched transfer per step — no per-GEMM `.item()`. This is what keeps a
  timing-sensitive repro reproducible; any change that adds a sync on the hot path
  defeats the tool's purpose.
- **Standalone-runnable.** No imports of the rest of `aorta`; the engine reads
  config from `NANLOG_*` env vars only (the `--config` flag and `NANLOG_SPEC_FILE`
  are thin front-ends that resolve to those vars). The script is handed to partners
  as a bare file.
- **Fail-soft.** A sidecar must never take the training job down: bad config warns
  and falls back, hooks catch their own errors.

## Notes / limitations

- `weight`/`bias` are read at the start of the step, so they hold the **previous**
  step's value (`value_is_from_prev_step=true`); `wgrad`/`bgrad` are read at the
  optimizer step boundary (current step) and need a `torch.optim.Optimizer`. If the
  update is fused into backward, a one-time warning is logged — use `weight`/`bias`.
- A stage read runs on a non-default stream; a "clean at copy" is stream-ordering
  dependent — treat it as suggestive until confirmed (see the module docstring).
- `tf32_path` is printed on the "FIRST BAD" stderr line, not stored in the summary.
