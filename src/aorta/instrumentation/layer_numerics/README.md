# layer_numerics — per-layer NaN / magnitude logger

Workload-agnostic instrumentation sidecar that traces a NaN/Inf back to the
**layer and step where it first appears**, and captures the finite-magnitude
run-up leading to it (a value growing large → huge → NaN over steps, which a
NaN-only check cannot see).

It arms hooks on a torchrec `DistributedModelParallel` root and (optionally)
every `torch.optim.Optimizer` the moment they are built — **no edit to the
training/repro script is required**. All per-layer reductions run on the GPU and
are drained to the host in **one batched transfer per step**, so no per-GEMM sync
is introduced and timing-sensitive behavior is preserved.

## Provenance

- **Upstream:** `instrument_nan_logger.py`, the NaN logger — captures activation,
  input, grad-input, parameter value (`weight`/`bias`), and parameter gradient
  (`wgrad`/`bgrad`) channels.
- **Treat as a drop.** Do not refactor `instrument_nan_logger.py`'s hook/reduction
  logic here — the same artifact is used for standalone handoff, and
  standalone/collector parity depends on it staying in lockstep. Behavior changes
  go upstream, then re-vendor.
- **Field-proven.** It has been used to isolate a training-time NaN to a
  **corrupted input activation** arriving from upstream of the flagged layer
  (the layer's own weights/grads stayed clean), distinguishing an injected
  out-of-distribution value from a numerical blowup in the layer's parameters.

## Two ways to run

### 1. Standalone (handoff — unchanged)

A self-contained invocation for handing to a partner running the repro on their
own host. Works from a bare checkout of just the script; no `aorta` install
needed:

```bash
HIP_VISIBLE_DEVICES=0 NUM_STEPS=1000 \
  NANLOG_DIR=/output/layer_numerics \
  NANLOG_WATCH_NAMES=encoder.blocks \
  NANLOG_PRE_CONTEXT=10 \
  python instrument_nan_logger.py /path/to/standalone_single_file.py
```

From an installed `aorta` package, the same thing without needing the file path:

```bash
python -m aorta.instrumentation.layer_numerics /path/to/standalone_single_file.py
```

Both forms are byte-equivalent (`__main__.py` just `runpy`-execs the script).

### 2. As the `layer_numerics` collector (sweeps)

Attach it to every cell of a mitigation × environment sweep:

```bash
aorta sweep run --recipe <recipe>.yaml \
  --mitigations-file <sidecar>.json \
  --collect layer_numerics
```

Each cell's outputs land under `<cell>/layer_numerics/` and are included by
`aorta bundle`. The collector applies a default `NANLOG_*` bundle (see
[`build_env`](__init__.py)); recipe/CLI overrides win.

## Output

Written under `NANLOG_DIR` (the collector points this at
`<results_dir>/layer_numerics`):

- `summary_rank<N>.json` — the headline: `first_bad` fingerprint
  (step / layer / direction / kind / `matmul_calls_so_far` / `tf32_path`) plus
  totals and per-channel stats.
- `layers_rank<N>.jsonl` — the full per-(layer, step, channel) trajectory.

## Tunables (`NANLOG_*`)

All configuration is via environment variables; the authoritative reference is
the module docstring at the top of
[`instrument_nan_logger.py`](instrument_nan_logger.py). Most-used knobs:

| Var | Purpose | Default |
|---|---|---|
| `NANLOG_DIR` | Output dir for the JSONL + summary | `nan_logger_out` |
| `NANLOG_CHANNELS` | Capture channels (`act,input,igrad,weight,bias,wgrad,bgrad`) | `act,igrad` |
| `NANLOG_WATCH_NAMES` | Substrings matched against module paths (target a block/layer) | (empty) |
| `NANLOG_WATCH_TYPES` | Module class names to watch (e.g. `Linear`, `MoELayer`) | `Linear` (unless `WATCH_NAMES` set) |
| `NANLOG_PRE_CONTEXT` | Buffer the last K clean steps; dump on first-bad for full run-up | `0` (off) |
| `NANLOG_SAMPLE_EVERY` | Write a clean layer's record 1 step in N (bad always written) | `50` |
| `NANLOG_HUGE_THRESHOLD` | `\|x\| > T` counts as "huge" | `1e10` |
| `NANLOG_MAX_RECORDS` | Hard cap on records written | `200000` |
| `NANLOG_STOP_ON_FIRST` | Stop writing clean records after first bad | `0` |
| `NANLOG_VERBOSE` | One log line per step | `0` |

Collector defaults (applied by `build_env`): `NANLOG_CHANNELS` = all seven,
`NANLOG_PRE_CONTEXT=10`, `NANLOG_SAMPLE_EVERY=50`.

## Notes / limitations

- Locates the first **layer/step** to go bad; it does not, by itself, prove a
  kernel-level timing race (that needs per-GEMM ordering / a per-GEMM sync, which
  would change timing).
- `weight`/`bias` value channels are read at the start of the step, so they hold
  the **previous** step's value (`value_is_from_prev_step=true`); `wgrad`/`bgrad`
  are read at the optimizer step boundary (current step). Grad channels require a
  `torch.optim.Optimizer`; if the update is fused into backward, a one-time
  warning is logged — use `weight`/`bias` instead.
