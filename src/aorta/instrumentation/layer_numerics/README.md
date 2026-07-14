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

## Notes

- **Keep it standalone-runnable.** `instrument_nan_logger.py` is run both as the
  collector and directly as a self-contained script (handoff to partners running a
  repro). Preserve that: no imports of the rest of `aorta`, configuration stays via
  `NANLOG_*` env vars only.
- It has been used to isolate a training-time NaN to a corrupted **input** arriving
  from upstream of the flagged layer (the layer's own weights/grads stayed clean),
  distinguishing an injected out-of-range value from a numerical blowup.

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
  (step / layer / direction / kind / `matmul_calls_so_far` / `tf32_path`), and
  when bounds are set `first_oob` / `oob_records` / `peak_finite_min` / `peak_finite_max`,
  plus totals.
- `layers_rank<N>.jsonl` — the full per-(layer, step, channel) trajectory. Each
  record carries `phase` and `batch_id` (see below).

## Two tracking modes

**Layer channels** (default). Hooks watched `nn.Module`s and records their
activations / grads / params — configured by `NANLOG_CHANNELS` + `NANLOG_WATCH_*`.

**Pipeline-stage tracking** (`NANLOG_PIPELINE=1`). Follows named tensors *by object*
off the TorchRec batch (default `embedding_features`) and scans them at each pipeline
stage — `copy`, `sparse_start`, `sparse_wait`, and `forward` — **before** the layers
read them. Each record is tagged with its `phase` and a `batch_id` (assigned at
`copy_batch_to_gpu`, re-read at every later stage) so one batch's records line up
across steps. This catches corruption that arrives *upstream* of the layers, which
the layer hooks alone cannot see. Needs no `WATCH_NAMES`.

## Tunables (`NANLOG_*`)

All configuration is via environment variables; the authoritative reference is the
module docstring at the top of
[`instrument_nan_logger.py`](instrument_nan_logger.py). All heavy features default
OFF and feed one batched host transfer per step (no per-GEMM sync). Most-used knobs:

| Var | Purpose | Default |
|---|---|---|
| `NANLOG_DIR` | Output dir for the JSONL + summary | `nan_logger_out` |
| `NANLOG_CHANNELS` | Layer channels (`act,input,igrad,weight,bias,wgrad,bgrad`) | `act,igrad` |
| `NANLOG_WATCH_NAMES` | Substrings matched against module paths (target a block/layer) | (empty) |
| `NANLOG_WATCH_TYPES` | Module class names to watch (e.g. `Linear`, `MoELayer`) | `Linear` (unless `WATCH_NAMES` set) |
| `NANLOG_PRE_CONTEXT` | Buffer the last K clean steps; dump on first-bad for full run-up | `0` (off) |
| `NANLOG_SAMPLE_EVERY` | Write a clean record 1 step in N (bad always written) | `50` |
| `NANLOG_HUGE_THRESHOLD` | `\|x\| > T` counts as "huge" | `1e10` |
| `NANLOG_ADDR` | Record data_ptr + backing-storage extent per tensor | `0` (off) |
| **Pipeline / bounds** | | |
| `NANLOG_PIPELINE` | Track tensors at the TorchRec stage boundaries | `0` (off) |
| `NANLOG_TRACK_ATTR` | Batch attribute(s) to follow as tracked tensors | `embedding_features` |
| `NANLOG_BOUNDS` | Per-tensor in-range check `substr:lo:hi;...` (out-of-range → `kind="oob"`) | (empty) |
| `NANLOG_BAD_VALUES` | First bad element's flat idx / row / col / value | `0` (off) |
| `NANLOG_SPARSE` | Cheap host-side KJT metadata at the sparse stage | `0` (off) |
| `NANLOG_TRACK_EVERY_LAYER` | Re-scan tracked tensors at each layer (strided) — high overhead | `0` (off) |

Collector defaults (applied by `build_env`): `NANLOG_CHANNELS` = all seven,
`NANLOG_PRE_CONTEXT=10`, `NANLOG_SAMPLE_EVERY=50`. Any `NANLOG_*` can be overridden
per-collector in a recipe's `collect:` mapping.

## Notes / limitations

- Locates the first **layer/stage/step** to go bad; it does not, by itself, prove a
  kernel-level timing race (that needs per-GEMM ordering, which would change timing).
- `weight`/`bias` channels are read at the start of the step, so they hold the
  **previous** step's value (`value_is_from_prev_step=true`); `wgrad`/`bgrad` are
  read at the optimizer step boundary (current step). Grad channels require a
  `torch.optim.Optimizer`; if the update is fused into backward, a one-time warning
  is logged — use `weight`/`bias` instead.
- A stage read runs on a non-default stream; a "clean at copy" is stream-ordering
  dependent — treat it as suggestive until confirmed (see the module docstring).
